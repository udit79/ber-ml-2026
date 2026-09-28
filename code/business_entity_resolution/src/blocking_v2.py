"""
blocking_v2.py — Hybrid candidate generation for Business Entity Resolution.

Architecture (three independent retrieval layers, unioned per S1 entity):

  Layer A  Deterministic exact blocking
           13 composite key passes (name-core+city, street-number+city, …)
           O(N) lookup via vectorised pandas groupby.

  Layer B  Inverted character n-gram retrieval
           Build a posting-list index over 3-char n-grams of normalised names
           and addresses.  For each S1 query, select its rarest n-grams,
           retrieve the union of their posting lists, score the resulting
           reduced candidate set with sparse TF-IDF cosine — never a full
           (n_queries × n_pool) matrix multiply.

  Layer C  GPU pairwise scoring (optional re-rank of B output)
           Sends only the Layer-B candidate pairs to PyTorch.
           Fixed batches, FP16 intermediate / FP32 accumulation,
           explicit VRAM monitoring, automatic CPU fallback.

Cache layer
  Polars streaming normalisation → requested-country Parquet with
  source_mtime / normalizer_version / schema_version / country-set stamps.
  Second run loads cache instantly.

Checkpoint layer
  Each run is content-addressed by source mtimes, query IDs, parameters, and
  strategy version. Layer-A, Layer-B, and final scored country results are
  written atomically, so an interrupted run never reuses a stale experiment.

Output
  output/candidates/v2_hybrid/candidate_pairs.tsv  (final blocking output)

Run from the repository root:
    $env:SAMPLE="0.05"
    python code/business_entity_resolution/src/blocking_v2.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

# ── PATH fix so spawned workers (if any) can load torch DLLs on Windows ──────
_ENV_ROOT  = os.path.dirname(sys.executable)
_TORCH_LIB = os.path.join(_ENV_ROOT, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_TORCH_LIB):
    os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402
from tqdm import tqdm  # noqa: E402

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)
from preprocess import (  # noqa: E402
    _ADDR_EXPANSIONS,
    _NAME_EXPANSIONS,
    _PUNCT_RE,
    _WS_RE,
    normalize_name,
    normalize_address,
)

# ── GPU probe ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    _DEVICE    = torch.device("cuda")
    _GPU       = True
    _FREE_VRAM = torch.cuda.mem_get_info(0)[0]
    print(f"[v2] GPU: {torch.cuda.get_device_name(0)}  "
          f"free VRAM: {_FREE_VRAM/1e9:.1f} GB", flush=True)
else:
    _DEVICE    = torch.device("cpu")
    _GPU       = False
    _FREE_VRAM = 0
    print("[v2] No CUDA — CPU only", flush=True)

# ── Version stamps ─────────────────────────────────────────────────────────────
NORMALIZER_VERSION = "v2"
SCHEMA_VERSION     = "2"
STRATEGY_VERSION   = "v2_hybrid_exact_gpu_v2"
INDEX_VERSION      = "ngram-index-v1"

# ── Configuration ──────────────────────────────────────────────────────────────
MAX_CANDIDATES  = 200       # hard cap after all layers are unioned
CHUNK_SIZE      = 500_000   # Polars streaming batch size
MAX_FEATURES    = 60_000    # TF-IDF vocab cap for Layer B scoring
SPARSE_SCORE_BATCH = 256    # query rows per bounded sparse scoring tile
SPARSE_FIT_DOCS = 300_000   # deterministic fit corpus cap per field
NGRAM_SIZE      = 3         # character n-gram length for inverted index
TOPK_NGRAMS     = 8         # rarest n-grams selected per query
MAX_POSTINGS    = 0         # 0 = retain all postings; query chooses rare grams
MAX_QUERY_CANDIDATES = 500 # bounded shortlist before local TF-IDF scoring
# Sparse products can become dense for common grams. Keep this conservative
# on laptop RAM; the query loop still runs in compiled SciPy kernels.
NGRAM_QUERY_BATCH = int(os.environ.get("NGRAM_QUERY_BATCH", "8"))
LAYER_B_CAP     = 150       # max candidates from Layer B per query
GPU_Q_BATCH     = 512       # query rows per GPU scoring batch
GPU_I_BATCH     = 10_000    # pool rows per GPU tile
N_WORKERS       = min(12, os.cpu_count() or 4)

_ROOT           = os.path.abspath(os.path.join(_SRC_DIR, "..", "..", ".."))
CACHE_ROOT      = os.path.join(_ROOT, "output", "cache")
CHECKPOINT_ROOT = os.path.join(_ROOT, "output", "checkpoints", STRATEGY_VERSION)
OUTPUT_DIR      = os.path.join(_ROOT, "output", "candidates", STRATEGY_VERSION)
INDEX_CACHE_ROOT = os.path.join(CACHE_ROOT, "ngram_indexes", INDEX_VERSION)


# ══════════════════════════════════════════════════════════════════════════════
# Helper text functions
# ══════════════════════════════════════════════════════════════════════════════

_STOPWORDS = {
    "the", "and", "of", "for", "a", "an",
    "company", "corporation", "limited", "private", "llp", "llc",
    "incorporated", "sa", "sarl", "sasu",
}


def _first_token(text: str) -> str:
    parts = text.split()
    return parts[0] if parts else ""


def _first_meaningful_token(text: str) -> str:
    return next(
        (t for t in text.split() if len(t) >= 3 and t not in _STOPWORDS),
        "",
    )


def _prefix_key(text: str) -> str:
    tokens = text.split()
    return " ".join(t[:2] for t in tokens[:2] if len(t) >= 2)


def _name_core(norm_name: str) -> str:
    """Strip legal suffixes for a tighter bucket key."""
    import re
    core = re.sub(
        r"\b(?:limited|llc|private|incorporated|corporation|company|llp|sarl|sasu|sa)\b",
        "",
        norm_name,
    )
    return re.sub(r"\s+", " ", core).strip()


def _addr_number(norm_addr: str) -> str:
    import re
    m = re.search(r"\b(\d+[a-z]?)\b", norm_addr)
    return m.group(1) if m else ""


def _addr_region(norm_addr: str) -> str:
    parts = norm_addr.split()
    return parts[-1] if parts else ""


def _addr_city(raw_addr: str) -> str:
    """Last comma-separated segment, normalised."""
    parts = raw_addr.split(",")
    return normalize_address(parts[-1]) if parts else ""


def _char_ngrams(text: str, n: int = NGRAM_SIZE) -> List[str]:
    """Extract character n-grams from text."""
    return [text[i:i+n] for i in range(len(text) - n + 1)] if len(text) >= n else []


# ══════════════════════════════════════════════════════════════════════════════
# Parallel normalisation (thread-based — no DLL reload on Windows)
# ══════════════════════════════════════════════════════════════════════════════

def _norm_slice(args: Tuple) -> Tuple:
    names, addrs = args
    norm_names  = [normalize_name(n)    for n in names]
    norm_addrs  = [normalize_address(a) for a in addrs]
    return norm_names, norm_addrs


def _normalize_df(df: pd.DataFrame, n_workers: int = 1) -> pd.DataFrame:
    """Add norm_name / norm_addr / derived key columns in parallel threads."""
    df = df.copy()
    n  = len(df)
    names = df["business_name"].fillna("").tolist()
    addrs = df["business_address"].fillna("").tolist()

    if n_workers <= 1 or n < 5_000:
        norm_names = [normalize_name(x)    for x in names]
        norm_addrs = [normalize_address(x) for x in addrs]
    else:
        step   = max(1, (n + n_workers - 1) // n_workers)
        slices = [(names[i:i+step], addrs[i:i+step]) for i in range(0, n, step)]
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_norm_slice, slices))
        norm_names, norm_addrs = [], []
        for nn, na in results:
            norm_names.extend(nn)
            norm_addrs.extend(na)

    df["norm_name"]      = norm_names
    df["norm_addr"]      = norm_addrs
    df["name_core"]      = [_name_core(n)             for n in norm_names]
    df["meaningful_tok"] = [_first_meaningful_token(n) for n in norm_names]
    df["first_tok"]      = [_first_token(n)            for n in norm_names]
    df["prefix_key"]     = [_prefix_key(n)             for n in norm_names]
    df["addr_number"]    = [_addr_number(a)             for a in norm_addrs]
    df["addr_region"]    = [_addr_region(a)             for a in norm_addrs]
    df["addr_city"]      = [_addr_city(r)
                            for r in df["business_address"].fillna("").tolist()]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Cache layer — Polars streaming → country-partitioned Parquet
# ══════════════════════════════════════════════════════════════════════════════

def _polars_normalize_expr(expr: pl.Expr, expansions: list) -> pl.Expr:
    out = expr.fill_null("").str.normalize("NFKC").str.to_lowercase()
    for pattern, replacement in expansions:
        out = out.str.replace_all(pattern, replacement, literal=False)
    return (
        out.str.replace_all(_PUNCT_RE.pattern, " ", literal=False)
        .str.replace_all(_WS_RE.pattern, " ", literal=False)
        .str.strip_chars()
    )


def _cache_stamp(path: str, target_countries: set[str]) -> str:
    return (
        f"normalizer={NORMALIZER_VERSION}"
        f"|schema={SCHEMA_VERSION}"
        f"|countries={','.join(sorted(target_countries))}"
        f"|mtime={os.stat(path).st_mtime_ns}"
    )


def _cache_path(source_path: str, target_countries: set[str]) -> Tuple[str, str]:
    os.makedirs(CACHE_ROOT, exist_ok=True)
    base  = os.path.basename(source_path)
    country_key = hashlib.sha256(
        ",".join(sorted(target_countries)).encode()
    ).hexdigest()[:10]
    cache = os.path.join(
        CACHE_ROOT, f"{base}.v{SCHEMA_VERSION}.{country_key}.parquet"
    )
    stamp = cache + ".stamp"
    return cache, stamp


def load_and_cache(source_path: str, target_countries: set) -> pd.DataFrame:
    """
    Load a source TSV, normalise with Polars, cache as Parquet.
    On subsequent calls the Parquet is returned directly if the stamp matches.
    """
    cache, stamp_path = _cache_path(source_path, target_countries)
    expected_stamp    = _cache_stamp(source_path, target_countries)

    if os.path.isfile(cache) and os.path.isfile(stamp_path):
        with open(stamp_path, encoding="utf-8") as f:
            if f.read().strip() == expected_stamp:
                df = pd.read_parquet(cache)
                df = df[df["country"].isin(target_countries)].reset_index(drop=True)
                print(f"    cache hit  {os.path.basename(source_path)}: "
                      f"{len(df):,} rows", flush=True)
                return df

    print(f"    building cache  {os.path.basename(source_path)} …", flush=True)
    t0 = time.perf_counter()

    # Polars streaming: normalise natively — no Python row loop
    lf = (
        pl.scan_csv(
            source_path,
            separator="\t",
            infer_schema_length=0,
            schema_overrides={
                "entity_id":        pl.String,
                "business_name":    pl.String,
                "business_address": pl.String,
                "country":          pl.String,
            },
        )
        .select(["entity_id", "business_name", "business_address", "country"])
        .filter(pl.col("country").is_in(sorted(target_countries)))
        .with_columns(
            norm_name = _polars_normalize_expr(pl.col("business_name"),  _NAME_EXPANSIONS),
            norm_addr = _polars_normalize_expr(pl.col("business_address"), _ADDR_EXPANSIONS),
            addr_city = _polars_normalize_expr(
                pl.col("business_address").str.split(",").list.last(),
                _ADDR_EXPANSIONS,
            ),
        )
        .with_columns(
            name_core      = pl.col("norm_name").str.replace_all(
                r"\b(?:limited|llc|private|incorporated|corporation"
                r"|company|llp|sarl|sasu|sa)\b", "", literal=False
            ).str.replace_all(r"\s+", " ", literal=False).str.strip_chars(),
            addr_number    = pl.col("norm_addr").str.extract(r"\b(\d+[a-z]?)\b", 1).fill_null(""),
            addr_region    = pl.col("norm_addr").str.extract(r"(\S+)$", 1).fill_null(""),
        )
    )

    df = lf.collect(engine="streaming").to_pandas()

    # Lightweight Python-side keys (fast, small columns)
    df["meaningful_tok"] = df["name_core"].map(_first_meaningful_token)
    df["first_tok"]      = df["norm_name"].map(_first_token)
    df["prefix_key"]     = df["norm_name"].map(_prefix_key)
    df["compact_name"]   = df["norm_name"].str.replace(r"\s+", "", regex=True)

    # Persist full cache (all countries)
    df.to_parquet(cache, index=False)
    with open(stamp_path, "w", encoding="utf-8") as f:
        f.write(expected_stamp)
    elapsed = time.perf_counter() - t0
    print(f"    cached  {os.path.basename(source_path)}: "
          f"{len(df):,} rows  ({elapsed:.1f}s)", flush=True)

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Layer A — Deterministic exact blocking
# ══════════════════════════════════════════════════════════════════════════════

_EXACT_PASSES = [
    # (key_col_expr, min_key_len, cap)          ← key_col_expr = column or None (= skip)
    ("name_core_city",   5, 160),
    ("meaningful_city",  3, 160),
    ("number_city",      4, 160),
    ("first_tok",        3,  50),
    ("prefix_key",       2,  80),
    ("name_core",        3, 120),
    ("compact_name",     3, 120),
    ("addr_region",      3, 120),
    ("addr_tail2",       5, 120),
    ("addr_tail3",       8, 120),
    ("region_number",    4,  80),
]


def _build_composite_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Append composite + derived key columns in-place."""
    df = df.copy()
    df["name_core_city"] = df["name_core"] + "\x1f" + df["addr_city"]
    df["meaningful_city"] = df["meaningful_tok"] + "\x1f" + df["addr_city"]
    df["number_city"]    = df["addr_number"] + "\x1f" + df["addr_city"]
    df["region_number"]  = df["addr_region"] + "\x1f" + df["addr_number"]
    # tail-n address tokens
    def tail_n(s: str, n: int) -> str:
        parts = s.split()
        return " ".join(parts[-n:]) if len(parts) >= n else ""
    df["addr_tail2"] = df["norm_addr"].map(lambda s: tail_n(s, 2))
    df["addr_tail3"] = df["norm_addr"].map(lambda s: tail_n(s, 3))
    return df


def _exact_block_one(
    s1_keys: pd.Series,
    pool_keys: pd.Series,
    pool_ids: pd.Series,
    min_len: int,
    cap: int,
) -> Dict[int, List[str]]:
    """Exact match using sorted integer codes instead of pandas groupby.apply.

    Text keys are factorized losslessly on CPU.  The lookup of query codes in
    the sorted pool-code table is moved to CUDA when available; candidate IDs
    remain exact strings and are capped after the lookup.
    """
    hits: Dict[int, List[str]] = {}
    valid_mask = s1_keys.str.len() >= min_len
    if not valid_mask.any():
        return hits

    query_index = s1_keys.index[valid_mask].to_numpy(dtype=np.int64)
    query_keys = s1_keys[valid_mask].astype(str).to_numpy()
    pool_mask = pool_keys.str.len() >= min_len
    pool_keys_array = pool_keys[pool_mask].astype(str).to_numpy()
    pool_ids_array = pool_ids[pool_mask].astype(str).to_numpy()
    if not len(pool_keys_array):
        return hits

    codes, _ = pd.factorize(
        np.concatenate((query_keys, pool_keys_array)), sort=False
    )
    query_codes = codes[: len(query_keys)].astype(np.int64, copy=False)
    pool_codes = codes[len(query_keys):].astype(np.int64, copy=False)
    pool_order = np.argsort(pool_codes, kind="stable")
    sorted_codes = pool_codes[pool_order]
    unique_codes, starts = np.unique(sorted_codes, return_index=True)
    ends = np.concatenate((starts[1:], np.array([len(sorted_codes)])))

    if _GPU:
        query_positions = torch.searchsorted(
            torch.as_tensor(unique_codes, device=_DEVICE),
            torch.as_tensor(query_codes, device=_DEVICE),
        ).cpu().numpy()
    else:
        query_positions = np.searchsorted(unique_codes, query_codes)

    for source_index, query_code, position in zip(
        query_index, query_codes, query_positions
    ):
        if position >= len(unique_codes) or unique_codes[position] != query_code:
            continue
        rows = pool_order[starts[position]:ends[position]][:cap]
        hits[int(source_index)] = pool_ids_array[rows].tolist()
    return hits


def layer_a(
    s1_c: pd.DataFrame,
    pool_c: pd.DataFrame,
) -> Dict[int, Dict[str, List[str]]]:
    """
    Returns {s1_row_idx: {'exact_name': [...], 'exact_address': [...]}}
    Oversized buckets (> cap * 3) are tagged in 'overflow' so Layer B can
    process them more carefully.
    """
    s1   = _build_composite_keys(s1_c)
    pool = _build_composite_keys(pool_c)

    # Accumulate with provenance
    name_hits: Dict[int, List[str]] = {}
    addr_hits: Dict[int, List[str]] = {}

    name_key_passes = {
        "name_core_city", "meaningful_city", "first_tok",
        "prefix_key", "name_core", "compact_name",
    }

    for key_col, min_len, cap in _EXACT_PASSES:
        if key_col not in s1.columns or key_col not in pool.columns:
            continue
        batch = _exact_block_one(
            s1[key_col], pool[key_col], pool["entity_id"], min_len, cap
        )
        target = name_hits if key_col in name_key_passes else addr_hits
        for idx, ids in batch.items():
            for eid in ids:
                if eid not in (target.get(idx) or []):
                    target.setdefault(idx, []).append(eid)

    result: Dict[int, Dict[str, List[str]]] = {}
    for idx in range(len(s1)):
        result[idx] = {
            "exact_name":    name_hits.get(idx, []),
            "exact_address": addr_hits.get(idx, []),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Layer B — Inverted character n-gram retrieval + sparse TF-IDF scoring
# ══════════════════════════════════════════════════════════════════════════════

class NgramInvertedIndex:
    """Compiled sparse character n-gram posting index."""

    def __init__(self, n: int = NGRAM_SIZE):
        self.n = n
        self._vectorizer = CountVectorizer(
            analyzer="char", ngram_range=(n, n), lowercase=False,
            binary=True, dtype=np.int8,
        )
        self._postings = None
        self._df = np.empty(0, dtype=np.int32)

    def build(self, texts: List[str], desc: str = "      n-gram index") -> None:
        """Build feature-to-row postings in compiled sparse code."""
        t0 = time.perf_counter()
        matrix = self._vectorizer.fit_transform(texts)
        self._postings = matrix.tocsc()
        del matrix
        self._df = np.diff(self._postings.indptr).astype(np.int32, copy=False)
        print(
            f"          {desc.strip()}: {self._postings.shape[1]:,} grams "
            f"({time.perf_counter() - t0:.1f}s)", flush=True,
        )

    def _cache_paths(self, prefix: str) -> Tuple[str, str, str]:
        return (
            f"{prefix}.vectorizer.pkl",
            f"{prefix}.postings.npz",
            f"{prefix}.meta.json",
        )

    def load(self, prefix: str) -> bool:
        """Load a complete index cache; return False for a partial/invalid cache."""
        vectorizer_path, postings_path, meta_path = self._cache_paths(prefix)
        if not all(os.path.isfile(path) for path in (
            vectorizer_path, postings_path, meta_path
        )):
            return False
        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("index_version") != INDEX_VERSION or meta.get("n") != self.n:
                return False
            with open(vectorizer_path, "rb") as handle:
                self._vectorizer = pickle.load(handle)
            self._postings = sp.load_npz(postings_path).tocsc()
            if tuple(meta.get("shape", ())) != self._postings.shape:
                self._postings = None
                return False
            self._df = np.diff(self._postings.indptr).astype(np.int32, copy=False)
            return True
        except (OSError, ValueError, pickle.PickleError, EOFError):
            self._postings = None
            return False

    def save(self, prefix: str) -> None:
        """Atomically save vectorizer, postings, and metadata."""
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        vectorizer_path, postings_path, meta_path = self._cache_paths(prefix)

        fd, vectorizer_tmp = tempfile.mkstemp(
            dir=os.path.dirname(prefix), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                pickle.dump(self._vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(vectorizer_tmp, vectorizer_path)
        finally:
            if os.path.exists(vectorizer_tmp):
                os.unlink(vectorizer_tmp)

        fd, postings_tmp = tempfile.mkstemp(
            dir=os.path.dirname(prefix), suffix=".npz"
        )
        os.close(fd)
        try:
            sp.save_npz(postings_tmp, self._postings)
            os.replace(postings_tmp, postings_path)
        finally:
            if os.path.exists(postings_tmp):
                os.unlink(postings_tmp)

        _atomic_write_json(
            {"index_version": INDEX_VERSION, "n": self.n,
             "shape": list(self._postings.shape)},
            meta_path,
        )

    def load_or_build(
        self,
        texts: List[str],
        prefix: str,
        desc: str,
    ) -> None:
        if self.load(prefix):
            print(
                f"          {desc.strip()} cache hit: "
                f"{self._postings.shape[1]:,} grams",
                flush=True,
            )
            return
        self.build(texts, desc=desc)
        self.save(prefix)
        print(f"          saved index cache: {prefix}", flush=True)

    def query_batch(
        self,
        texts: List[str],
        topk_ngrams: int = TOPK_NGRAMS,
        desc: str = "      n-gram queries",
    ) -> List[List[int]]:
        """Return posting unions for rare query grams with progress."""
        if self._postings is None:
            raise RuntimeError("n-gram index must be built before querying")
        query_matrix = self._vectorizer.transform(texts).tocsr()
        results: List[List[int]] = [[] for _ in range(query_matrix.shape[0])]
        n_features = self._postings.shape[1]
        total_batches = (
            query_matrix.shape[0] + NGRAM_QUERY_BATCH - 1
        ) // NGRAM_QUERY_BATCH
        for batch_start in tqdm(
            range(0, query_matrix.shape[0], NGRAM_QUERY_BATCH),
            total=total_batches,
            desc=desc,
            unit="query-batch",
            ncols=80,
            leave=False,
        ):
            batch_end = min(batch_start + NGRAM_QUERY_BATCH, query_matrix.shape[0])
            rows: List[int] = []
            cols: List[int] = []
            for local_row, row in enumerate(range(batch_start, batch_end)):
                start, end = query_matrix.indptr[row], query_matrix.indptr[row + 1]
                features = query_matrix.indices[start:end]
                if len(features):
                    rare = features[np.argsort(self._df[features])[:topk_ngrams]]
                    rows.extend([local_row] * len(rare))
                    cols.extend(rare.tolist())
            if not rows:
                continue

            selected = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.int8), (rows, cols)),
                shape=(batch_end - batch_start, n_features),
            )
            vote_matrix = selected @ self._postings.T
            for local_row in range(batch_end - batch_start):
                row = vote_matrix.getrow(local_row)
                if not row.nnz:
                    continue
                candidates = row.indices
                scores = row.data
                keep = min(MAX_QUERY_CANDIDATES, row.nnz)
                if row.nnz > keep:
                    top = np.argpartition(scores, -keep)[-keep:]
                    candidates, scores = candidates[top], scores[top]
                order = np.lexsort((candidates, -scores))
                results[batch_start + local_row] = (
                    candidates[order].astype(int).tolist()
                )
        return results


def _sparse_score_candidates(
    query_texts: List[str],
    cand_idx_per_query: List[List[int]],
    pool_texts: List[str],
    k: int,
) -> Dict[int, List[int]]:
    """
    Score (query, candidate) pairs with sparse TF-IDF cosine.
    Never builds a full (n_query × n_pool) matrix.
    Returns {query_idx: [pool_row_idx, ...]} sorted by descending score.
    """
    def bounded(values: List[str], limit: int) -> List[str]:
        if len(values) <= limit:
            return values
        positions = np.linspace(0, len(values) - 1, limit, dtype=np.int64)
        return [values[int(position)] for position in positions]

    needed_pool = sorted({
        idx for lst in cand_idx_per_query for idx in lst
        if 0 <= idx < len(pool_texts)
    })
    if not needed_pool or not query_texts:
        return {}

    query_fit = bounded(query_texts, SPARSE_FIT_DOCS // 2)
    pool_fit = bounded(
        [pool_texts[i] for i in needed_pool],
        SPARSE_FIT_DOCS - len(query_fit),
    )
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        min_df=1, sublinear_tf=True,
        max_features=MAX_FEATURES, dtype=np.float32,
    )
    vec.fit(query_fit + pool_fit)

    results: Dict[int, List[int]] = {}
    for batch_start in tqdm(
        range(0, len(query_texts), SPARSE_SCORE_BATCH),
        desc="      reduced TF-IDF scoring",
        unit="query-batch",
        ncols=80,
        leave=False,
    ):
        batch_end = min(batch_start + SPARSE_SCORE_BATCH, len(query_texts))
        batch_lists = cand_idx_per_query[batch_start:batch_end]
        batch_pool = sorted({
            idx for lst in batch_lists for idx in lst
            if 0 <= idx < len(pool_texts)
        })
        if not batch_pool:
            continue
        q_mat = normalize(
            vec.transform(query_texts[batch_start:batch_end]), norm="l2"
        )
        p_mat = normalize(
            vec.transform([pool_texts[idx] for idx in batch_pool]), norm="l2"
        )
        pool_local = {original: local for local, original in enumerate(batch_pool)}
        for local_query, cand_list in enumerate(batch_lists):
            valid_candidates = [idx for idx in cand_list if idx in pool_local]
            local_rows = [pool_local[idx] for idx in valid_candidates]
            if not local_rows:
                continue
            scores = q_mat[local_query].dot(p_mat[local_rows].T).toarray()[0]
            order = np.argsort(-scores)
            results[batch_start + local_query] = [
                valid_candidates[j] for j in order[:k] if scores[j] > 0
            ]
    return results


def _index_signature(
    s2_path: str,
    s3_path: str,
    country: str,
    pool_c: pd.DataFrame,
) -> str:
    """Fingerprint the exact pool version used by a country index."""
    h = hashlib.sha256()
    h.update(INDEX_VERSION.encode())
    h.update(country.encode())
    h.update(str(len(pool_c)).encode())
    if len(pool_c):
        h.update(str(pool_c["entity_id"].iloc[0]).encode())
        h.update(str(pool_c["entity_id"].iloc[-1]).encode())
    for path in (s2_path, s3_path):
        h.update(os.path.abspath(path).encode())
        h.update(str(os.stat(path).st_mtime_ns).encode())
    return h.hexdigest()[:20]


def layer_b(
    s1_c: pd.DataFrame,
    pool_c: pd.DataFrame,
    already_found: Dict[int, set],
    index_cache_key: str,
) -> Dict[int, Dict[str, List[str]]]:
    """
    Inverted n-gram retrieval on names + addresses.
    already_found: {s1_row_idx: set(pool_entity_ids)} — skip if already ≥ MAX_CANDIDATES.
    Returns {s1_row_idx: {'ngram_name': [...], 'ngram_address': [...]}}
    """
    pool_ids       = pool_c["entity_id"].tolist()
    pool_names     = pool_c["norm_name"].tolist()
    pool_addrs     = pool_c["norm_addr"].tolist()

    result: Dict[int, Dict[str, List[str]]] = {
        i: {"ngram_name": [], "ngram_address": []} for i in range(len(s1_c))
    }

    # ── Name n-gram retrieval ──────────────────────────────────────────────
    print("      [B] building name n-gram index …", flush=True)
    t0 = time.perf_counter()
    name_idx = NgramInvertedIndex()
    name_prefix = os.path.join(INDEX_CACHE_ROOT, f"{index_cache_key}.name")
    name_idx.load_or_build(
        pool_names, name_prefix, desc="      name n-gram index"
    )
    print(f"          name index: {name_idx._postings.shape[1]:,} n-grams  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    q_names   = s1_c["norm_name"].tolist()
    cand_rows = name_idx.query_batch(
        q_names, desc="      name n-gram queries"
    )
    # Sparse TF-IDF scoring on reduced candidate set
    scored    = _sparse_score_candidates(q_names, cand_rows, pool_names, LAYER_B_CAP)
    for qi, rows in scored.items():
        existing = already_found.get(qi, set())
        result[qi]["ngram_name"] = [
            pool_ids[r] for r in rows if pool_ids[r] not in existing
        ][:LAYER_B_CAP]

    # ── Address n-gram retrieval ───────────────────────────────────────────
    print("      [B] building address n-gram index …", flush=True)
    t0 = time.perf_counter()
    addr_idx = NgramInvertedIndex()
    addr_prefix = os.path.join(INDEX_CACHE_ROOT, f"{index_cache_key}.address")
    addr_idx.load_or_build(
        pool_addrs, addr_prefix, desc="      address n-gram index"
    )
    print(f"          addr index: {addr_idx._postings.shape[1]:,} n-grams  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    q_addrs    = s1_c["norm_addr"].tolist()
    cand_rows_a = addr_idx.query_batch(
        q_addrs, desc="      address n-gram queries"
    )
    scored_a   = _sparse_score_candidates(q_addrs, cand_rows_a, pool_addrs, LAYER_B_CAP)
    for qi, rows in scored_a.items():
        existing = already_found.get(qi, set())
        result[qi]["ngram_address"] = [
            pool_ids[r] for r in rows if pool_ids[r] not in existing
        ][:LAYER_B_CAP]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Layer C — GPU pairwise scoring of merged candidates
# ══════════════════════════════════════════════════════════════════════════════

def _gpu_score_pairs(
    query_vecs: np.ndarray,   # (n_pairs, vocab) float32 dense
    cand_vecs:  np.ndarray,   # (n_pairs, vocab) float32 dense
) -> np.ndarray:
    """Score (query_i, candidate_i) pairs by cosine similarity on GPU/CPU."""
    if not len(query_vecs):
        return np.empty(0, dtype=np.float32)
    if not _GPU:
        return (query_vecs * cand_vecs).sum(axis=1)

    # Bound the pair tile from actual free VRAM. This prevents a single
    # unusually large candidate list from materialising an unsafe dense tile.
    free = torch.cuda.mem_get_info(0)[0]
    row_bytes = query_vecs.shape[1] * 4 * 2
    batch_rows = max(1, min(len(query_vecs), int(free * 0.20 / row_bytes)))
    output = np.empty(len(query_vecs), dtype=np.float32)
    try:
        for start in range(0, len(query_vecs), batch_rows):
            end = min(start + batch_rows, len(query_vecs))
            q_t = torch.from_numpy(query_vecs[start:end]).to(_DEVICE)
            c_t = torch.from_numpy(cand_vecs[start:end]).to(_DEVICE)
            output[start:end] = (q_t * c_t).sum(dim=1).cpu().numpy()
            del q_t, c_t
        torch.cuda.empty_cache()
        return output
    except RuntimeError:
        torch.cuda.empty_cache()
        return (query_vecs * cand_vecs).sum(axis=1)


def layer_c_score(
    s1_c: pd.DataFrame,
    pool_c: pd.DataFrame,
    merged: Dict[int, List[str]],        # {s1_idx: [pool_entity_ids]}
    pool_id_to_idx: Dict[str, int],
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Compute per-pair (name_score, addr_score) for each merged candidate list.
    Returns {s1_idx: [(entity_id, combined_score), ...]} sorted desc.
    """
    if not merged:
        return {}

    # Build shared TF-IDF vectoriser over name+address
    needed_ids = {cid for ids in merged.values() for cid in ids}
    needed_rows = [pool_id_to_idx[cid] for cid in needed_ids
                   if cid in pool_id_to_idx]
    needed_pool = pool_c.iloc[needed_rows].reset_index(drop=True)
    pool_name_list = needed_pool["norm_name"].tolist()
    pool_addr_list = needed_pool["norm_addr"].tolist()
    s1_name_list   = s1_c["norm_name"].tolist()
    s1_addr_list   = s1_c["norm_addr"].tolist()

    all_names = s1_name_list + pool_name_list
    all_addrs = s1_addr_list + pool_addr_list

    name_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        min_df=1, sublinear_tf=True, max_features=MAX_FEATURES, dtype=np.float32,
    )
    addr_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        min_df=1, sublinear_tf=True, max_features=MAX_FEATURES, dtype=np.float32,
    )
    name_vec.fit(all_names)
    addr_vec.fit(all_addrs)

    s1_name_mat  = normalize(name_vec.transform(s1_name_list),  norm="l2")
    pool_name_mat = normalize(name_vec.transform(pool_name_list), norm="l2")
    s1_addr_mat  = normalize(addr_vec.transform(s1_addr_list),  norm="l2")
    pool_addr_mat = normalize(addr_vec.transform(pool_addr_list), norm="l2")
    local_by_global = {row: local for local, row in enumerate(needed_rows)}

    scored: Dict[int, List[Tuple[str, float]]] = {}
    for qi, cand_ids in tqdm(merged.items(), desc="      [C] GPU scoring",
                             ncols=80, leave=False):
        if not cand_ids:
            scored[qi] = []
            continue
        valid_ids = [eid for eid in cand_ids if eid in pool_id_to_idx]
        pool_rows = [local_by_global[pool_id_to_idx[eid]] for eid in valid_ids]
        if not pool_rows:
            scored[qi] = []
            continue
        n_pairs = len(pool_rows)
        # Dense batches — only for this query's small candidate set
        q_name_dense = np.repeat(s1_name_mat[qi].toarray(), n_pairs, axis=0)
        c_name_dense = pool_name_mat[pool_rows].toarray()
        q_addr_dense = np.repeat(s1_addr_mat[qi].toarray(), n_pairs, axis=0)
        c_addr_dense = pool_addr_mat[pool_rows].toarray()

        name_scores = _gpu_score_pairs(q_name_dense, c_name_dense)
        addr_scores = _gpu_score_pairs(q_addr_dense, c_addr_dense)
        combined    = 0.6 * name_scores + 0.4 * addr_scores

        order = np.argsort(-combined)
        scored[qi] = [(valid_ids[j], float(combined[j])) for j in order]
    return scored


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run_signature(
    s1_df: pd.DataFrame,
    s2_path: str,
    s3_path: str,
    max_candidates: int,
    run_layer_c: bool,
) -> str:
    h = hashlib.sha256()
    h.update(STRATEGY_VERSION.encode())
    h.update(NORMALIZER_VERSION.encode())
    h.update(SCHEMA_VERSION.encode())
    h.update(f"{max_candidates}:{int(run_layer_c)}".encode())
    for path in (s2_path, s3_path):
        h.update(os.path.abspath(path).encode())
        h.update(str(os.stat(path).st_mtime_ns).encode())
    for entity_id in s1_df["entity_id"].astype(str):
        h.update(entity_id.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def _country_ckpt_dir(country: str, run_signature: str) -> str:
    safe = hashlib.sha256(country.encode()).hexdigest()[:12]
    d    = os.path.join(CHECKPOINT_ROOT, run_signature, safe)
    os.makedirs(d, exist_ok=True)
    return d


def _atomic_write_parquet(df: pd.DataFrame, path: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        os.close(fd)
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_write_json(obj, path: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ══════════════════════════════════════════════════════════════════════════════
# Main blocking function
# ══════════════════════════════════════════════════════════════════════════════

def build_candidates(
    s1_df:          pd.DataFrame,
    s2_path:        str,
    s3_path:        str,
    max_candidates: int = MAX_CANDIDATES,
    run_layer_c:    bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Build {s1_entity_id: [candidate_ids]} plus a scored candidate DataFrame.

    Returns
    -------
    candidates  : dict  {s1_id: [cand_id, ...]}
    scored_df   : pd.DataFrame  with columns
                  [source1_entity_id, candidate_entity_id,
                   retrieval_sources, name_score, address_score, country_match]
    """
    t_start = time.perf_counter()
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── S1 normalisation ──────────────────────────────────────────────────
    print("  [1/4] Normalising S1 …", flush=True)
    t0 = time.perf_counter()
    s1 = _normalize_df(s1_df.copy(), n_workers=N_WORKERS)
    print(f"    {len(s1):,} rows  ({time.perf_counter()-t0:.1f}s)", flush=True)

    target_countries = set(s1["country"].unique())
    print(f"  [2/4] Countries: {sorted(target_countries)}", flush=True)

    # ── Pool loading (Polars + cache) ─────────────────────────────────────
    print("  [3/4] Loading S2/S3 (Polars cache) …", flush=True)
    pool2 = load_and_cache(s2_path, target_countries)
    pool3 = load_and_cache(s3_path, target_countries)
    pool  = pd.concat([pool2, pool3], ignore_index=True)
    del pool2, pool3
    gc.collect()
    print(f"    combined pool: {len(pool):,} rows", flush=True)

    # ── Per-country blocking ──────────────────────────────────────────────
    print("  [4/4] Per-country hybrid blocking …", flush=True)
    candidates: Dict[str, List[str]] = {eid: [] for eid in s1_df["entity_id"]}
    all_scored_rows: List[dict] = []
    run_signature = _run_signature(
        s1_df, s2_path, s3_path, max_candidates, run_layer_c
    )
    print(f"    run signature: {run_signature}", flush=True)

    for country in sorted(target_countries):
        print(f"\n  ── {country} ──", flush=True)
        s1_c   = s1[s1["country"] == country].reset_index(drop=True)
        pool_c = pool[pool["country"] == country].reset_index(drop=True)

        if len(s1_c) == 0 or len(pool_c) == 0:
            print("    skipped (empty)", flush=True)
            continue

        print(f"    S1={len(s1_c):,}  pool={len(pool_c):,}", flush=True)

        ckpt_dir  = _country_ckpt_dir(country, run_signature)
        ckpt_file = os.path.join(ckpt_dir, "candidates.json")
        scored_ckpt = os.path.join(ckpt_dir, "scored.parquet")
        layer_a_ckpt = os.path.join(ckpt_dir, "layer_a.json")
        layer_b_ckpt = os.path.join(ckpt_dir, "layer_b.json")

        if os.path.isfile(ckpt_file):
            with open(ckpt_file, encoding="utf-8") as f:
                saved = json.load(f)
            candidates.update(saved)
            if os.path.isfile(scored_ckpt):
                all_scored_rows.extend(pd.read_parquet(scored_ckpt).to_dict("records"))
            print(f"    ✓ resumed  {len(saved):,} S1 rows from checkpoint", flush=True)
            continue

        pool_id_to_idx = {eid: i for i, eid in enumerate(pool_c["entity_id"])}

        # ── Layer A ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if os.path.isfile(layer_a_ckpt):
            with open(layer_a_ckpt, encoding="utf-8") as f:
                a_result = {int(k): v for k, v in json.load(f).items()}
            print("    [A] resumed exact-blocking checkpoint", flush=True)
        else:
            print("    [A] exact blocking …", flush=True)
            a_result = layer_a(s1_c, pool_c)
            _atomic_write_json(a_result, layer_a_ckpt)
        n_a = sum(
            1 for v in a_result.values()
            if v["exact_name"] or v["exact_address"]
        )
        print(f"      {n_a:,} S1 rows got exact candidates  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

        # ── Layer B ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        index_key = _index_signature(s2_path, s3_path, country, pool_c)
        already: Dict[int, set] = {
            i: set(v["exact_name"] + v["exact_address"])
            for i, v in a_result.items()
        }
        if os.path.isfile(layer_b_ckpt):
            with open(layer_b_ckpt, encoding="utf-8") as f:
                b_result = {int(k): v for k, v in json.load(f).items()}
            print("    [B] resumed n-gram checkpoint", flush=True)
        else:
            print("    [B] n-gram retrieval …", flush=True)
            b_result = layer_b(s1_c, pool_c, already, index_key)
            _atomic_write_json(b_result, layer_b_ckpt)
        n_b = sum(
            1 for v in b_result.values()
            if v["ngram_name"] or v["ngram_address"]
        )
        print(f"      {n_b:,} S1 rows got n-gram candidates  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

        # ── Union A + B ────────────────────────────────────────────────────
        merged: Dict[int, List[str]] = {}
        provenance: Dict[int, Dict[str, List[str]]] = {}
        for idx in range(len(s1_c)):
            seen: Dict[str, set] = {
                "exact_name":    set(),
                "exact_address": set(),
                "ngram_name":    set(),
                "ngram_address": set(),
            }
            order_list: List[str] = []
            for src in ("exact_name", "exact_address", "ngram_name", "ngram_address"):
                layer = a_result[idx] if src.startswith("exact") else b_result[idx]
                for eid in layer.get(src, []):
                    if eid not in seen[src]:
                        seen[src].add(eid)
                        if eid not in {e for s in seen.values() for e in s} - seen[src]:
                            order_list.append(eid)

            # In GPU mode, retain the complete retrieved union until scoring;
            # truncating before scoring can discard the true match forever.
            all_unique: List[str] = list(dict.fromkeys(order_list))
            if not run_layer_c:
                all_unique = all_unique[:max_candidates]
            merged[idx]    = all_unique
            provenance[idx] = {src: list(seen[src]) for src in seen}

        # ── Layer C — GPU pairwise scoring ─────────────────────────────────
        if run_layer_c:
            t0 = time.perf_counter()
            print("    [C] GPU pairwise scoring …", flush=True)
            c_scored = layer_c_score(s1_c, pool_c, merged, pool_id_to_idx)
            # Re-rank merged by combined score, keep same cap
            for idx, scored_pairs in c_scored.items():
                merged[idx] = [eid for eid, _ in scored_pairs[:max_candidates]]
            print(f"      scored {len(c_scored):,} S1 rows  "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
        else:
            c_scored = {}

        # ── Persist results ────────────────────────────────────────────────
        s1_ids    = s1_c["entity_id"].tolist()
        s1_ctries = s1_c["country"].tolist()
        country_cands: Dict[str, List[str]] = {}
        scored_rows: List[dict] = []

        for idx, s1_id in enumerate(s1_ids):
            cands = merged.get(idx, [])
            country_cands[s1_id] = cands
            candidates[s1_id]    = cands

            prov  = provenance.get(idx, {})
            c_map = {eid: score for eid, score in c_scored.get(idx, [])}
            for cid in cands:
                prov_flags = ",".join(
                    src for src, lst in prov.items() if cid in lst
                )
                scored_rows.append({
                    "source1_entity_id":   s1_id,
                    "candidate_entity_id": cid,
                    "retrieval_sources":   prov_flags,
                    "combined_score":      c_map.get(cid, 0.0),
                    "country_match":       (
                        pool_c.loc[pool_id_to_idx[cid], "country"] == s1_ctries[idx]
                        if cid in pool_id_to_idx else False
                    ),
                })

        # Atomic checkpoint
        _atomic_write_json(country_cands, ckpt_file)
        if scored_rows:
            scored_df_country = pd.DataFrame(scored_rows)
            _atomic_write_parquet(scored_df_country, scored_ckpt)
            all_scored_rows.extend(scored_rows)

        avg = sum(len(v) for v in country_cands.values()) / max(1, len(country_cands))
        print(f"    ✓ checkpointed  {len(country_cands):,} S1 rows  "
              f"avg_cands={avg:.1f}", flush=True)

        del pool_c, s1_c, a_result, b_result, merged
        gc.collect()
        if _GPU:
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t_start
    n_ne = sum(1 for v in candidates.values() if v)
    print(f"\n  Done. {n_ne:,}/{len(candidates):,} S1 entities have candidates  "
          f"({elapsed:.1f}s total)", flush=True)

    scored_df = pd.DataFrame(all_scored_rows) if all_scored_rows else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id",
                 "retrieval_sources", "combined_score", "country_match"]
    )
    return candidates, scored_df


# ══════════════════════════════════════════════════════════════════════════════
# Recall evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_recall(
    candidates: Dict[str, List[str]],
    ground_truth: pd.DataFrame,
    layer_results: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> dict:
    gt_map: Dict[str, set] = {}
    for _, row in ground_truth.iterrows():
        m = row["matched_entity_ids"]
        if m and str(m).strip():
            gt_map[row["source1_entity_id"]] = {
                x.strip() for x in str(m).split(",") if x.strip()
            }

    total_true  = sum(len(v) for v in gt_map.values())
    recalled    = sum(len(gt_map[s] & set(candidates.get(s, []))) for s in gt_map)

    cand_counts = [len(v) for v in candidates.values()]
    pct95 = float(np.percentile(cand_counts, 95)) if cand_counts else 0.0
    pct99 = float(np.percentile(cand_counts, 99)) if cand_counts else 0.0

    result = {
        "recall":                recalled / total_true if total_true else 0.0,
        "pairs_recalled":        recalled,
        "pairs_in_gt":           total_true,
        "total_candidates":      sum(cand_counts),
        "avg_candidates_per_s1": sum(cand_counts) / len(cand_counts) if cand_counts else 0,
        "p95_candidates":        pct95,
        "p99_candidates":        pct99,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Output writers
# ══════════════════════════════════════════════════════════════════════════════

def write_candidate_pairs(
    candidates: Dict[str, List[str]],
    s1_df: pd.DataFrame,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = [
        {"source1_entity_id": eid,
         "candidate_entity_ids": ",".join(candidates.get(eid, []))}
        for eid in s1_df["entity_id"]
    ]
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
    print(f"  Written: {output_path}  ({len(rows):,} rows)", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v2 hybrid blocking")
    parser.add_argument("--sample", type=float, default=None,
                        help="fraction of S1 rows (default: env SAMPLE or 0.05)")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--no-layer-c", action="store_true",
                        help="skip GPU pairwise scoring (faster, lower quality)")
    args = parser.parse_args()

    DATA    = os.path.join(_ROOT, "student_resource", "dataset", "train")
    S1_PATH = os.path.join(DATA, "train_source1.tsv")
    S2_PATH = os.path.join(DATA, "train_source2.tsv")
    S3_PATH = os.path.join(DATA, "train_source3.tsv")
    GT_PATH = os.path.join(DATA, "train_ground_truth.tsv")

    tracemalloc.start()
    if _GPU:
        torch.cuda.reset_peak_memory_stats()

    print("Loading S1 + GT …", flush=True)
    s1 = pd.read_csv(S1_PATH, sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str).fillna("")
    print(f"  S1={len(s1):,}  GT={len(gt):,}", flush=True)

    SAMPLE = args.sample if args.sample is not None else float(
        os.environ.get("SAMPLE", "0.05")
    )
    if not 0 < SAMPLE <= 1:
        parser.error("--sample must be > 0 and ≤ 1")
    if SAMPLE < 1.0:
        s1 = s1.sample(frac=SAMPLE, random_state=42).reset_index(drop=True)
        gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))].reset_index(drop=True)
        print(f"  Sampled {SAMPLE:.4%}: S1={len(s1):,}  GT={len(gt):,}", flush=True)

    print("\nBuilding candidates …", flush=True)
    cands, scored_df = build_candidates(
        s1, S2_PATH, S3_PATH,
        max_candidates=args.max_candidates,
        run_layer_c=not args.no_layer_c,
    )

    # ── Metrics ───────────────────────────────────────────────────────────
    print("\nEvaluating recall …", flush=True)
    m = evaluate_recall(cands, gt)
    peak_ram_mb  = tracemalloc.get_traced_memory()[1] / 1e6
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if _GPU else 0.0
    )

    print(f"\n{'─'*52}")
    print(f"  Blocking Recall        : {m['recall']:.4f}  ({m['recall']*100:.2f}%)")
    print(f"  Pairs in GT            : {m['pairs_in_gt']:,}")
    print(f"  Pairs recalled         : {m['pairs_recalled']:,}")
    print(f"  Total candidates       : {m['total_candidates']:,}")
    print(f"  Avg candidates / S1    : {m['avg_candidates_per_s1']:.1f}")
    print(f"  P95 candidates / S1    : {m['p95_candidates']:.0f}")
    print(f"  P99 candidates / S1    : {m['p99_candidates']:.0f}")
    print(f"  Peak RAM               : {peak_ram_mb:.0f} MB")
    print(f"  Peak VRAM              : {peak_vram_mb:.0f} MB")
    print(f"{'─'*52}")

    # ── Write outputs ──────────────────────────────────────────────────────
    out_tsv = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    write_candidate_pairs(cands, s1, out_tsv)

    if not scored_df.empty:
        scored_path = os.path.join(OUTPUT_DIR, "scored_candidates.parquet")
        scored_df.to_parquet(scored_path, index=False)
        print(f"  Scored pairs: {scored_path}  ({len(scored_df):,} rows)", flush=True)
