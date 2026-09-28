"""
features.py — Pairwise feature engineering for Business Entity Resolution.

For every (S1, candidate) pair produced by blocking, computes a feature vector
used to train/run the XGBoost matching classifier.

Feature groups
--------------
Name similarity
  name_jaccard            token Jaccard on normalised names
  name_levenshtein_norm   normalised edit distance  (0=identical, 1=totally different)
  name_jaro_winkler       Jaro-Winkler on normalised names
  name_cosine_tfidf       TF-IDF cosine (fitted on training corpus)
  name_len_ratio          min/max character length ratio
  name_common_tokens      raw count of shared name tokens

Address similarity
  addr_jaccard            token Jaccard on normalised addresses
  addr_levenshtein_norm   normalised edit distance
  addr_cosine_tfidf       TF-IDF cosine
  addr_len_ratio          min/max character length ratio
  addr_common_tokens      raw count of shared address tokens

Structural
  country_match           1 if country strings match exactly (lowercased)
  both_addr_empty         1 if both sides have no address
  one_addr_empty          1 if exactly one side has no address

Usage
-----
# Build feature matrix from blocking output + ground truth labels
python code/business_entity_resolution/src/features.py

Outputs
-------
output/features/train_features.parquet
output/features/train_labels.parquet
output/features/tfidf_name.pkl
output/features/tfidf_addr.pkl
"""

from __future__ import annotations

import os
import json
import sys
import pickle
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from rapidfuzz.distance import Levenshtein, JaroWinkler

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)
from preprocess import (  # noqa: E402
    _ADDR_EXPANSIONS,
    _NAME_EXPANSIONS,
    _PUNCT_RE,
    _WS_RE,
    normalize_name,
    normalize_address,
    tokenize,
)

_ROOT       = os.path.abspath(os.path.join(_SRC_DIR, "..", "..", ".."))
OUTPUT_DIR  = os.path.join(_ROOT, "output", "features")
DATA_TRAIN  = os.path.join(_ROOT, "student_resource", "dataset", "train")
CANDS_PATH  = os.path.join(_ROOT, "output", "candidates", "v2_hybrid", "candidate_pairs.tsv")
FEATURE_VERSION = "features-v2"
TFIDF_FIT_MAX_DOCS = 1_000_000


# ── Similarity helpers ────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _lev_norm(a: str, b: str) -> float:
    """Normalised Levenshtein: 0 = identical, 1 = totally different."""
    if not a and not b:
        return 0.0
    return Levenshtein.normalized_distance(a, b)


def _jaro_winkler(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return JaroWinkler.similarity(a, b)


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _common_tokens(a: str, b: str) -> int:
    return len(tokenize(a) & tokenize(b))


# ── TF-IDF vectorisers ────────────────────────────────────────────────────────

def fit_tfidf(
    texts: List[str],
    max_features: int = 50_000,
) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        sublinear_tf=True,
        max_features=max_features,
        dtype=np.float32,
    )
    vec.fit(texts)
    return vec


def _fit_tfidf_for_candidates(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    text_column: str,
    max_docs: int = TFIDF_FIT_MAX_DOCS,
) -> TfidfVectorizer:
    """Fit TF-IDF on relevant records, with a deterministic memory bound."""
    s1_texts = s1_df[text_column].tolist()
    pool_texts = pool_df[text_column].drop_duplicates().tolist()
    remaining = max(0, max_docs - len(s1_texts))
    if len(pool_texts) > remaining:
        positions = np.linspace(
            0, len(pool_texts) - 1, num=remaining, dtype=np.int64
        ) if remaining else np.empty(0, dtype=np.int64)
        pool_texts = [pool_texts[int(i)] for i in positions]
    corpus = s1_texts + pool_texts
    print(
        f"  TF-IDF {text_column}: fitting on {len(corpus):,} documents "
        f"(pool candidates={len(pool_texts):,})",
        flush=True,
    )
    return fit_tfidf(corpus)


def _cosine_row(vec: TfidfVectorizer, a: str, b: str) -> float:
    """Cosine similarity between two strings using a pre-fitted vectoriser."""
    mat = normalize(vec.transform([a, b]), norm="l2")
    return float(mat[0].dot(mat[1].T).toarray()[0, 0])


# ── Batch cosine (much faster than row-by-row for large pair sets) ────────────

def _batch_cosine(
    vec: TfidfVectorizer,
    a_texts: List[str],
    b_texts: List[str],
    batch_size: int = 10_000,
) -> np.ndarray:
    """
    Compute cosine similarity for paired (a_texts[i], b_texts[i]).
    Returns float32 array of length len(a_texts).
    """
    n = len(a_texts)
    scores = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        a_mat = normalize(vec.transform(a_texts[start:end]), norm="l2")
        b_mat = normalize(vec.transform(b_texts[start:end]), norm="l2")
        # element-wise dot product of paired rows
        scores[start:end] = np.asarray(a_mat.multiply(b_mat).sum(axis=1)).ravel()
    return scores


def _polars_normalize_expr(expr: pl.Expr, expansions: list) -> pl.Expr:
    out = expr.fill_null("").str.normalize("NFKC").str.to_lowercase()
    for pattern, replacement in expansions:
        out = out.str.replace_all(pattern, replacement)
    return out.str.replace_all(_PUNCT_RE.pattern, " ").str.replace_all(
        _WS_RE.pattern, " "
    ).str.strip_chars()


# ── Core feature computation ──────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "name_jaccard",
    "name_levenshtein_norm",
    "name_jaro_winkler",
    "name_cosine_tfidf",
    "name_len_ratio",
    "name_common_tokens",
    "addr_jaccard",
    "addr_levenshtein_norm",
    "addr_cosine_tfidf",
    "addr_len_ratio",
    "addr_common_tokens",
    "country_match",
    "both_addr_empty",
    "one_addr_empty",
]


def compute_feature_matrix(
    pairs: pd.DataFrame,       # columns: source1_entity_id, candidate_entity_id
    s1_df: pd.DataFrame,       # columns: entity_id, norm_name, norm_addr, country
    pool_df: pd.DataFrame,     # columns: entity_id, norm_name, norm_addr, country
    name_vec: TfidfVectorizer,
    addr_vec: TfidfVectorizer,
) -> pd.DataFrame:
    """
    Compute the full feature matrix for a set of candidate pairs.

    Parameters
    ----------
    pairs     : DataFrame with (source1_entity_id, candidate_entity_id)
    s1_df     : normalised S1 records indexed by entity_id
    pool_df   : normalised S2+S3 records indexed by entity_id
    name_vec  : fitted TF-IDF vectoriser for names
    addr_vec  : fitted TF-IDF vectoriser for addresses

    Returns
    -------
    DataFrame with FEATURE_COLUMNS, same row order as `pairs`.
    """
    s1_idx   = s1_df.set_index("entity_id")
    pool_idx = pool_df.set_index("entity_id")
    s1_maps = {column: s1_idx[column].to_dict() for column in (
        "norm_name", "norm_addr", "country"
    )}
    pool_maps = {column: pool_idx[column].to_dict() for column in (
        "norm_name", "norm_addr", "country"
    )}

    s1_ids   = pairs["source1_entity_id"].tolist()
    cand_ids = pairs["candidate_entity_id"].tolist()

    # Vectorised lookup — much faster than iterrows
    s1_names   = [s1_maps["norm_name"].get(i, "") for i in s1_ids]
    s1_addrs   = [s1_maps["norm_addr"].get(i, "") for i in s1_ids]
    s1_ctries  = [s1_maps["country"].get(i, "") for i in s1_ids]
    c_names    = [pool_maps["norm_name"].get(i, "") for i in cand_ids]
    c_addrs    = [pool_maps["norm_addr"].get(i, "") for i in cand_ids]
    c_ctries   = [pool_maps["country"].get(i, "") for i in cand_ids]

    n = len(pairs)
    print(f"  Computing string features for {n:,} pairs …", flush=True)
    t0 = time.perf_counter()

    name_jaccard         = np.array([_jaccard(a, b)      for a, b in zip(s1_names, c_names)],  dtype=np.float32)
    name_lev             = np.array([_lev_norm(a, b)     for a, b in zip(s1_names, c_names)],  dtype=np.float32)
    name_jw              = np.array([_jaro_winkler(a, b) for a, b in zip(s1_names, c_names)],  dtype=np.float32)
    name_len_ratio       = np.array([_len_ratio(a, b)    for a, b in zip(s1_names, c_names)],  dtype=np.float32)
    name_common          = np.array([_common_tokens(a, b) for a, b in zip(s1_names, c_names)], dtype=np.float32)

    addr_jaccard         = np.array([_jaccard(a, b)      for a, b in zip(s1_addrs, c_addrs)],  dtype=np.float32)
    addr_lev             = np.array([_lev_norm(a, b)     for a, b in zip(s1_addrs, c_addrs)],  dtype=np.float32)
    addr_len_ratio       = np.array([_len_ratio(a, b)    for a, b in zip(s1_addrs, c_addrs)],  dtype=np.float32)
    addr_common          = np.array([_common_tokens(a, b) for a, b in zip(s1_addrs, c_addrs)], dtype=np.float32)

    country_match        = np.array([int(a == b) for a, b in zip(s1_ctries, c_ctries)], dtype=np.float32)
    both_addr_empty      = np.array([int(not a and not b) for a, b in zip(s1_addrs, c_addrs)], dtype=np.float32)
    one_addr_empty       = np.array([int(bool(a) != bool(b)) for a, b in zip(s1_addrs, c_addrs)], dtype=np.float32)

    print(f"    string features done  ({time.perf_counter()-t0:.1f}s)", flush=True)

    print("  Computing TF-IDF cosine (name) …", flush=True)
    t0 = time.perf_counter()
    name_cosine = _batch_cosine(name_vec, s1_names, c_names)
    print(f"    name cosine done  ({time.perf_counter()-t0:.1f}s)", flush=True)

    print("  Computing TF-IDF cosine (address) …", flush=True)
    t0 = time.perf_counter()
    addr_cosine = _batch_cosine(addr_vec, s1_addrs, c_addrs)
    print(f"    addr cosine done  ({time.perf_counter()-t0:.1f}s)", flush=True)

    return pd.DataFrame({
        "name_jaccard":          name_jaccard,
        "name_levenshtein_norm": name_lev,
        "name_jaro_winkler":     name_jw,
        "name_cosine_tfidf":     name_cosine,
        "name_len_ratio":        name_len_ratio,
        "name_common_tokens":    name_common,
        "addr_jaccard":          addr_jaccard,
        "addr_levenshtein_norm": addr_lev,
        "addr_cosine_tfidf":     addr_cosine,
        "addr_len_ratio":        addr_len_ratio,
        "addr_common_tokens":    addr_common,
        "country_match":         country_match,
        "both_addr_empty":       both_addr_empty,
        "one_addr_empty":        one_addr_empty,
    })


# ── Label construction ────────────────────────────────────────────────────────

def build_labels(
    pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> np.ndarray:
    """
    Returns int8 array: 1 if the candidate is a true match, 0 otherwise.
    ground_truth columns: source1_entity_id, matched_entity_ids (comma-sep)
    """
    gt_map: Dict[str, set] = {}
    for row in ground_truth[["source1_entity_id", "matched_entity_ids"]].itertuples(
        index=False
    ):
        m = row.matched_entity_ids
        if m and str(m).strip():
            gt_map[row.source1_entity_id] = {
                x.strip() for x in str(m).split(",") if x.strip()
            }
    labels = np.array(
        [int(cid in gt_map.get(sid, set()))
         for sid, cid in pairs[["source1_entity_id", "candidate_entity_id"]].itertuples(
             index=False, name=None
         )],
        dtype=np.int8,
    )
    return labels


# ── Normalised pool builder ───────────────────────────────────────────────────

def _load_normalised(path: str) -> pd.DataFrame:
    """Load and normalize a source TSV through Polars streaming."""
    scan = (
        pl.scan_csv(
            path,
            separator="\t",
            infer_schema_length=0,
            schema_overrides={
                "entity_id": pl.String,
                "business_name": pl.String,
                "business_address": pl.String,
                "country": pl.String,
            },
        )
        .select(["entity_id", "business_name", "business_address", "country"])
        .with_columns(
            norm_name=_polars_normalize_expr(
                pl.col("business_name"), _NAME_EXPANSIONS
            ),
            norm_addr=_polars_normalize_expr(
                pl.col("business_address"), _ADDR_EXPANSIONS
            ),
        )
    )
    df = scan.collect(engine="streaming").to_pandas()
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    print(f"  Normalising {os.path.basename(path)} ({len(df):,} rows) …", flush=True)
    t0 = time.perf_counter()
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["norm_addr"] = df["business_address"].apply(normalize_address)
    print(f"    done  ({time.perf_counter()-t0:.1f}s)", flush=True)
    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading candidates …", flush=True)
    cands = (
        pl.read_csv(
            CANDS_PATH,
            separator="\t",
            empty_string_is_null=True,
            schema_overrides={
                "source1_entity_id": pl.String,
                "candidate_entity_ids": pl.String,
            },
        )
        .with_columns(
            pl.col("candidate_entity_ids").fill_null("").str.split(",")
        )
        .explode("candidate_entity_ids")
        .with_columns(
            pl.col("candidate_entity_ids").str.strip_chars().alias(
                "candidate_entity_id"
            )
        )
        .filter(pl.col("candidate_entity_id") != "")
        .select(["source1_entity_id", "candidate_entity_id"])
    )
    pairs = cands.to_pandas()
    print(f"  {len(pairs):,} candidate pairs expanded with Polars", flush=True)

    print("\nLoading source files …", flush=True)
    s1   = _load_normalised(os.path.join(DATA_TRAIN, "train_source1.tsv"))
    s2   = _load_normalised(os.path.join(DATA_TRAIN, "train_source2.tsv"))
    s3   = _load_normalised(os.path.join(DATA_TRAIN, "train_source3.tsv"))
    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    gt = pl.read_csv(
        os.path.join(DATA_TRAIN, "train_ground_truth.tsv"),
        separator="\t",
        empty_string_is_null=True,
        schema_overrides={
            "source1_entity_id": pl.String,
            "matched_entity_ids": pl.String,
        },
    ).fill_null("").to_pandas()

    # Filter pairs to only rows present in both s1 and pool
    s1_ids   = set(s1["entity_id"])
    pool_ids = set(pool["entity_id"])
    before   = len(pairs)
    pairs    = pairs[
        pairs["source1_entity_id"].isin(s1_ids) &
        pairs["candidate_entity_id"].isin(pool_ids)
    ].reset_index(drop=True)
    print(f"  Pairs after ID filter: {len(pairs):,} (dropped {before - len(pairs):,})", flush=True)
    covered_s1 = pairs["source1_entity_id"].nunique()
    coverage = covered_s1 / max(1, len(s1))
    print(
        f"  Candidate S1 coverage: {covered_s1:,}/{len(s1):,} "
        f"({coverage:.2%})",
        flush=True,
    )
    if coverage < 0.99:
        print(
            "  WARNING: this is a sampled/incomplete candidate file; "
            "do not use it as the final training dataset.",
            flush=True,
        )

    print("\nFitting TF-IDF vectorisers …", flush=True)
    t0 = time.perf_counter()
    candidate_pool = pool[pool["entity_id"].isin(
        set(pairs["candidate_entity_id"])
    )]
    name_vec  = _fit_tfidf_for_candidates(s1, candidate_pool, "norm_name")
    addr_vec  = _fit_tfidf_for_candidates(s1, candidate_pool, "norm_addr")
    print(f"  Fitted in {time.perf_counter()-t0:.1f}s", flush=True)

    print("\nComputing feature matrix …", flush=True)
    feat_df = compute_feature_matrix(pairs, s1, pool, name_vec, addr_vec)

    print("\nBuilding labels …", flush=True)
    labels = build_labels(pairs, gt)
    pos = labels.sum()
    print(f"  Positives: {pos:,} / {len(labels):,}  ({pos/len(labels)*100:.2f}%)", flush=True)

    # Save
    meta_cols = pd.DataFrame({
        "source1_entity_id":   pairs["source1_entity_id"].values,
        "candidate_entity_id": pairs["candidate_entity_id"].values,
    })
    out_feats  = os.path.join(OUTPUT_DIR, "train_features.parquet")
    out_labels = os.path.join(OUTPUT_DIR, "train_labels.parquet")
    out_meta   = os.path.join(OUTPUT_DIR, "train_meta.parquet")
    out_nvec   = os.path.join(OUTPUT_DIR, "tfidf_name.pkl")
    out_avec   = os.path.join(OUTPUT_DIR, "tfidf_addr.pkl")

    feat_df.to_parquet(out_feats, index=False)
    pd.DataFrame({"label": labels}).to_parquet(out_labels, index=False)
    meta_cols.to_parquet(out_meta, index=False)
    with open(out_nvec, "wb") as f:
        pickle.dump(name_vec, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_avec, "wb") as f:
        pickle.dump(addr_vec, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_version": FEATURE_VERSION,
                "candidate_path": CANDS_PATH,
                "candidate_pairs": len(pairs),
                "candidate_s1_entities": int(covered_s1),
                "source1_entities": len(s1),
                "candidate_coverage": coverage,
                "positive_labels": int(pos),
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print(f"  {out_feats}")
    print(f"  {out_labels}")
    print(f"  {out_meta}")
    print(f"  {out_nvec}")
    print(f"  {out_avec}")
