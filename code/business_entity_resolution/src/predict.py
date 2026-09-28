"""
predict.py — Test-set inference for Business Entity Resolution.

Pipeline
--------
1. Run blocking on test set → test candidate pairs
2. Load fitted TF-IDF vectorisers from output/features/
3. Compute feature matrix for all test candidate pairs
4. Load trained XGBoost model + threshold from output/model/
5. Predict match/no-match; aggregate per S1 entity
6. Write output/matching_results.tsv and output/candidate_pairs.tsv

Outputs
-------
output/matching_results.tsv   — final matches (leaderboard upload)
output/candidate_pairs.tsv    — blocking candidate set (submitted in zip)

Usage
-----
$env:PYTHONIOENCODING = "utf-8"
python code/business_entity_resolution/src/predict.py
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)
from preprocess import normalize_name, normalize_address          # noqa: E402
from features import compute_feature_matrix, FEATURE_COLUMNS     # noqa: E402
from blocking_v2 import build_candidates, write_candidate_pairs  # noqa: E402

_ROOT      = os.path.abspath(os.path.join(_SRC_DIR, "..", "..", ".."))
FEAT_DIR   = os.path.join(_ROOT, "output", "features")
MODEL_DIR  = os.path.join(_ROOT, "output", "model")
OUTPUT_DIR = _ROOT
DATA_TEST  = os.path.join(_ROOT, "student_resource", "dataset", "test")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_normalised(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    print(f"  Normalising {os.path.basename(path)} ({len(df):,} rows) …", flush=True)
    t0 = time.perf_counter()
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["norm_addr"] = df["business_address"].apply(normalize_address)
    print(f"    done  ({time.perf_counter()-t0:.1f}s)", flush=True)
    return df


def _expand_candidates(cands: Dict[str, List[str]]) -> pd.DataFrame:
    """Flatten {s1_id: [cand_ids]} → DataFrame of pairs."""
    rows: List[dict] = []
    for s1_id, cand_ids in cands.items():
        for cid in cand_ids:
            rows.append({"source1_entity_id": s1_id, "candidate_entity_id": cid})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id"]
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load test source files ─────────────────────────────────────────────
    print("Loading test sources …", flush=True)
    s1_test = pd.read_csv(
        os.path.join(DATA_TEST, "test_source1.tsv"), sep="\t", dtype=str
    ).fillna("")
    S2_PATH = os.path.join(DATA_TEST, "test_source2.tsv")
    S3_PATH = os.path.join(DATA_TEST, "test_source3.tsv")
    print(f"  S1={len(s1_test):,}", flush=True)

    # ── Blocking on test set ───────────────────────────────────────────────
    print("\nRunning blocking on test set …", flush=True)
    test_cands, _ = build_candidates(
        s1_test, S2_PATH, S3_PATH,
        max_candidates=200,
        run_layer_c=True,
    )

    # Write candidate_pairs.tsv (required submission file)
    cand_out = os.path.join(OUTPUT_DIR, "output", "candidate_pairs.tsv")
    os.makedirs(os.path.join(OUTPUT_DIR, "output"), exist_ok=True)
    write_candidate_pairs(test_cands, s1_test, cand_out)

    # ── Normalise test pool ────────────────────────────────────────────────
    print("\nNormalising test pool …", flush=True)
    s2_test   = _load_normalised(S2_PATH)
    s3_test   = _load_normalised(S3_PATH)
    pool_test = pd.concat([s2_test, s3_test], ignore_index=True)
    del s2_test, s3_test

    # Normalise S1 test
    print("  Normalising S1 test …", flush=True)
    s1_test["norm_name"] = s1_test["business_name"].apply(normalize_name)
    s1_test["norm_addr"] = s1_test["business_address"].apply(normalize_address)

    # ── Expand candidates to pairs ─────────────────────────────────────────
    print("\nExpanding candidates to pairs …", flush=True)
    pairs = _expand_candidates(test_cands)
    print(f"  {len(pairs):,} test candidate pairs", flush=True)

    if pairs.empty:
        print("  WARNING: No candidate pairs — writing empty results", flush=True)
        pd.DataFrame({
            "source1_entity_id": s1_test["entity_id"],
            "matched_entity_ids": "",
        }).to_csv(
            os.path.join(OUTPUT_DIR, "output", "matching_results.tsv"),
            sep="\t", index=False,
        )
        sys.exit(0)

    # Filter pairs to IDs that exist in test pool
    pool_ids = set(pool_test["entity_id"])
    s1_ids   = set(s1_test["entity_id"])
    before   = len(pairs)
    pairs    = pairs[
        pairs["source1_entity_id"].isin(s1_ids) &
        pairs["candidate_entity_id"].isin(pool_ids)
    ].reset_index(drop=True)
    print(f"  After ID filter: {len(pairs):,} (dropped {before - len(pairs):,})", flush=True)

    # ── Load TF-IDF vectorisers ────────────────────────────────────────────
    print("\nLoading TF-IDF vectorisers …", flush=True)
    with open(os.path.join(FEAT_DIR, "tfidf_name.pkl"), "rb") as f:
        name_vec = pickle.load(f)
    with open(os.path.join(FEAT_DIR, "tfidf_addr.pkl"), "rb") as f:
        addr_vec = pickle.load(f)

    # ── Compute feature matrix ─────────────────────────────────────────────
    print("\nComputing test feature matrix …", flush=True)
    feat_df = compute_feature_matrix(pairs, s1_test, pool_test, name_vec, addr_vec)
    X_test  = feat_df[FEATURE_COLUMNS].values.astype(np.float32)

    # ── Load model + threshold ─────────────────────────────────────────────
    print("\nLoading model …", flush=True)
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    with open(os.path.join(MODEL_DIR, "threshold.txt")) as f:
        threshold = float(f.read().strip())
    print(f"  Threshold: {threshold}", flush=True)

    # ── Inference ──────────────────────────────────────────────────────────
    print("\nRunning inference …", flush=True)
    t0    = time.perf_counter()
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    print(f"  {preds.sum():,} positive predictions  ({time.perf_counter()-t0:.1f}s)", flush=True)

    # ── Aggregate per S1 entity ────────────────────────────────────────────
    print("\nAggregating per S1 entity …", flush=True)
    match_map: Dict[str, List[str]] = {eid: [] for eid in s1_test["entity_id"]}
    for i, row in enumerate(
        tqdm(pairs.itertuples(index=False), total=len(pairs), desc="  aggregating", ncols=80)
    ):
        if preds[i] == 1:
            match_map[row.source1_entity_id].append(row.candidate_entity_id)

    # Deduplicate (should already be unique from blocking, but be safe)
    for k in match_map:
        match_map[k] = list(dict.fromkeys(match_map[k]))

    n_matched   = sum(1 for v in match_map.values() if v)
    n_singleton = sum(1 for v in match_map.values() if not v)
    print(f"  Entities with matches  : {n_matched:,}", flush=True)
    print(f"  Predicted singletons   : {n_singleton:,}", flush=True)

    # ── Write matching_results.tsv ─────────────────────────────────────────
    results_out = os.path.join(OUTPUT_DIR, "output", "matching_results.tsv")
    rows = [
        {
            "source1_entity_id": eid,
            "matched_entity_ids": ",".join(match_map[eid]),
        }
        for eid in s1_test["entity_id"]
    ]
    pd.DataFrame(rows).to_csv(results_out, sep="\t", index=False)
    print(f"\nWritten: {results_out}  ({len(rows):,} rows)", flush=True)
    print(f"Written: {cand_out}  ({len(test_cands):,} rows)", flush=True)

    print("\nNext: run validate_submission.py to check format before uploading.", flush=True)
    print(
        "  python student_resource/utils/validate_submission.py"
        " --matching output/matching_results.tsv"
        " --candidate output/candidate_pairs.tsv"
        " --test-dir student_resource/dataset/test",
        flush=True,
    )
