"""
train.py — Train XGBoost matching classifier for Business Entity Resolution.

Pipeline
--------
1. Load feature matrix + labels from output/features/
2. Group-aware 80/20 split by S1 entity (no pair-level leakage)
3. Train XGBoost binary classifier with scale_pos_weight for class imbalance
4. Sweep decision threshold 0.05–0.95, pick threshold maximising macro F_0.5
5. Save model + threshold to output/model/

Outputs
-------
output/model/xgb_model.json
output/model/threshold.txt
output/model/val_metrics.json

Usage
-----
python code/business_entity_resolution/src/train.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

_ROOT      = os.path.abspath(os.path.join(_SRC_DIR, "..", "..", ".."))
FEAT_DIR   = os.path.join(_ROOT, "output", "features")
MODEL_DIR  = os.path.join(_ROOT, "output", "model")

VAL_FRAC   = 0.20      # fraction of S1 entities held out for validation
RANDOM_STATE = 42


# ── F_0.5 helpers ─────────────────────────────────────────────────────────────

def _f05_entity(pred_ids: set, true_ids: set) -> float:
    """F_0.5 for a single S1 entity."""
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids:
        return 0.0   # false merge on singleton
    if not pred_ids:
        # precision undefined; recall = 0  → F_0.5 = 0
        return 0.0
    tp = len(pred_ids & true_ids)
    prec = tp / len(pred_ids)
    rec  = tp / len(true_ids)
    if prec + rec == 0:
        return 0.0
    return (1.25 * prec * rec) / (0.25 * prec + rec)


def macro_f05(
    meta: pd.DataFrame,
    proba: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute macro-average F_0.5 exactly as the leaderboard does.
    Every S1 entity in meta contributes one score; singletons score 1.0
    when correctly predicted empty, 0.0 otherwise.
    """
    preds = (proba >= threshold).astype(int)
    # Build per-entity sets
    gt_map: Dict[str, set] = {}
    pred_map: Dict[str, set] = {}
    s1_ids_all: set = set(meta["source1_entity_id"].unique())

    for i, row in meta.iterrows():
        s1_id  = row["source1_entity_id"]
        cand   = row["candidate_entity_id"]
        if labels[i] == 1:
            gt_map.setdefault(s1_id, set()).add(cand)
        if preds[i] == 1:
            pred_map.setdefault(s1_id, set()).add(cand)

    scores: List[float] = []
    for s1_id in s1_ids_all:
        true_ids = gt_map.get(s1_id, set())
        pred_ids = pred_map.get(s1_id, set())
        scores.append(_f05_entity(pred_ids, true_ids))

    return float(np.mean(scores))


# ── Group-aware train/val split ───────────────────────────────────────────────

def group_split(
    meta: pd.DataFrame,
    feat: np.ndarray,
    labels: np.ndarray,
    val_frac: float = VAL_FRAC,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           pd.DataFrame, pd.DataFrame]:
    """
    Split by S1 entity group so no S1 entity appears in both train and val.
    Returns (X_tr, X_val, y_tr, y_val, meta_tr, meta_val).
    """
    groups = meta["source1_entity_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    train_idx, val_idx = next(gss.split(feat, labels, groups=groups))
    return (
        feat[train_idx], feat[val_idx],
        labels[train_idx], labels[val_idx],
        meta.iloc[train_idx].reset_index(drop=True),
        meta.iloc[val_idx].reset_index(drop=True),
    )


# ── Model training ────────────────────────────────────────────────────────────

def train_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    scale_pos_weight: float,
    feature_names: List[str],
) -> xgb.XGBClassifier:
    build_info = xgb.build_info()
    cuda_enabled = str(build_info.get("USE_CUDA", "0")).lower() in {
        "1", "true", "yes"
    }
    device = "cuda" if cuda_enabled else "cpu"
    print(f"  XGBoost device: {device}", flush=True)
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        device=device,
        early_stopping_rounds=30,
    )
    # Small eval set for early stopping — use 10% of train
    n_es = max(1, int(len(X_tr) * 0.10))
    rng  = np.random.default_rng(RANDOM_STATE)
    es_idx = rng.choice(len(X_tr), size=n_es, replace=False)
    tr_idx = np.setdiff1d(np.arange(len(X_tr)), es_idx)

    model.fit(
        X_tr[tr_idx], y_tr[tr_idx],
        eval_set=[(X_tr[es_idx], y_tr[es_idx])],
        verbose=50,
    )
    model.get_booster().feature_names = feature_names
    return model


# ── Threshold sweep ───────────────────────────────────────────────────────────

def find_best_threshold(
    meta_val: pd.DataFrame,
    proba_val: np.ndarray,
    labels_val: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Sweep thresholds and return (best_threshold, {threshold: f05}) dict.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.025)

    results: Dict[str, float] = {}
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        f = macro_f05(meta_val, proba_val, labels_val, float(t))
        results[f"{t:.3f}"] = round(f, 6)
        if f > best_f:
            best_f = f
            best_t = float(t)

    return best_t, results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading features …", flush=True)
    manifest_path = os.path.join(FEAT_DIR, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        coverage = float(manifest.get("candidate_coverage", 0.0))
        print(
            f"  Candidate coverage: {coverage:.2%} "
            f"({manifest.get('candidate_pairs', 0):,} pairs)",
            flush=True,
        )
        if coverage < 0.99:
            print(
                "  WARNING: training features come from a sampled/incomplete "
                "candidate file; use only for pipeline validation.",
                flush=True,
            )
    feat_df  = pd.read_parquet(os.path.join(FEAT_DIR, "train_features.parquet"))
    label_df = pd.read_parquet(os.path.join(FEAT_DIR, "train_labels.parquet"))
    meta_df  = pd.read_parquet(os.path.join(FEAT_DIR, "train_meta.parquet"))

    feature_names = feat_df.columns.tolist()
    X      = feat_df.values.astype(np.float32)
    y      = label_df["label"].values.astype(np.int8)

    print(f"  {len(X):,} pairs  |  positives: {y.sum():,} ({y.mean()*100:.2f}%)", flush=True)

    print("\nSplitting train/val by S1 entity …", flush=True)
    X_tr, X_val, y_tr, y_val, meta_tr, meta_val = group_split(meta_df, X, y)
    pos_tr  = y_tr.sum()
    neg_tr  = len(y_tr) - pos_tr
    spw     = float(neg_tr) / max(1, pos_tr)
    print(f"  Train: {len(X_tr):,} pairs  pos={pos_tr:,}  neg={neg_tr:,}  scale_pos_weight={spw:.1f}", flush=True)
    print(f"  Val  : {len(X_val):,} pairs  pos={y_val.sum():,}", flush=True)

    print("\nTraining XGBoost …", flush=True)
    t0    = time.perf_counter()
    model = train_xgb(X_tr, y_tr, spw, feature_names)
    print(f"  Training done  ({time.perf_counter()-t0:.1f}s)", flush=True)

    print("\nValidation probabilities …", flush=True)
    proba_val = model.predict_proba(X_val)[:, 1]

    print("Sweeping thresholds …", flush=True)
    best_t, sweep = find_best_threshold(meta_val, proba_val, y_val)
    best_f = sweep[f"{best_t:.3f}"]
    print(f"\n  Best threshold : {best_t:.3f}")
    print(f"  Best macro F_0.5: {best_f:.4f}")

    # Also print a few nearby thresholds for context
    print("\n  Threshold sweep (selected):")
    for k, v in sorted(sweep.items()):
        marker = " ◄" if float(k) == best_t else ""
        print(f"    {k}  →  {v:.4f}{marker}")

    # Save
    model_path  = os.path.join(MODEL_DIR, "xgb_model.json")
    thresh_path = os.path.join(MODEL_DIR, "threshold.txt")
    sweep_path  = os.path.join(MODEL_DIR, "val_metrics.json")

    model.save_model(model_path)
    with open(thresh_path, "w") as f:
        f.write(str(best_t))
    with open(sweep_path, "w") as f:
        json.dump({"best_threshold": best_t, "best_f05": best_f, "sweep": sweep}, f, indent=2)

    print("\nSaved:")
    print(f"  {model_path}")
    print(f"  {thresh_path}")
    print(f"  {sweep_path}")
