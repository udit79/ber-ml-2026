# Business Entity Resolution — ML Challenge 2026

An end-to-end machine learning pipeline that resolves business records across three independent, noisy data sources using a classical two-stage approach: **candidate generation (blocking)** followed by a **pairwise XGBoost matching classifier**.

> **Evaluation metric:** Macro-average **F₀.₅** (precision-weighted — false merges are penalised twice as heavily as missed links)  
> **Validation F₀.₅ achieved: 0.8248** (threshold = 0.95)

---

## Problem Statement

Given business records from three independent, noisy sources (S1, S2, S3), determine which records in S2 and S3 refer to the same real-world business as each S1 entity.

| Split | S1 rows | S2 rows | S3 rows |
|-------|---------|---------|---------|
| Train | 2.2 M   | 5.0 M   | 5.3 M   |
| Test  | 1.7 M   | 4.9 M   | 5.1 M   |

**Countries:** US + India in training; US + India + **France** (completely unseen) in test.  
**Key noise patterns:** ALL-CAPS (~19% of S2), legal suffix variants (Pvt Ltd / Private Limited / LLC), address abbreviations (Rd/Road, St/Street), Devanagari/Tamil/French scripts, empty addresses (~3.3% of S2/S3).

---

## Architecture

```
train_source1/2/3.tsv
        │
        ▼
 [Text Normalisation]          preprocess.py
   NFKC · lowercase · legal-suffix expansion
   address-abbrev expansion · punct strip
        │
        ▼
 [Blocking — 3 Layers]         blocking_v2.py
   Layer A  13 deterministic exact-key passes (name-core+city, …)
   Layer B  Inverted char n-gram index + sparse TF-IDF cosine
   Layer C  GPU pairwise re-ranking (optional, RTX 4050 / CUDA 12.1)
        │
        ▼
 candidate_pairs.tsv
        │
        ▼
 [Pairwise Feature Engineering] features.py
   Jaccard · Levenshtein · Jaro-Winkler · TF-IDF cosine (name + address)
   length ratios · common token counts · country match · empty-address flags
        │
        ▼
 [XGBoost Classifier]          train.py
   Group-aware 80/20 split · scale_pos_weight · threshold sweep
   Best threshold @ max macro F₀.₅
        │
        ▼
 [Inference + Output]          predict.py
        │
        ▼
 matching_results.tsv          ← leaderboard upload
 candidate_pairs.tsv           ← submission zip
```

---

## Repository Layout

```
.
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt
│       └── src/
│           ├── preprocess.py      # normalize_name(), normalize_address(), tokenize(), jaccard()
│           ├── blocking_v2.py     # Layer A + B + C hybrid blocking  ← active
│           ├── features.py        # pairwise feature computation
│           ├── train.py           # XGBoost training + threshold tuning
│           └── predict.py         # test-set inference + output writing
├── output/
│   ├── candidates/                # candidate_pairs.tsv + scored_candidates.parquet
│   ├── features/                  # train_features.parquet, TF-IDF vectorisers
│   └── model/
│       ├── xgb_model.json
│       ├── threshold.txt          # 0.95
│       └── val_metrics.json       # best F₀.₅ = 0.8248
├── student_resource/              # competition dataset + utilities (not committed)
├── business-entity-resolution-plan.md
└── README.md
```

> **`student_resource/` and `output/cache/` are not committed to git.**  
> Contact a team member for access to the pre-built Parquet caches (saves ~4 min on first blocking run).

---

## Quick Start

### Prerequisites

- Python 3.11 (Anaconda `torch_env` recommended)
- NVIDIA GPU with CUDA 12.1 (optional — Layer C only; CPU fallback is automatic)

### Install dependencies

```powershell
pip install -r code/business_entity_resolution/requirements.txt
```

### Environment (Windows — set once per session)

```powershell
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
```

---

## Running the Pipeline

### Step 1 — Blocking / Candidate Generation

Builds `output/candidates/v2_hybrid/candidate_pairs.tsv`.  
First run normalises S2/S3 (~4 min); subsequent runs load from `output/cache/`.

```powershell
# Full run, CPU-only (fastest for recall benchmarks)
python code/business_entity_resolution/src/blocking_v2.py --sample 1.0 --no-layer-c

# Full run with GPU re-ranking (best recall ordering)
python code/business_entity_resolution/src/blocking_v2.py --sample 1.0

# Smoke test — ~1 100 S1 rows, finishes in a few minutes
python code/business_entity_resolution/src/blocking_v2.py --sample 0.0005 --max-candidates 200
```

### Step 2 — Feature Engineering

```powershell
python code/business_entity_resolution/src/features.py
```

Writes `output/features/train_features.parquet`, `train_labels.parquet`, `tfidf_name.pkl`, `tfidf_addr.pkl`.

### Step 3 — Train Classifier

```powershell
python code/business_entity_resolution/src/train.py
```

Writes `output/model/xgb_model.json`, `output/model/threshold.txt`, `output/model/val_metrics.json`.

### Step 4 — Inference

```powershell
python code/business_entity_resolution/src/predict.py
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### Step 5 — Validate Submission

```powershell
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

Expected: `PASS`.

---

## Features (14 total)

| Feature | Description |
|---------|-------------|
| `name_jaccard` | Token Jaccard on normalised names |
| `name_levenshtein_norm` | Normalised edit distance on names |
| `name_jaro_winkler` | Jaro-Winkler on normalised names |
| `name_cosine_tfidf` | TF-IDF cosine (char n-gram vectoriser) |
| `name_len_ratio` | min / max character-length ratio |
| `name_common_tokens` | Raw count of shared name tokens |
| `addr_jaccard` | Token Jaccard on normalised addresses |
| `addr_levenshtein_norm` | Normalised edit distance on addresses |
| `addr_cosine_tfidf` | TF-IDF cosine on addresses |
| `addr_len_ratio` | min / max character-length ratio |
| `addr_common_tokens` | Raw count of shared address tokens |
| `country_match` | 1 if country strings match (lowercased) |
| `both_addr_empty` | 1 if both sides have no address |
| `one_addr_empty` | 1 if exactly one side has no address |

---

## Model & Results

| Item | Value |
|------|-------|
| Classifier | XGBoost (binary:logistic) |
| Validation split | 20% of S1 entities (group-aware, no pair-level leakage) |
| Class imbalance handling | `scale_pos_weight` |
| Threshold selection | Sweep 0.05 → 0.95; pick argmax macro F₀.₅ |
| **Best threshold** | **0.95** |
| **Validation macro F₀.₅** | **0.8248** |

---

## Blocking Strategy

`blocking_v2.py` unions three independent retrieval layers per S1 entity:

| Layer | Method | Strength |
|-------|--------|----------|
| **A** | 13 deterministic exact-key passes (name core + city, street number + city, …) | Zero-latency O(N) groupby; perfect recall on exact variants |
| **B** | Inverted character 3-gram index + sparse TF-IDF cosine scoring | Handles abbreviations, typos, transliterations |
| **C** | GPU pairwise re-ranking of Layer-B output (FP16, auto CPU fallback) | Improves score ordering for threshold-sensitive decisions |

**Cache & Checkpoint system:** Source normalisation is cached as Parquet (invalidated by source file mtime). Each country's Layer-A/B/C results are checkpointed atomically — interrupted runs resume without repeating completed countries.

---

## Text Normalisation Highlights

`preprocess.py` handles:
- **NFKC Unicode normalisation** — composed vs. decomposed diacritics (critical for French test records)
- **Lowercase** — resolves ~19% ALL-CAPS in S2
- **Legal suffix expansion:** `Pvt Ltd → private limited`, `Corp → corporation`, `LLC → llc`, French forms (`SARL`, `SAS`, `SASU`)
- **Address abbreviation expansion:** `Rd → road`, `St → street`, `Ave → avenue`, 20+ mappings
- **ASCII-only punctuation stripping** — preserves Devanagari/Tamil combining marks (Unicode category M)

---

## Caches (not committed)

| Path | Contents | Rebuild time |
|------|----------|--------------|
| `output/cache/` | Polars-normalised Parquet per source (stamped by mtime) | ~4 min |
| `output/checkpoints/v2_hybrid/` | Per-country atomic JSON/Parquet checkpoints | Rebuilt by blocking |
| `output/candidates/v2_hybrid/` | `candidate_pairs.tsv` + `scored_candidates.parquet` | Rebuilt by blocking |

---

## Constraints

| Constraint | Detail |
|------------|--------|
| No external data | No APIs, geocoding, or business registries |
| Model licence | MIT or Apache 2.0 |
| Model size | ≤ 8 B parameters (XGBoost complies) |
| Output format | Tab-separated; exact column names; every S1 row present |
| France handling | Fully unseen in training — blocking is language-agnostic by design |

---

## License

[MIT](LICENSE)
