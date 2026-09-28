# Business Entity Resolution — ML Challenge 2026

End-to-end pipeline that resolves business records across three independent
noisy sources using a two-stage approach: candidate generation (blocking) +
pairwise matching classifier.

---

## Environment

| Item | Value |
|---|---|
| Python | 3.11 (Anaconda `torch_env`) |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB VRAM, CUDA 12.1 |
| Key libs | PyTorch 2.5.1+cu121, scikit-learn, polars, pandas, rapidfuzz, xgboost |

All commands below assume the repo root is the working directory:

```
cd C:\Users\uditj\Downloads\aws_hackathon
```

Install dependencies:

```powershell
C:\Users\uditj\anaconda3\envs\torch_env\python.exe -m pip install -r code/business_entity_resolution/requirements.txt
```

---

## Quick start (dev / GitHub clone)

After cloning, set these two env vars once per PowerShell session:

```powershell
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
```

### Smoke test — ~1 100 S1 rows, finishes in a few minutes

```powershell
& "C:\Users\uditj\anaconda3\envs\torch_env\python.exe" `
    code/business_entity_resolution/src/blocking_v2.py `
    --sample 0.0005 `
    --max-candidates 200
```

### Medium dev run — ~11 000 S1 rows

```powershell
& "C:\Users\uditj\anaconda3\envs\torch_env\python.exe" `
    code/business_entity_resolution/src/blocking_v2.py `
    --sample 0.005 `
    --max-candidates 200
```

### Skip GPU Layer C (faster, useful for recall-only benchmarks)

Append `--no-layer-c` to any command above:

```powershell
& "C:\Users\uditj\anaconda3\envs\torch_env\python.exe" `
    code/business_entity_resolution/src/blocking_v2.py `
    --sample 0.0005 `
    --max-candidates 200 `
    --no-layer-c
```

> **Note:** The first run streams and normalises S2/S3 (~4 min, ~500 MB per file).
> Subsequent runs load from `output/cache/` instantly.
> Ask a teammate for the pre-built cache Parquets to skip the first-run cost.

---

## Reproducing outputs end-to-end

Set the encoding env var once per session (avoids cp1252 errors on Windows):

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### Step 1 — Blocking / candidate generation

Builds `output/candidates/v2_hybrid/candidate_pairs.tsv` and
`output/candidates/v2_hybrid/scored_candidates.parquet`.

First run normalises S2/S3 (~4 min, cached afterwards):

```powershell
C:\Users\uditj\anaconda3\envs\torch_env\python.exe `
    code/business_entity_resolution/src/blocking_v2.py `
    --sample 1.0 --no-layer-c
```

Re-run with GPU pairwise re-ranking (Layer C) for best recall ordering:

```powershell
C:\Users\uditj\anaconda3\envs\torch_env\python.exe `
    code/business_entity_resolution/src/blocking_v2.py `
    --sample 1.0
```

Dev / smoke iterations (fast):

```powershell
# ~1 100 S1 rows, <5 min
$env:SAMPLE = "0.0005"
C:\Users\uditj\anaconda3\envs\torch_env\python.exe `
    code/business_entity_resolution/src/blocking_v2.py --no-layer-c
```

### Step 2 — Feature engineering  *(in progress)*

```powershell
C:\Users\uditj\anaconda3\envs\torch_env\python.exe `
    code/business_entity_resolution/src/features.py
```

### Step 3 — Train classifier  *(in progress)*

```powershell
C:\Users\uditj\anaconda3\envs\torch_env\python.exe `
    code/business_entity_resolution/src/train.py
```

### Step 4 — Inference + output files  *(in progress)*

```powershell
C:\Users\uditj\anaconda3\envs\torch_env\python.exe `
    code/business_entity_resolution/src/predict.py
```

### Step 5 — Validate submission

```powershell
C:\Users\uditj\anaconda3\envs\torch_env\python.exe `
    student_resource/utils/validate_submission.py `
    --matching output/matching_results.tsv `
    --candidate output/candidate_pairs.tsv `
    --test-dir student_resource/dataset/test
```

---

## Source layout

```
src/
  preprocess.py          # normalize_name(), normalize_address(), tokenize(), jaccard()
  blocking_v2.py         # Layer A (exact keys) + Layer B (n-gram) + Layer C (GPU)  ← active
  blocking.py            # legacy bucket+TF-IDF (kept for reference)
  features.py            # pairwise feature computation          (TODO)
  train.py               # XGBoost training + threshold tuning   (TODO)
  predict.py             # test-set inference + output writing   (TODO)
  test_preprocess.py     # 39 unit tests for preprocess.py
  eda.py                 # exploratory data analysis script
```

---

## Caches and checkpoints

| Path | Contents | Safe to delete? |
|---|---|---|
| `output/cache/` | Polars-normalised Parquet (stamped per source file mtime) | Yes — rebuilt in ~4 min |
| `output/checkpoints/v2_hybrid/` | Per-country atomic JSON/Parquet checkpoints | Yes — rebuilt on next blocking run |
| `output/candidates/v2_hybrid/` | Final `candidate_pairs.tsv` + `scored_candidates.parquet` | Yes — rebuilt by blocking step |

Caches are **not committed to git**. To share with a teammate, upload
`output/cache/` to Google Drive / S3 and drop into the same path on their machine.

---

## Run tests

```powershell
C:\Users\uditj\anaconda3\envs\torch_env\python.exe -m pytest `
    code/business_entity_resolution/src/test_preprocess.py -v
```

Expected: 39 passed.

---

## License

MIT — see [`LICENSE`](../../LICENSE) at the repo root.
