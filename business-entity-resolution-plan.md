# Business Entity Resolution — ML Challenge 2026 Plan

## EDA Findings (Sub-Task 1 — COMPLETE)

Key numbers to carry into implementation:

| Metric | Value |
|---|---|
| Train S1 / S2 / S3 | 2.2M / 5.0M / 5.3M rows |
| Test S1 / S2 / S3 | 1.7M / 4.9M / 5.1M rows |
| Singletons (train) | 5.6% (123K entities) |
| Most common cardinality | 3 matches (24.1%), then 4 (21.9%) |
| ALL-CAPS in S2 | ~19% — lowercasing is critical |
| Non-ASCII names in S2 | 15%, S3 11% — multi-script |
| Empty addresses S2/S3 | ~3.3% — must default to 0.0 |
| France in test | 15% — completely unseen in training |
| Legal suffixes dominant | Limited 1.7M, LLC 1.46M, Private 1.6M |
| Address abbrevs | Rd vs Road (676K vs 1.75M), St vs Street (681K vs 1M) |
| GPU env | RTX 4050, torch_env, CuPy 14.2.0 + CUDA 12.1 working |

Run all scripts with: `C:\Users\uditj\anaconda3\envs\torch_env\python.exe`


## Top-Level Overview

**Goal:** Build an ML pipeline that, given business records from 3 independent noisy sources, determines which records in Source 2 and Source 3 refer to the same real-world business as each Source 1 entity.

**Scope:**
- Train data: 3 TSV source files + ground truth labels (`train_ground_truth.tsv`)
- Test data: 3 TSV source files, no labels; pipeline must produce `matching_results.tsv` and `candidate_pairs.tsv`
- Evaluation metric: macro-average **F_0.5** (precision-weighted; false merges penalised 2× over missed links)
- Countries: US, India in training; US, India, **France** in test → pipeline must be country-agnostic

**High-Level Approach:**  
Classic two-stage Entity Resolution:
1. **Blocking / Candidate Generation** — dramatically reduce the O(N²) comparison space by grouping records that are likely to match. High recall is the goal here.
2. **Pairwise Matching Model** — score each (S1, S2/S3) candidate pair; threshold to binary match/no-match. High precision is the goal here (F_0.5 penalises false positives heavily).

---

## Sub-Tasks

---

### Sub-Task 1: Exploratory Data Analysis (EDA)

**Intent:** Understand the actual noise patterns, data distributions, language mix, and scale before choosing features or models.

**Expected Outcomes:**
- Know the count of records per source (train & test)
- Know the distribution of matches per S1 entity (0, 1, 2, … N)
- Know what fraction are singletons (no matches)
- Understand the dominant noise patterns (name abbreviations, address reordering, transliterations, Unicode scripts)
- Know whether `country` is a reliable filtering signal

**Todo List:**
- [ ] Load all train source files and ground truth with `pd.read_csv(sep="\t")`
- [ ] Print row counts for each source (train + test)
- [ ] Compute match cardinality distribution: how many S1 entities have 0, 1, 2, 3+ matches
- [ ] Compute singleton fraction in training ground truth
- [ ] Sample 20–30 matched pairs and manually inspect name/address noise patterns
- [ ] Check presence of non-ASCII scripts (Devanagari, Tamil, Kannada, French diacritics) in each source
- [ ] Check null/empty rates for `business_name` and `business_address` per source

**Relevant Context:**
- `student_resource/dataset/train/train_source1.tsv`
- `student_resource/dataset/train/train_source2.tsv`
- `student_resource/dataset/train/train_source3.tsv`
- `student_resource/dataset/train/train_ground_truth.tsv`
- Sample from README: Source 2 contains Hindi/Devanagari names; Source 3 contains Tamil script, transliterations

**Status:** [x] done — see EDA Findings section above

---

### Sub-Task 2: Text Normalisation & Feature Preprocessing

**Intent:** Build reusable text-cleaning and normalisation utilities that reduce surface-form variation so that downstream similarity features are meaningful.

**Expected Outcomes:**
- A `normalize_name(text)` function covering: lowercase, punctuation strip, legal-suffix expansion (Corp→corporation, Ltd→limited, Pvt→private, LLC, Inc, etc.), Unicode NFKC normalisation, extra whitespace removal
- A `normalize_address(text)` function covering: lowercase, abbreviation expansion (Rd→Road, St→Street, Ave→Avenue), remove pin/zip codes or isolate them as a separate token, NFKC normalisation
- A `tokenize(text)` helper returning a sorted frozenset of tokens (for Jaccard)
- All functions tested on representative noisy examples from EDA

**Todo List:**
- [ ] Create `code/business_entity_resolution/src/preprocess.py`
- [ ] Implement `normalize_name()` with legal-suffix dictionary (EN + common FR abbreviations for test set)
- [ ] Implement `normalize_address()` with street-type abbreviation dictionary
- [ ] Implement `tokenize()` using `normalize_name()` / `normalize_address()` output
- [ ] Write quick sanity tests on 5–10 real noisy pairs observed in EDA

**Relevant Context:**
- Noise patterns documented in README: Corp/Corporation, Pvt/Private, Ltd/Limited, Rd/Road, St/Street
- Source 2 & 3 contain ALL-CAPS addresses — lowercasing is mandatory
- French test records contain diacritics (é, à, è) — NFKC normalisation handles composed vs decomposed forms

**Status:** [x] done — `preprocess.py` with 39/39 tests passing. Critical fix: ASCII-only punct stripping preserves Devanagari/Tamil combining marks.

---

### Sub-Task 3: Blocking / Candidate Generation

**Intent:** For each S1 entity, generate a small set of candidate S2/S3 records that could plausibly match, keeping recall as high as possible while dramatically cutting the comparison count vs. the brute-force O(N²) cartesian product.

**Expected Outcomes:**
- A `build_candidates(s1_df, s2_df, s3_df) → Dict[str, List[str]]` function returning `{s1_id: [s2/s3 candidate ids]}`
- Recall on training validation split ≥ 90% (i.e., ≤ 10% of true matches are missing from candidates)
- Candidate set size per S1 entity kept manageable (target ≤ 200 candidates/entity)
- `candidate_pairs.tsv` produced from this stage

**Blocking strategies to combine (OR-union):**
1. **Country filter** — only compare records with the same `country` string (exact or lowercased match)
2. **TF-IDF + cosine (name)** — build a TF-IDF matrix of normalised business names; for each S1 record retrieve top-K (e.g. K=50) nearest S2/S3 records using approximate nearest-neighbour (sparse cosine via `sklearn` or `sparse_dot_topn`)
3. **TF-IDF + cosine (address)** — same but on normalised addresses; retrieve top-K
4. **Trigram/n-gram Jaccard blocking key** — for short name tokens, emit bigrams and group records sharing at least one bigram
5. **Prefix/first-token blocking** — group records whose first normalised name token matches

**Todo List:**
- [ ] Create `code/business_entity_resolution/src/blocking.py`
- [ ] Implement country-filtered TF-IDF name blocking (sklearn `TfidfVectorizer` + `linear_kernel` or `sparse_dot_topn`)
- [ ] Implement country-filtered TF-IDF address blocking
- [ ] Implement union of blocking strategies into a single candidate dictionary
- [ ] Evaluate recall@candidate on a 20% validation hold-out of the training ground truth
- [ ] Tune K (top-K per query) to hit ≥ 90% recall
- [ ] Write `output/candidate_pairs.tsv` from this stage

**Relevant Context:**
- `student_resource/dataset/train/train_ground_truth.tsv` — use for validation recall measurement
- `sklearn.feature_extraction.text.TfidfVectorizer` with `analyzer='char_wb'` or `analyzer='word'`
- `sparse_dot_topn` is optional but fast for large N; `sklearn` cosine on sparse matrices works at moderate scale
- Country field is a reliable first filter; all France entities only appear in test so they fall through to name/address-only blocking

**Status:** [ ] pending

---

### Sub-Task 4: Pairwise Feature Engineering

**Intent:** For every (S1, candidate) pair produced by blocking, compute a rich feature vector capturing name similarity, address similarity, and structural signals.

**Expected Outcomes:**
- A `compute_features(row_s1, row_candidate) → dict` function
- Feature matrix ready for classifier training (one row per candidate pair, labelled 1 if true match, 0 otherwise)

**Feature Groups:**

| Feature | Description |
|---|---|
| `name_jaccard` | Token Jaccard on normalised names |
| `name_levenshtein_norm` | Normalised edit distance on names |
| `name_cosine_tfidf` | TF-IDF cosine similarity (reuse blocking vectors) |
| `name_jaro_winkler` | Jaro-Winkler on raw names |
| `addr_jaccard` | Token Jaccard on normalised addresses |
| `addr_levenshtein_norm` | Normalised edit distance on addresses |
| `addr_cosine_tfidf` | TF-IDF cosine similarity on addresses |
| `country_match` | 1 if `country` strings match exactly (lowercased) |
| `name_len_ratio` | min(len_a, len_b) / max(len_a, len_b) |
| `addr_len_ratio` | same for addresses |
| `common_token_count` | Raw count of shared normalised name tokens |
| `addr_common_token_count` | Raw count of shared normalised address tokens |

**Todo List:**
- [ ] Create `code/business_entity_resolution/src/features.py`
- [ ] Implement each feature function using `rapidfuzz` (Levenshtein, Jaro-Winkler) and `sklearn` TF-IDF vectors
- [ ] Build training feature matrix from blocking candidates + ground truth labels
- [ ] Build test feature matrix from blocking candidates over test set
- [ ] Verify no NaN values; fill missing (empty name or address) with 0.0

**Relevant Context:**
- `rapidfuzz` library: `rapidfuzz.distance.Levenshtein`, `rapidfuzz.distance.JaroWinkler`
- Reuse the same `TfidfVectorizer` fitted on train data for test inference (no leakage)
- For pairs where `business_address` is empty (e.g. Source 3 row 2 in sample), addr features = 0.0

**Status:** [ ] pending

---

### Sub-Task 5: Train Matching Classifier & Threshold Optimisation

**Intent:** Train a binary classifier on labelled candidate pairs to predict match/no-match, then tune the decision threshold to maximise F_0.5 on a validation split.

**Expected Outcomes:**
- Trained model serialised to `code/business_entity_resolution/src/model/`
- Validation macro F_0.5 ≥ 0.75 (aspirational baseline)
- Decision threshold chosen to maximise macro F_0.5 (not AUC, not accuracy)
- Threshold value documented

**Model Choice:** XGBoost or LightGBM gradient-boosted classifier
- Both are MIT/Apache licensed and well under 8B parameters
- Fast to train and interpret; handle mixed numerical features well
- Alternative: `sklearn` `RandomForestClassifier` as a quick baseline

**Todo List:**
- [ ] Split training data into 80% train / 20% validation by S1 entity (group-aware split to avoid leakage)
- [ ] Create `code/business_entity_resolution/src/train.py`
- [ ] Train XGBoost binary classifier on training feature matrix; use `scale_pos_weight` to handle class imbalance (many more non-matches than matches)
- [ ] Evaluate on validation: sweep threshold from 0.1 to 0.9, compute macro F_0.5 at each threshold
- [ ] Select threshold maximising validation F_0.5
- [ ] Save model (`joblib.dump`) and threshold value to `src/model/`

**Relevant Context:**
- Class imbalance: after blocking, true-match pairs are a small fraction of all candidate pairs — use `scale_pos_weight` or `class_weight`
- Group-aware split: when splitting, hold out complete S1 entities (with all their candidate pairs), never split at the pair level (that leaks)
- Macro F_0.5 formula: compute per S1 entity, then average — must mirror the leaderboard formula exactly

**Status:** [ ] pending

---

### Sub-Task 6: Inference & Output Generation

**Intent:** Run the trained pipeline on the test set, produce the two required output files, and validate them with the provided script.

**Expected Outcomes:**
- `output/matching_results.tsv` — one row per test S1 entity, with comma-separated matched S2/S3 IDs (or empty for singletons)
- `output/candidate_pairs.tsv` — one row per test S1 entity with all blocking candidates
- `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` exits with `PASS`

**Todo List:**
- [ ] Create `code/business_entity_resolution/src/predict.py`
- [ ] Load test source files, run normalisation and blocking to produce test candidates
- [ ] Compute feature matrix for all test candidate pairs
- [ ] Run trained classifier; apply saved threshold
- [ ] Aggregate per-S1-entity: collect all candidate IDs with score ≥ threshold as matched IDs
- [ ] Ensure every test S1 entity has exactly one row (add empty rows for singletons)
- [ ] Write `output/matching_results.tsv` (tab-sep, no index)
- [ ] Write `output/candidate_pairs.tsv` (tab-sep, no index)
- [ ] Run `validate_submission.py`; fix any reported issues

**Relevant Context:**
- `student_resource/utils/validate_submission.py` — the validator to run before submitting
- Every S1 ID in `test_source1.tsv` must appear exactly once in `matching_results.tsv`
- France test entities must be included; they have no training analogues — blocking must not filter them out by country since the country filter is applied by string match (France == France), not a hard-coded allowlist

**Status:** [ ] pending

---

### Sub-Task 7: Package & Document Submission

**Intent:** Package all code, outputs, and documentation into the required zip structure.

**Expected Outcomes:**
- `<team_name>_submission.zip` with correct structure
- `Documentation_template.md` filled in
- `code/business_entity_resolution/README.md` with exact run instructions
- `code/business_entity_resolution/requirements.txt` with pinned versions

**Todo List:**
- [ ] Fill in `student_resource/Documentation_template.md` with methodology, blocking strategy, features, model, results
- [ ] Write `code/business_entity_resolution/README.md` with step-by-step run instructions
- [ ] Write `requirements.txt` pinning: pandas, scikit-learn, xgboost (or lightgbm), rapidfuzz, numpy
- [ ] Verify zip structure matches the spec: `output/`, `code/business_entity_resolution/src/`, `Documentation_template.md`
- [ ] Final re-run of `validate_submission.py` on outputs inside the zip

**Relevant Context:**
- README spec in `student_resource/README.md` — Final Submission Package section
- Model must be MIT/Apache 2.0 licensed and ≤ 8B parameters (XGBoost/LightGBM both comply)

**Status:** [ ] pending

---

## Key Architecture Diagram

```
train_source1/2/3.tsv
        │
        ▼
 [Normalisation]  ──── preprocess.py
        │
        ▼
 [Blocking]  ──────── blocking.py
  (TF-IDF name + address, country filter, n-gram keys)
        │
        ▼
 candidate_pairs  ─── candidate_pairs.tsv (output)
        │
        ▼
 [Feature Engineering] ── features.py
  (Jaccard, Levenshtein, cosine, Jaro-Winkler, ...)
        │
        ▼
 [XGBoost Classifier + Threshold] ── train.py / predict.py
        │
        ▼
 matching_results.tsv (output)  ──  Leaderboard upload
```

---

## Constraints Summary

| Constraint | Detail |
|---|---|
| No external data | No APIs, geocoding, business registries |
| Model license | MIT or Apache 2.0 |
| Model size | ≤ 8B parameters |
| Output format | Tab-separated; exact column names; every S1 row present |
| Evaluation | Macro F_0.5 (precision-heavy) |
| France | Must be handled; no training examples — blocking must stay language-agnostic |
