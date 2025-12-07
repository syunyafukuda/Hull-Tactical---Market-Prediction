# Phase 2: Model-Based Feature Selection - Usage Guide

This directory contains scripts and notebooks for Phase 2 of the feature selection process, which uses model-based importance to identify and remove low-importance features.

## Overview

Phase 2 takes the Tier1 feature set (160 features after Phase 1 statistical filtering) and further reduces it to 100-120 features using:
1. **LGBM gain importance** to identify low-importance candidates
2. **Permutation importance** to confirm which features can be safely deleted

## Scripts

### 1. compute_importance.py

Computes LGBM feature importance (gain and split) using 5-fold TimeSeriesSplit CV.

**Input**:
- Tier1 feature set (with `tier1_excluded.json` applied)
- Configuration files

**Output**:
- `results/feature_selection/tier1_importance.csv` - Fold-wise importance
- `results/feature_selection/tier1_importance_summary.csv` - Aggregated statistics

**Usage**:
```bash
python src/feature_selection/compute_importance.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --exclude-features configs/feature_selection/tier1_excluded.json \
  --out-dir results/feature_selection \
  --n-splits 5 \
  --random-state 42
```

**Options**:
- `--config-path`: Path to feature generation config
- `--preprocess-config`: Path to preprocess config
- `--data-dir`: Directory containing raw data files
- `--exclude-features`: JSON file with features to exclude (Tier1 exclusions)
- `--out-dir`: Output directory for results
- `--n-splits`: Number of CV folds (default: 5)
- `--random-state`: Random seed (default: 42)

### 2. importance_analysis.ipynb

Interactive notebook for analyzing importance results and selecting deletion candidates.

**Input**:
- `results/feature_selection/tier1_importance_summary.csv`

**Output**:
- Visualizations (histograms, scatter plots, bar plots)
- `results/feature_selection/phase2_importance_candidates.json`

**Usage**:
```bash
jupyter notebook notebooks/feature_selection/importance_analysis.ipynb
```

**Steps**:
1. Load importance summary
2. Visualize distributions
3. Apply selection criteria (bottom 25%, stable low variance)
4. Generate candidate JSON

### 3. permutation_importance.py

Computes permutation importance for candidate features by shuffling each feature and measuring ΔRMSE.

**Input**:
- Tier1 feature set
- `results/feature_selection/phase2_importance_candidates.json`
- `configs/feature_selection/tier1_excluded.json`

**Output**:
- `results/feature_selection/phase2_permutation_results.csv`

**Usage**:
```bash
python src/feature_selection/permutation_importance.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --exclude-features configs/feature_selection/tier1_excluded.json \
  --candidates results/feature_selection/phase2_importance_candidates.json \
  --out-path results/feature_selection/phase2_permutation_results.csv \
  --n-permutations 5 \
  --random-seed 42 \
  --n-splits 5
```

**Options**:
- `--candidates`: JSON file with candidate features from Phase 2-1
- `--n-permutations`: Number of permutations per feature per fold (default: 5)
- Other options same as `compute_importance.py`

## Workflow

### Step 1: Compute Importance
```bash
python src/feature_selection/compute_importance.py \
  --exclude-features configs/feature_selection/tier1_excluded.json \
  --out-dir results/feature_selection
```

Expected output:
- `tier1_importance.csv` (fold-wise)
- `tier1_importance_summary.csv` (aggregated)

### Step 2: Analyze and Extract Candidates
```bash
jupyter notebook notebooks/feature_selection/importance_analysis.ipynb
```

Run all cells to:
1. Load and visualize importance data
2. Apply selection criteria
3. Generate `phase2_importance_candidates.json`

### Step 3: Compute Permutation Importance
```bash
python src/feature_selection/permutation_importance.py \
  --exclude-features configs/feature_selection/tier1_excluded.json \
  --candidates results/feature_selection/phase2_importance_candidates.json \
  --out-path results/feature_selection/phase2_permutation_results.csv
```

Expected output:
- `phase2_permutation_results.csv` with ΔRMSE per feature

### Step 4: Create Tier2 Exclusion List

Manually review `phase2_permutation_results.csv` and create `tier2_excluded.json` by:
1. Identifying features with `|mean_delta_rmse| < threshold` (e.g., 1e-5)
2. Combining with Tier1 exclusions
3. Saving to `configs/feature_selection/tier2_excluded.json`

Example format:
```json
{
  "version": "tier2-v1",
  "created_at": "2025-12-07T00:00:00Z",
  "candidates": [
    {"feature_name": "...", "reason": "phase1_filter", ...},
    {"feature_name": "...", "reason": "phase2_permutation", "mean_delta_rmse": 0.00001}
  ],
  "summary": {
    "phase1_count": 417,
    "phase2_count": 40,
    "total_candidates": 457
  }
}
```

### Step 5: Evaluate Tier2
```bash
python src/feature_selection/evaluate_baseline.py \
  --exclude-features configs/feature_selection/tier2_excluded.json \
  --out-dir results/feature_selection \
  --artifacts-dir artifacts/tier2
```

Expected output:
- `tier2_evaluation.json`
- `tier2_fold_logs.csv`
- `tier2_importance.csv` and summary

### Step 6: Compare and Decide

Compare Tier1 vs Tier2:
- Feature count reduction
- OOF RMSE change (target: within +0.0001)
- OOF MSR change (reference)

Document results in `docs/feature_selection/phase2_report.md`.

## Selection Criteria

### Phase 2-1: LGBM Importance

**Candidate selection**:
- `mean_gain < quantile(0.25)` (bottom 25%)
- `std_gain < median(std_gain)` (stable across folds)

**Rationale**: Identifies features consistently low importance across all CV folds.

### Phase 2-2: Permutation Importance

**Deletion confirmation**:
- `|mean_delta_rmse| < 1e-5` (initial threshold)
- `std_delta_rmse < 1e-5` (stable across folds)

**Rationale**: Confirms that shuffling the feature has negligible impact on RMSE.

**Note**: Thresholds are initial proposals. Final thresholds should be determined after reviewing the actual distribution in `phase2_permutation_results.csv`.

## Output Schema

### tier1_importance_summary.csv

| Column | Type | Description |
|--------|------|-------------|
| feature_name | str | Feature name |
| mean_gain | float | Mean gain importance across folds |
| std_gain | float | Std deviation of gain importance |
| min_gain | float | Minimum gain importance |
| max_gain | float | Maximum gain importance |
| mean_split | float | Mean split importance across folds |
| std_split | float | Std deviation of split importance |
| min_split | float | Minimum split importance |
| max_split | float | Maximum split importance |
| mean_gain_normalized | float | Share of total gain (0-1) |

### phase2_importance_candidates.json

```json
{
  "version": "phase2-v1",
  "created_at": "ISO-8601 timestamp",
  "source_tier": "tier1",
  "selection_criteria": {
    "method": "lgbm_importance",
    "metric": "gain",
    "threshold_quantile": 0.25,
    "threshold_mean_gain": float,
    "threshold_std_gain": float,
    "require_stable_low": true
  },
  "candidates": [
    {
      "feature_name": "...",
      "mean_gain": float,
      "std_gain": float,
      "share_of_total": float,
      "note": "..."
    }
  ],
  "summary": {
    "total_features": int,
    "candidate_count": int,
    "candidate_ratio": float
  }
}
```

### phase2_permutation_results.csv

| Column | Type | Description |
|--------|------|-------------|
| feature_name | str | Feature name |
| mean_delta_rmse | float | Mean ΔRMSE across folds |
| std_delta_rmse | float | Std deviation of ΔRMSE |
| fold_1_delta | float | ΔRMSE for fold 1 |
| fold_2_delta | float | ΔRMSE for fold 2 |
| ... | ... | ... |
| fold_5_delta | float | ΔRMSE for fold 5 |
| decision | str | "remove" or "keep" (initial, subject to review) |

## Computational Considerations

### Runtime Estimates

- `compute_importance.py`: ~10-30 minutes (depends on data size, 5 folds × training)
- `permutation_importance.py`: ~30-60 minutes (40 candidates × 5 folds × 5 permutations)
- `evaluate_baseline.py` (Tier2): ~10-30 minutes (5 folds × training)

### Memory Requirements

- Peak memory: ~4-8 GB (depending on dataset size)
- Augmented feature matrix size: ~160 columns × ~1000-2000 rows

### Optimization Tips

1. **Reduce candidate count**: Only test bottom 20-30% features in permutation test
2. **Parallel processing**: Can be added to permutation loop if needed
3. **Reduce permutations**: Start with n=3-5, increase if results are unstable

## Troubleshooting

### Issue: compute_importance.py fails with "Model does not have feature_importances_"

**Solution**: Ensure LightGBM is properly installed and model training succeeded.

### Issue: permutation_importance.py is too slow

**Solutions**:
1. Reduce `--n-permutations` (try 3 instead of 5)
2. Reduce candidate count (tighten Phase 2-1 criteria)
3. Use a smaller dataset for testing

### Issue: Feature names are generic (feature_0, feature_1, ...)

**Solution**: This means the preprocessing step doesn't expose feature names. Check that your preprocessing pipeline implements `get_feature_names_out()`.

## Testing

Run unit tests:
```bash
pytest tests/feature_selection/test_phase2.py -v
```

Tests cover:
- `compute_fold_importance()` function
- `aggregate_importance()` function
- JSON schema validation
- Edge cases (empty data, mismatched lengths, etc.)

## References

- Specification: `docs/feature_selection/phase2_spec.md`
- Phase 1 Report: `docs/feature_selection/phase1_report.md`
- Overall Plan: `docs/feature_selection/README.md`
