# Feature Selection Scripts

## Overview

This directory contains scripts for evaluating and filtering features through multiple phases:

- **Phase 0**: Tier0 Baseline Evaluation (SU1 + SU5 + Brushup)
- **Phase 1**: Filter-based Feature Selection (statistical filtering)

## Files

### Scripts

- `evaluate_baseline.py`: Main evaluation script for Tier0/Tier1 baselines
- `filter_trivial_phase1.py`: Statistical feature filtering script

### Phase 0 Results (in `results/feature_selection/`)

- `tier0_evaluation.json`: OOF metrics summary
- `tier0_importance.csv`: Fold-wise feature importance
- `tier0_importance_summary.csv`: Aggregated importance statistics
- `tier0_fold_logs.csv`: Fold-wise RMSE/MSR logs

### Phase 1 Results (in `results/feature_selection/`)

- `phase1_filter_candidates.json`: Features marked for removal
- `tier1_evaluation.json`: OOF metrics after filtering
- `tier1_importance_summary.csv`: Aggregated importance after filtering
- `tier1_fold_logs.csv`: Fold-wise logs after filtering

---

## Phase 0 - Tier0 Baseline Evaluation

### Usage

#### Basic Execution

```bash
python src/feature_selection/evaluate_baseline.py \
  --config-path configs/tier0_snapshot/feature_generation.yaml \
  --preprocess-config configs/tier0_snapshot/preprocess.yaml \
  --data-dir data/raw \
  --out-dir results/feature_selection
```

#### With Custom Parameters

```bash
python src/feature_selection/evaluate_baseline.py \
  --config-path configs/tier0_snapshot/feature_generation.yaml \
  --preprocess-config configs/tier0_snapshot/preprocess.yaml \
  --data-dir data/raw \
  --out-dir results/feature_selection \
  --n-splits 5 \
  --random-state 42 \
  --verbosity -1
```

### Expected Output

When run with the Tier0 configuration:

- **OOF RMSE**: ~0.012134 (±0.0001)
- **Feature Count**: 577 features
- **Evaluation Files**: 4 output files (JSON and CSV)

---

## Phase 1 - Filter-based Feature Selection

### Step 1: Identify Candidate Features

Use `filter_trivial.py` to identify features that should be excluded based on statistical criteria:

```bash
python src/feature_selection/filter_trivial.py \
  --config-path configs/tier0_snapshot/feature_generation.yaml \
  --preprocess-config configs/tier0_snapshot/preprocess.yaml \
  --data-dir data/raw \
  --out-path results/feature_selection/phase1_filter_candidates.json \
  --importance-path results/feature_selection/tier0_importance_summary.csv \
  --variance-threshold 1e-10 \
  --missing-threshold 0.99 \
  --correlation-threshold 0.999
```

#### Filtering Criteria

| Criterion | Threshold | Description |
|-----------|-----------|-------------|
| Low Variance | `< 1e-10` | Near-constant features |
| High Missing | `> 0.99` | Near-empty features (>99% missing) |
| High Correlation | `> 0.999` | Redundant features (drops lower importance) |

#### Output Format

```json
{
  "version": "phase1-v1",
  "created_at": "2025-12-06T00:00:00Z",
  "thresholds": {
    "variance_min": 1e-10,
    "missing_rate_max": 0.99,
    "correlation_max": 0.999
  },
  "candidates": [
    {
      "feature_name": "M1",
      "reason": "low_variance",
      "value": 1.2e-11
    }
  ],
  "summary": {
    "total_features": 577,
    "low_variance_count": 5,
    "high_missing_count": 2,
    "high_correlation_count": 10,
    "total_candidates": 15
  }
}
```

### Step 2: Evaluate Filtered Features

Use `evaluate_baseline.py` with the `--exclude-features` flag to evaluate performance after filtering:

```bash
python src/feature_selection/evaluate_baseline.py \
  --config-path configs/tier0_snapshot/feature_generation.yaml \
  --preprocess-config configs/tier0_snapshot/preprocess.yaml \
  --data-dir data/raw \
  --out-dir results/feature_selection \
  --exclude-features results/feature_selection/phase1_filter_candidates.json
```

This will:
1. Load the filter candidates from JSON
2. Exclude those features after SU5 augmentation
3. Train and evaluate with remaining features
4. Save results with `tier1_` prefix

### Step 3: Compare Results

Compare Tier0 (baseline) vs Tier1 (filtered):

| Metric | Tier0 | Tier1 | Diff | Interpretation |
|--------|-------|-------|------|----------------|
| Features | 577 | ? | ? | Features removed |
| OOF RMSE | 0.012134 | ? | ? | Should be ~equal |
| OOF MSR | 0.019929 | ? | ? | Should be maintained |

**Acceptance Criteria**:
- RMSE within ±0.0001: Adopt filtering
- RMSE +0.0001 to +0.0002 with significant feature reduction: Consider adoption
- RMSE > +0.0002: Reject filtering or adjust thresholds

---

## Output Format Reference

### tier0_evaluation.json / tier1_evaluation.json

```json
{
  "oof_rmse": 0.012134,
  "oof_coverage": 0.8331,
  "oof_msr": 0.019929,
  "oof_msr_down": 0.030318,
  "oof_best_mult": 1.5,
  "oof_best_lo": 0.8,
  "oof_best_hi": 1.1,
  "fold_count": 5
}
```

### tier0_importance.csv / tier1_importance.csv

```csv
feature_name,importance_gain,importance_split,fold
M1,0.0123,45,1
M2,0.0089,32,1
...
```

### tier0_importance_summary.csv / tier1_importance_summary.csv

```csv
feature_name,mean_gain,std_gain,min_gain,max_gain,mean_split,std_split,min_split,max_split
M1,0.0120,0.0015,0.0098,0.0145,42.0,5.2,35,50
...
```

### tier0_fold_logs.csv / tier1_fold_logs.csv

```csv
fold,train_size,val_size,rmse,msr,msr_down,best_mult,best_lo,best_hi
1,2000,500,0.012150,0.0245,0.0358,1.5,0.8,1.1
...
```

---

## Notes

- Both scripts reuse logic from `src/feature_generation/su5/train_su5.py`
- Feature importance is extracted from LightGBM using both `gain` and `split` metrics
- MSR (Mean-Sharpe-Ratio) is calculated as a secondary metric for utility assessment
- All results are saved in the `results/feature_selection/` directory
- Feature filtering is applied **after** SU5 augmentation but **before** preprocessing

## Requirements

- Python 3.12+
- Dependencies: numpy, pandas, scikit-learn, lightgbm, pyyaml
- Input data: `data/raw/train.csv` or `data/raw/train.parquet`
- Input data: `data/raw/test.csv` or `data/raw/test.parquet` (optional for filter_trivial.py)

## Related Documentation

- Phase 0 Spec: `docs/feature_selection/phase0_spec.md`
- Phase 1 Spec: `docs/feature_selection/phase1_spec.md`
- Phase 1 Report: `docs/feature_selection/phase1_report.md`
- Tier0 Artifacts: `artifacts/tier0/`
