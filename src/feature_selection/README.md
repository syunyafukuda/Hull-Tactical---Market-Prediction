# Phase 0 - Tier0 Baseline Evaluation

## Overview

This directory contains the evaluation script and results for the Tier0 baseline (SU1 + SU5 + Brushup).

## Files

- `evaluate_baseline.py`: Main evaluation script
- `tier0_evaluation.json`: OOF metrics summary
- `tier0_importance.csv`: Fold-wise feature importance
- `tier0_importance_summary.csv`: Aggregated importance statistics
- `tier0_fold_logs.csv`: Fold-wise RMSE/MSR logs

## Usage

### Basic Execution

```bash
python src/feature_selection/evaluate_baseline.py \
  --config-path configs/tier0_snapshot/feature_generation.yaml \
  --preprocess-config configs/tier0_snapshot/preprocess.yaml \
  --data-dir data/raw \
  --out-dir results/feature_selection
```

### With Custom Parameters

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

## Expected Output

When run with the Tier0 configuration:

- **OOF RMSE**: ~0.012134 (±0.0001)
- **Feature Count**: 577 features
- **Evaluation Files**: 4 output files (JSON and CSV)

## Output Format

### tier0_evaluation.json

```json
{
  "oof_rmse": 0.012134,
  "oof_coverage": 1.0,
  "oof_msr": 0.024,
  "oof_msr_down": 0.035,
  "oof_best_mult": 1.0,
  "oof_best_lo": 0.9,
  "oof_best_hi": 1.1,
  "fold_count": 5
}
```

### tier0_importance.csv

```csv
feature_name,importance_gain,importance_split,fold
M1,0.0123,45,1
M2,0.0089,32,1
...
```

### tier0_importance_summary.csv

```csv
feature_name,mean_gain,std_gain,min_gain,max_gain,mean_split,std_split,min_split,max_split
M1,0.0120,0.0015,0.0098,0.0145,42.0,5.2,35,50
...
```

### tier0_fold_logs.csv

```csv
fold,train_size,val_size,rmse,msr,msr_down,best_mult,best_lo,best_hi
1,2000,500,0.012150,0.0245,0.0358,1.0,0.9,1.1
...
```

## Notes

- The script reuses logic from `src/feature_generation/su5/train_su5.py`
- Feature importance is extracted from LightGBM using both `gain` and `split` metrics
- MSR (Mean-Sharpe-Ratio) is calculated as a secondary metric for utility assessment
- All results are saved in the `results/feature_selection/` directory

## Requirements

- Python 3.12+
- Dependencies: numpy, pandas, scikit-learn, lightgbm, pyyaml
- Input data: `data/raw/train.csv` and `data/raw/test.csv`
