# SU3 Hyperparameter Sweep

This directory contains the implementation of SU3 hyperparameter sweep functionality for optimizing feature selection parameters.

## Overview

The sweep evaluates different combinations of top-k values for SU3 feature generation:
- `reappear_top_k`: Number of top reappearance patterns to keep (default: 20)
- `temporal_top_k`: Number of top temporal bias features to keep (default: 20)
- `holiday_top_k`: Number of top holiday interaction features to keep (default: 20)
- `include_imputation_trace`: Whether to include imputation trace features (Stage 2)

## Files

- `sweep_oof.py`: Main sweep implementation (553 lines)
- `../../../tests/feature_generation/test_su3_sweep.py`: Unit tests (327 lines, 4 test cases)
- `../../../scripts/run_su3_sweep.sh`: Helper script for running sweeps

## Usage

### Basic Usage

```bash
# Stage 1: Without imputation trace (48 configurations)
python3 src/feature_generation/su3/sweep_oof.py \
    --data-dir data/raw \
    --config-path configs/feature_generation.yaml \
    --preprocess-config configs/preprocess.yaml \
    --output-dir results/ablation/SU3 \
    --n-splits 5 \
    --n-estimators 600 \
    --verbosity -1
```

### Using the Helper Script

```bash
# With default parameters
./scripts/run_su3_sweep.sh

# With custom parameters
DATA_DIR=data/raw N_SPLITS=5 N_ESTIMATORS=600 ./scripts/run_su3_sweep.sh
```

### Stage 2: With Imputation Trace

```bash
python3 src/feature_generation/su3/sweep_oof.py \
    --include-imputation-trace \
    --data-dir data/raw \
    --output-dir results/ablation/SU3 \
    --n-splits 5
```

## Parameters

### Required
- `--data-dir`: Directory containing training data (default: `data/raw`)
- `--config-path`: Path to feature generation config (default: `configs/feature_generation.yaml`)
- `--preprocess-config`: Path to preprocessing config (default: `configs/preprocess.yaml`)

### Optional
- `--train-file`: Explicit path to training file (auto-detected if not provided)
- `--output-dir`: Output directory for results (default: `results/ablation/SU3`)
- `--n-splits`: Number of TimeSeriesSplit folds (default: 5)
- `--gap`: Gap between train and validation in each fold (default: 0)
- `--min-val-size`: Minimum validation size (default: 0)
- `--n-estimators`: LightGBM estimators (default: 600)
- `--learning-rate`: LightGBM learning rate (default: 0.05)
- `--reg-alpha`: L1 regularization (default: 0.1)
- `--reg-lambda`: L2 regularization (default: 0.1)
- `--random-state`: Random seed (default: 42)
- `--include-imputation-trace`: Enable Stage 2 features

## Output

### JSON (Detailed Results)
`results/ablation/SU3/sweep_YYYY-MM-DD_HHMMSS.json`:
```json
{
  "metadata": {
    "timestamp": "2025-11-22_153045",
    "n_configs": 48,
    "n_splits": 5,
    "gap": 0,
    "model_params": {...}
  },
  "results": [
    {
      "config": {
        "reappear_top_k": 20,
        "temporal_top_k": 20,
        "holiday_top_k": 20,
        "include_imputation_trace": false
      },
      "oof_rmse": 0.012104,
      "oof_msr": 0.018567,
      "n_features": 474,
      "train_time_sec": 45.2,
      "fold_scores": [...]
    }
  ]
}
```

### CSV (Summary)
`results/ablation/SU3/sweep_summary.csv`:
```csv
timestamp,config_id,reappear_top_k,temporal_top_k,holiday_top_k,include_imputation_trace,oof_rmse,oof_msr,n_features,train_time_sec
2025-11-22_153045,1,20,20,20,False,0.012104,0.018567,474,45.2
...
```

## Parameter Grid

### Stage 1 (Default - 48 combinations)
```python
PARAM_GRID = {
    'reappear_top_k': [10, 20, 30, 50],      # 4 values
    'temporal_top_k': [10, 20, 30],          # 3 values
    'holiday_top_k': [10, 20, 30, 50],       # 4 values
}
# Total: 4 × 3 × 4 = 48 configurations
```

### Stage 2 (Future)
Additional parameter: `include_imputation_trace: [False, True]`

## Estimated Runtime

- **Small test data** (200 rows, 2 configs): ~0.4 seconds
- **Full sweep** (48 configs, 5 folds, 600 estimators): ~2-3 hours

## Baseline Comparison

- **SU1 baseline**: OOF RMSE=0.01212, OOF MSR=0.01821, Features=368
- **SU3 target**: OOF MSR ≤ 0.01921 (SU1 + 0.001)

## Test Coverage

4 unit tests in `test_su3_sweep.py`:
1. `test_build_param_combinations`: Parameter grid generation
2. `test_evaluate_single_config_small_data`: End-to-end OOF evaluation
3. `test_save_results`: Result file format validation
4. `test_build_param_combinations_with_imputation`: Stage 2 parameter generation

All tests pass with Ruff and Pyright validation.

## Implementation Notes

- Uses TimeSeriesSplit for cross-validation
- Supports custom gap between train/validation
- Results sorted by OOF MSR (ascending - lower is better)
- Automatically saves to both JSON (detailed) and CSV (summary)
- CSV uses append mode for incremental runs
- Full pipeline includes: SU1 generation → SU3 generation → Preprocessing → LightGBM

## See Also

- [SU3 Design Document](../../../docs/feature_generation/SU3.md)
- [SU2 Sweep Implementation](../../su2/sweep_oof.py) (reference pattern)
- [Train SU3 Script](train_su3.py)
