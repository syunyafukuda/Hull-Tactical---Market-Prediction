# Phase 3 Quick Start Guide

This guide provides a quick walkthrough to run Phase 3 of the feature selection process.

## Prerequisites

Before running Phase 3, ensure you have:

1. **Completed Phase 2** with the following files:
   - `configs/feature_selection/tier2/excluded.json` - Tier2 exclusion list
   - `results/feature_selection/tier2/importance_summary.csv` - Feature importance from Tier2
   - `results/feature_selection/tier2/evaluation.json` - Tier2 OOF evaluation results

2. **Data files** in `data/raw/` directory

3. **Configuration files**:
   - `configs/feature_generation.yaml`
   - `configs/preprocess.yaml`

## Quick Start: Run Complete Pipeline

The simplest way to run Phase 3 is using the orchestration script:

```bash
python src/feature_selection/phase3/run_phase3.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --tier2-excluded configs/feature_selection/tier2/excluded.json \
  --tier2-importance results/feature_selection/tier2/importance_summary.csv \
  --tier2-evaluation results/feature_selection/tier2/evaluation.json \
  --correlation-threshold 0.95 \
  --topk 50
```

This will:
1. ✅ Perform correlation clustering on Tier2 features (120 columns)
2. ✅ Select cluster representatives based on importance
3. ✅ Create Tier3 excluded features list
4. ✅ Evaluate Tier3 with OOF cross-validation
5. ✅ Generate feature set configurations (FS_full, FS_compact, FS_topK)
6. ✅ Create Phase 3 execution report

## Expected Outputs

After successful completion, you'll have:

```
configs/feature_selection/
├── tier3/
│   └── excluded.json              # Tier3 exclusion list
├── tier_topK/
│   └── excluded.json              # Top-K exclusion list
└── feature_sets.json              # Feature set definitions

results/feature_selection/
├── tier3/
│   ├── evaluation.json            # Tier3 OOF evaluation
│   ├── fold_logs.csv              # Per-fold evaluation logs
│   └── importance_summary.csv     # Tier3 feature importance
└── phase3/
    ├── correlation_clusters.json  # Clustering results
    └── cluster_representatives.json # Representative selection

docs/feature_selection/
└── phase3_report.md               # Execution report
```

## Review Results

1. **Check the execution report**:
   ```bash
   cat docs/feature_selection/phase3_report.md
   ```

2. **Review feature sets**:
   ```bash
   cat configs/feature_selection/feature_sets.json
   ```

3. **Compare RMSE scores**:
   - Tier2 RMSE: See `results/feature_selection/tier2/evaluation.json`
   - Tier3 RMSE: See `results/feature_selection/tier3/evaluation.json`
   - Tier3 is adopted if `delta <= 0.0001`

## Alternative: Skip Correlation Clustering

If you want to use Tier2 as the final feature set without clustering:

```bash
python src/feature_selection/phase3/run_phase3.py \
  --skip-clustering \
  --tier2-excluded configs/feature_selection/tier2/excluded.json \
  --tier2-importance results/feature_selection/tier2/importance_summary.csv \
  --tier2-evaluation results/feature_selection/tier2/evaluation.json
```

This will only create feature set configurations without performing clustering.

## Next Steps

After Phase 3 completion:

1. **Review the recommended feature set** in `feature_sets.json`
2. **Proceed to model selection phase** using the recommended set
3. **Evaluate multiple models** (XGBoost, CatBoost, Ridge, etc.)
4. **Perform hyperparameter optimization**
5. **Consider ensemble strategies**

## Troubleshooting

### Issue: Missing Tier2 files

**Solution**: Complete Phase 2 first to generate required input files.

### Issue: Correlation clustering finds no clusters

**Solution**: This is normal if Tier2 features are already diverse. Use `--skip-clustering` to proceed with Tier2 as final set.

### Issue: Tier3 RMSE is worse than Tier2

**Solution**: The pipeline will automatically maintain Tier2 as the recommended set. Review `phase3_report.md` for details.

## Advanced Options

See `src/feature_selection/phase3/README.md` for:
- Individual script usage
- Custom thresholds
- Manual step-by-step execution
- Output schema details

## Support

For detailed documentation:
- Phase 3 specification: `docs/feature_selection/phase3_spec.md`
- Implementation details: `src/feature_selection/phase3/README.md`
- Overall feature selection: `docs/feature_selection/README.md`
