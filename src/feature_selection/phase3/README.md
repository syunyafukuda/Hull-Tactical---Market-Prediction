# Phase 3: Correlation Clustering and Final Feature Set Definition

This directory contains scripts for Phase 3 of the feature selection process.

## Overview

Phase 3 performs correlation-based clustering on Tier2 features (120 columns) to identify and remove redundant features, then defines multiple feature set variants for the model selection phase.

## Scripts

### Core Pipeline Scripts

1. **correlation_clustering.py** - Phase 3-1: Correlation Clustering
   - Computes correlation matrix for Tier2 features
   - Performs hierarchical clustering using Ward's method
   - Identifies groups of highly correlated features (|ρ| > 0.95)
   - Output: `results/feature_selection/phase3/correlation_clusters.json`

2. **select_representatives.py** - Phase 3-2: Cluster Representative Selection
   - Selects one representative feature from each cluster
   - Uses LGBM feature importance (mean_gain) as selection criterion
   - Generates deletion candidate list
   - Output: `results/feature_selection/phase3/cluster_representatives.json`

3. **create_tier3_excluded.py** - Phase 3-3a: Tier3 Exclusion List Creation
   - Merges Tier2 exclusions with Phase 3 deletions
   - Creates Tier3 excluded features configuration
   - Output: `configs/feature_selection/tier3/excluded.json`

4. **create_feature_sets.py** - Phase 3-4: Feature Sets Configuration
   - Defines multiple feature set variants (FS_full, FS_compact, FS_topK)
   - Creates configuration for model selection phase
   - Output: `configs/feature_selection/feature_sets.json`

### Orchestration

5. **run_phase3.py** - Complete Phase 3 Pipeline
   - Runs all Phase 3 steps in sequence
   - Evaluates Tier3 using existing evaluate_baseline.py
   - Generates Phase 3 report
   - Output: `docs/feature_selection/phase3_report.md`

## Usage

### Run Complete Pipeline

```bash
# Full pipeline with correlation clustering
python src/feature_selection/phase3/run_phase3.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --tier2-excluded configs/feature_selection/tier2/excluded.json \
  --tier2-importance results/feature_selection/tier2/importance_summary.csv \
  --tier2-evaluation results/feature_selection/tier2/evaluation.json \
  --correlation-threshold 0.95 \
  --topk 50

# Skip clustering (use Tier2 as final)
python src/feature_selection/phase3/run_phase3.py \
  --skip-clustering \
  --tier2-excluded configs/feature_selection/tier2/excluded.json \
  --tier2-importance results/feature_selection/tier2/importance_summary.csv

# Skip evaluation (for testing)
python src/feature_selection/phase3/run_phase3.py \
  --skip-evaluation
```

### Run Individual Steps

```bash
# Step 1: Correlation clustering
python src/feature_selection/phase3/correlation_clustering.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --exclude-features configs/feature_selection/tier2/excluded.json \
  --correlation-threshold 0.95 \
  --out-dir results/feature_selection/phase3

# Step 2: Select representatives
python src/feature_selection/phase3/select_representatives.py \
  --clusters-json results/feature_selection/phase3/correlation_clusters.json \
  --importance-csv results/feature_selection/tier2/importance_summary.csv \
  --out-dir results/feature_selection/phase3

# Step 3a: Create Tier3 excluded list
python src/feature_selection/phase3/create_tier3_excluded.py \
  --tier2-excluded configs/feature_selection/tier2/excluded.json \
  --representatives-json results/feature_selection/phase3/cluster_representatives.json \
  --out-file configs/feature_selection/tier3/excluded.json

# Step 3b: Evaluate Tier3 (uses existing evaluate_baseline.py)
python src/feature_selection/common/evaluate_baseline.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --exclude-features configs/feature_selection/tier3/excluded.json \
  --out-dir results/feature_selection/tier3

# Step 4: Create feature sets configuration
python src/feature_selection/phase3/create_feature_sets.py \
  --tier2-excluded configs/feature_selection/tier2/excluded.json \
  --tier3-excluded configs/feature_selection/tier3/excluded.json \
  --tier2-evaluation results/feature_selection/tier2/evaluation.json \
  --tier3-evaluation results/feature_selection/tier3/evaluation.json \
  --tier2-importance results/feature_selection/tier2/importance_summary.csv \
  --topk 50 \
  --recommended FS_compact \
  --out-file configs/feature_selection/feature_sets.json
```

## Outputs

### Directory Structure After Completion

```
configs/feature_selection/
├── tier2/
│   └── excluded.json          # Input: Tier2 exclusions
├── tier3/
│   └── excluded.json          # Output: Tier3 exclusions (Tier2 + Phase3)
├── tier_topK/
│   └── excluded.json          # Output: Top-K exclusions
└── feature_sets.json          # Output: Feature set definitions

results/feature_selection/
├── tier2/
│   ├── evaluation.json        # Input: Tier2 evaluation
│   └── importance_summary.csv # Input: Tier2 importance
├── tier3/
│   ├── evaluation.json        # Output: Tier3 evaluation
│   ├── fold_logs.csv          # Output: Tier3 fold logs
│   └── importance_summary.csv # Output: Tier3 importance
└── phase3/
    ├── correlation_clusters.json      # Output: Cluster assignments
    └── cluster_representatives.json   # Output: Representatives & deletions

docs/feature_selection/
└── phase3_report.md           # Output: Phase 3 execution report
```

## Feature Sets

Phase 3 defines three feature set variants:

1. **FS_full** (Tier2)
   - All Tier2 features (~120 features)
   - Maximum performance baseline
   - Use for: Best possible accuracy

2. **FS_compact** (Tier3)
   - Tier2 minus correlated features (~80-100 features)
   - Balanced performance and efficiency
   - Use for: Production deployment

3. **FS_topK** (Top K by importance)
   - Top K features by mean_gain (~50 features by default)
   - Minimal feature set
   - Use for: Fast inference, debugging

## Decision Criteria

**Tier3 Adoption Criteria:**
- If `Tier3_RMSE - Tier2_RMSE <= 0.0001`: ✅ Adopt Tier3
- If `Tier3_RMSE - Tier2_RMSE > 0.0001`: ❌ Maintain Tier2

## Next Steps

After Phase 3 completion:
1. Review `docs/feature_selection/phase3_report.md`
2. Check feature sets in `configs/feature_selection/feature_sets.json`
3. Proceed to model selection phase
4. Evaluate multiple models (XGBoost, CatBoost, Ridge, etc.) with recommended feature set
5. Perform hyperparameter optimization
6. Consider ensemble strategies
