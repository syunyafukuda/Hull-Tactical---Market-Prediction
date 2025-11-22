#!/bin/bash
# src/feature_generation/su3/run_su3_sweep.sh
# SU3 hyperparameter sweep execution script

set -e

echo "=== SU3 Hyperparameter Sweep ==="
echo "Stage 1: Grid search without imputation trace features"
echo ""

# Default parameters
DATA_DIR="${DATA_DIR:-data/raw}"
CONFIG_PATH="${CONFIG_PATH:-configs/feature_generation.yaml}"
PREPROCESS_CONFIG="${PREPROCESS_CONFIG:-configs/preprocess.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ablation/SU3}"
N_SPLITS="${N_SPLITS:-5}"
GAP="${GAP:-0}"
N_ESTIMATORS="${N_ESTIMATORS:-600}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
REG_ALPHA="${REG_ALPHA:-0.1}"
REG_LAMBDA="${REG_LAMBDA:-0.1}"
RANDOM_STATE="${RANDOM_STATE:-42}"

echo "Parameters:"
echo "  Data directory: $DATA_DIR"
echo "  Config path: $CONFIG_PATH"
echo "  Preprocess config: $PREPROCESS_CONFIG"
echo "  Output directory: $OUTPUT_DIR"
echo "  N splits: $N_SPLITS"
echo "  Gap: $GAP"
echo "  N estimators: $N_ESTIMATORS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Reg alpha: $REG_ALPHA"
echo "  Reg lambda: $REG_LAMBDA"
echo "  Random state: $RANDOM_STATE"
echo ""

# Run the sweep
python3 src/feature_generation/su3/sweep_oof.py \
    --data-dir "$DATA_DIR" \
    --config-path "$CONFIG_PATH" \
    --preprocess-config "$PREPROCESS_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --n-splits "$N_SPLITS" \
    --gap "$GAP" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate "$LEARNING_RATE" \
    --reg-alpha "$REG_ALPHA" \
    --reg-lambda "$REG_LAMBDA" \
    --random-state "$RANDOM_STATE" \
    --verbosity -1

echo ""
echo "=== Sweep Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  JSON (detailed): $OUTPUT_DIR/sweep_YYYY-MM-DD_HHMMSS.json"
echo "  CSV (summary): $OUTPUT_DIR/sweep_summary.csv"
echo ""
echo "For Stage 2 (with imputation trace), add: --include-imputation-trace"
