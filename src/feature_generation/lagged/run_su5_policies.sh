#!/bin/bash
# SU5 ポリシー1とポリシー2の学習・推論を実行

set -e

# Dynamically determine the project root based on the script's location
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"
cd "$PROJECT_ROOT"

echo "==================================="
echo "SU5 ポリシー実行スクリプト"
echo "==================================="
echo ""

# ポリシー1: top_k=10, windows=[5]
echo "========================================="
echo "ポリシー1: top_k=10, windows=[5]"
echo "========================================="
echo ""

# configを一時的にポリシー1に設定
cat > /tmp/su5_policy1.yaml << 'EOF'
su5:
  enabled: true
  id_column: date_id
  output_prefix: su5
  top_k_pairs: 10
  top_k_pairs_per_group: null
  windows: [5]
  reset_each_fold: true
  dtype:
    flag: uint8
    int: int16
    float: float32
EOF

echo "[Policy 1] Training..."
python src/feature_generation/su5/train_su5.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --out-dir artifacts/SU5/policy1_top10_w5 \
  --n-splits 5 \
  --gap 0 \
  --random-state 42 \
  --verbosity -1

echo ""
echo "[Policy 1] Inference..."
python src/feature_generation/su5/predict_su5.py \
  --data-dir data/raw \
  --artifacts-dir artifacts/SU5/policy1_top10_w5 \
  --out-parquet artifacts/SU5/policy1_top10_w5/submission.parquet \
  --out-csv artifacts/SU5/policy1_top10_w5/submission.csv

echo ""
echo "========================================="
echo "ポリシー2: top_k=5, windows=[5,20]"
echo "========================================="
echo ""

# configをポリシー2に一時変更（YAMLの書き換え）
python3 << 'PYTHON_EOF'
import yaml
from pathlib import Path

config_path = Path("configs/feature_generation.yaml")
with config_path.open("r") as f:
    config = yaml.safe_load(f)

# ポリシー2に切り替え
config["su5"]["top_k_pairs"] = 5
config["su5"]["windows"] = [5, 20]

# 一時保存
with Path("/tmp/feature_generation_policy2.yaml").open("w") as f:
    yaml.dump(config, f)
PYTHON_EOF

echo "[Policy 2] Training..."
python src/feature_generation/su5/train_su5.py \
  --config-path /tmp/feature_generation_policy2.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --out-dir artifacts/SU5/policy2_top5_w5-20 \
  --n-splits 5 \
  --gap 0 \
  --random-state 42 \
  --verbosity -1

echo ""
echo "[Policy 2] Inference..."
python src/feature_generation/su5/predict_su5.py \
  --data-dir data/raw \
  --artifacts-dir artifacts/SU5/policy2_top5_w5-20 \
  --out-parquet artifacts/SU5/policy2_top5_w5-20/submission.parquet \
  --out-csv artifacts/SU5/policy2_top5_w5-20/submission.csv

echo ""
echo "==================================="
echo "完了！"
echo "==================================="
echo ""
echo "生成物:"
echo "  - artifacts/SU5/policy1_top10_w5/"
echo "    - inference_bundle.pkl"
echo "    - model_meta.json"
echo "    - feature_list.json"
echo "    - submission.csv/parquet"
echo ""
echo "  - artifacts/SU5/policy2_top5_w5-20/"
echo "    - inference_bundle.pkl"
echo "    - model_meta.json"
echo "    - feature_list.json"
echo "    - submission.csv/parquet"
echo ""
echo "次のステップ:"
echo "  1. submission.csvをKaggleにSubmit"
echo "  2. LBスコアを確認"
echo "  3. SU1比較（LB 0.674がベースライン）"
