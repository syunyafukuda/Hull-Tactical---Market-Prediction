#!/usr/bin/env bash
# Kaggle競技データの取得と展開を自動化します。
# データはGit管理しないため、毎回このスクリプトで再現してください。
set -euo pipefail

# ディレクトリ作成
mkdir -p data/raw data/interim data/processed data/external artifacts

# 競技データのダウンロード
kaggle competitions download -c hull-tactical-market-prediction -p data/raw

# 展開
unzip -o data/raw/hull-tactical-market-prediction.zip -d data/raw

echo "[ok] downloaded to data/raw"
