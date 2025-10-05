#!/usr/bin/env bash
# Kaggle競技データの取得と展開を自動化します。
# データはGit管理しないため、毎回このスクリプトで再現してください。
# 
# This script automates the download and extraction of Kaggle competition data.
# Since data is not tracked by Git, please use this script to reproduce the data each time.
set -euo pipefail

# ディレクトリ作成
# Create directories
mkdir -p data/raw data/interim data/processed data/external artifacts

# 競技データのダウンロード
# Download competition data
kaggle competitions download -c hull-tactical-market-prediction -p data/raw

# 展開
# Extract files
unzip -o data/raw/hull-tactical-market-prediction.zip -d data/raw

echo "[ok] downloaded to data/raw"
