# SU3 ハイパーパラメータスイープ

SU3特徴生成の特徴選択パラメータを最適化するハイパーパラメータスイープ機能の実装です。

## 概要

スイープは、SU3特徴生成のさまざまなtop-k値の組み合わせを評価します：
- `reappear_top_k`: 保持する再出現パターンの上位数（デフォルト: 20）
- `temporal_top_k`: 保持する時系列バイアス特徴量の上位数（デフォルト: 20）
- `holiday_top_k`: 保持する祝日相互作用特徴量の上位数（デフォルト: 20）
- `include_imputation_trace`: 代入影響特徴量を含めるかどうか（Stage 2）

## ファイル

- `sweep_oof.py`: メインスイープ実装（553行）
- `../../../tests/feature_generation/test_su3_sweep.py`: ユニットテスト（327行、4テストケース）
- `../../../scripts/run_su3_sweep.sh`: スイープ実行用ヘルパースクリプト

## 使用方法

### 基本的な使用方法

```bash
# Stage 1: 代入影響トレースなし（48構成）
python3 src/feature_generation/su3/sweep_oof.py \
    --data-dir data/raw \
    --config-path configs/feature_generation.yaml \
    --preprocess-config configs/preprocess.yaml \
    --output-dir results/ablation/SU3 \
    --n-splits 5 \
    --n-estimators 600 \
    --verbosity -1
```

### ヘルパースクリプトの使用

```bash
# デフォルトパラメータで実行
./scripts/run_su3_sweep.sh

# カスタムパラメータで実行
DATA_DIR=data/raw N_SPLITS=5 N_ESTIMATORS=600 ./scripts/run_su3_sweep.sh
```

### Stage 2: 代入影響トレース付き

```bash
python3 src/feature_generation/su3/sweep_oof.py \
    --include-imputation-trace \
    --data-dir data/raw \
    --output-dir results/ablation/SU3 \
    --n-splits 5
```

## パラメータ

### 必須
- `--data-dir`: 学習データを含むディレクトリ（デフォルト: `data/raw`）
- `--config-path`: 特徴生成設定へのパス（デフォルト: `configs/feature_generation.yaml`）
- `--preprocess-config`: 前処理設定へのパス（デフォルト: `configs/preprocess.yaml`）

### オプション
- `--train-file`: 学習ファイルへの明示的なパス（指定しない場合は自動検出）
- `--output-dir`: 結果の出力ディレクトリ（デフォルト: `results/ablation/SU3`）
- `--n-splits`: TimeSeriesSplit の分割数（デフォルト: 5）
- `--gap`: 各分割での学習と検証の間のギャップ（デフォルト: 0）
- `--min-val-size`: 最小検証サイズ（デフォルト: 0）
- `--n-estimators`: LightGBM の推定器数（デフォルト: 600）
- `--learning-rate`: LightGBM の学習率（デフォルト: 0.05）
- `--reg-alpha`: L1正則化（デフォルト: 0.1）
- `--reg-lambda`: L2正則化（デフォルト: 0.1）
- `--random-state`: ランダムシード（デフォルト: 42）
- `--include-imputation-trace`: Stage 2特徴量を有効化

## 出力

### JSON（詳細結果）
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

### CSV（サマリー）
`results/ablation/SU3/sweep_summary.csv`:
```csv
timestamp,config_id,reappear_top_k,temporal_top_k,holiday_top_k,include_imputation_trace,oof_rmse,oof_msr,n_features,train_time_sec
2025-11-22_153045,1,20,20,20,False,0.012104,0.018567,474,45.2
...
```

## パラメータグリッド

### Stage 1（デフォルト - 48組み合わせ）
```python
PARAM_GRID = {
    'reappear_top_k': [10, 20, 30, 50],      # 4値
    'temporal_top_k': [10, 20, 30],          # 3値
    'holiday_top_k': [10, 20, 30, 50],       # 4値
}
# 合計: 4 × 3 × 4 = 48構成
```

### Stage 2（将来）
追加パラメータ: `include_imputation_trace: [False, True]`

## 推定実行時間

- **小規模テストデータ**（200行、2構成）: 約0.4秒
- **完全スイープ**（48構成、5分割、600推定器）: 約2-3時間

## ベースラインとの比較

- **SU1ベースライン**: OOF RMSE=0.01212、OOF MSR=0.01821、特徴量数=368
- **SU3目標**: OOF MSR ≤ 0.01921（SU1 + 0.001）

## テストカバレッジ

`test_su3_sweep.py`の4ユニットテスト：
1. `test_build_param_combinations`: パラメータグリッド生成
2. `test_evaluate_single_config_small_data`: エンドツーエンドOOF評価
3. `test_save_results`: 結果ファイル形式の検証
4. `test_build_param_combinations_with_imputation`: Stage 2パラメータ生成

すべてのテストはRuffおよびPyrightの検証に合格しています。

## 実装メモ

- クロスバリデーションにTimeSeriesSplitを使用
- 学習と検証の間のカスタムギャップをサポート
- 結果はOOF MSRの昇順でソート（小さいほど良い）
- 詳細（JSON）とサマリー（CSV）の両方に自動保存
- CSVは増分実行のために追記モードを使用
- 完全なパイプラインを含む: SU1生成 → SU3生成 → 前処理 → LightGBM

## 参照

- [SU3設計ドキュメント](../../../docs/feature_generation/SU3.md)
- [SU2スイープ実装](../../su2/sweep_oof.py)（参照パターン）
- [SU3学習スクリプト](train_su3.py)
