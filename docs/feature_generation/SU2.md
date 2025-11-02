# SU2: 二次特徴量生成パイプライン

## 概要

SU2は、SU1の出力を入力として受け取り、より高度な統計的特徴量を生成するパイプラインです。
時系列データに対して、ローリング統計、EWMA（指数加重移動平均）、遷移統計、正規化などを適用します。

## 主要な特徴

### 1. 未来参照の防止

- **厳格な時間制約**: t時点での特徴量計算には、t-1時点までのデータのみを使用
- **Fold境界でのリセット**: 各foldの開始時に統計量の状態をリセット
- これにより、リークを完全に防止し、本番環境での挙動を正確に再現

### 2. データ型の最適化

メモリ効率を考慮したデータ型の使用：

- `flag`: uint8（0/1のバイナリフラグ）
- `small_int`: int16（小さな整数値）
- `float`: float32（浮動小数点数）

### 3. 生成される特徴量

#### ローリング統計
- 移動平均（SMA）
- 移動標準偏差
- 移動最小値/最大値
- ローリングウィンドウ内での変化率

#### 指数加重移動平均（EWMA）
- 複数のスムージング係数での EWMA
- EWMA の差分・変化率

#### 遷移統計
- 前期との差分
- 変化率（パーセント変化）
- 符号変化の検出

#### 正規化
- Z-scoreによる標準化（ローリング統計を使用）
- Min-Max正規化（過去データのみ使用）

## パイプライン構成

### A系統: 既存パイプライン（SU1→確定前処理）
従来通りの処理フロー

### B系統: SU2導入パイプライン（SU1→SU2→GroupImputer）
新しい処理フローでSU2を組み込み

## 設定パラメータ

`configs/feature_generation.yaml` の `su2` セクションで以下を設定：

```yaml
su2:
  enabled: true
  
  rolling:
    windows: [7, 14, 30]  # ローリングウィンドウサイズ
    functions: [mean, std, min, max]  # 適用する統計関数
  
  ewma:
    spans: [5, 10, 20]  # EWMAのスパンパラメータ
  
  transitions:
    lags: [1, 7, 30]  # 遷移統計のラグ
  
  normalization:
    methods: [zscore, minmax]  # 正規化手法
    rolling_window: 30  # 正規化用のローリングウィンドウ
```

## 入力・出力

### 入力
- **SU1出力**: `artifacts/SU1/` 配下の特徴量DataFrame
- フォーマット: Pandas DataFrame（parquet形式）

### 出力

#### 学習時
- `artifacts/SU2/feature_list.json`: 生成された特徴量のリスト
- `artifacts/SU2/inference_bundle.pkl`: 推論用の状態とパラメータ
- `artifacts/SU2/model_meta.json`: メタデータ（SU2列とパラメータ）
- `artifacts/SU2/cv_fold_logs.csv`: CV foldごとのログ
- `artifacts/SU2/oof_predictions.csv`: Out-of-fold予測

#### 推論時
- `artifacts/SU2/submission.csv`: 提出用CSVファイル
- `artifacts/SU2/submission.parquet`: 提出用Parquetファイル

#### スイープ時
- `results/ablation/SU2/`: 各ポリシー候補のスイープ結果
  - 個別結果JSON
  - サマリCSV

## 実装詳細

### モジュール構成

```
src/feature_generation/su2/
├── __init__.py           # パッケージ初期化
├── feature_su2.py        # 特徴量生成ロジック
├── train_su2.py          # 学習パイプライン
├── predict_su2.py        # 推論パイプライン
└── sweep_oof.py          # ハイパーパラメータスイープ
```

### 主要クラス・関数

#### feature_su2.py
```python
class SU2FeatureGenerator:
    """SU2特徴量生成器
    
    SU1出力から二次特徴量を生成。状態を保持し、
    fold境界でリセット可能。
    """
    
    def fit_transform(self, df: pd.DataFrame, fold_id: int) -> pd.DataFrame:
        """学習データに対する特徴量生成"""
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """推論データに対する特徴量生成（状態を使用）"""
```

#### train_su2.py
```python
def train_su2_pipeline(config: dict) -> None:
    """SU2学習パイプラインのメイン関数
    
    - SU1出力を読み込み
    - SU2特徴量を生成
    - CVでモデル学習
    - artifacts/SU2/配下に出力
    """
```

#### predict_su2.py
```python
def predict_su2_pipeline(config: dict) -> None:
    """SU2推論パイプラインのメイン関数
    
    - 学習済みバンドルを読み込み
    - テストデータにSU2特徴量を適用
    - 予測結果を生成
    """
```

#### sweep_oof.py
```python
def sweep_su2_policies(config: dict) -> None:
    """SU2ポリシーのスイープ
    
    - 設定ファイルの候補リストを走査
    - 各候補でOOF評価
    - 結果をresults/ablation/SU2/に保存
    """
```

## テスト仕様

`tests/feature_generation/test_su2.py` で以下をテスト：

### 1. 全NaNケース
すべての値が欠損している場合の挙動を検証

### 2. 交互NaNケース
交互に欠損値が存在する場合の補完・計算を検証

### 3. 島状NaNケース
連続した欠損値の塊が存在する場合を検証

### 4. Fold境界リセット
Fold境界で状態が正しくリセットされることを検証

### 5. 未来参照チェック
t時点の特徴量が t-1 までのデータのみで計算されていることを検証

## 使用方法

### 学習
```bash
python -m src.feature_generation.su2.train_su2
```

### 推論
```bash
python -m src.feature_generation.su2.predict_su2
```

### ポリシースイープ
```bash
python -m src.feature_generation.su2.sweep_oof
```

## 注意事項

1. **メモリ使用量**: 大規模データセットでは、ローリング統計の計算でメモリを多く消費する可能性があります
2. **計算時間**: ウィンドウサイズや特徴量の数に応じて計算時間が増加します
3. **状態管理**: 推論時には学習時の状態（ローリング統計の履歴など）を正確に再現する必要があります

## 参考資料

- 親ディレクトリの README: [../README.md](./README.md)
- 設定ファイル: `configs/feature_generation.yaml`
- プロジェクトREADME: [../../README.md](../../README.md)
