# Feature Generation Pipeline

このディレクトリには、Hull Tactical Market Prediction プロジェクトの特徴量生成パイプラインに関するドキュメントが含まれています。

## パイプライン構成

特徴量生成は複数のステージで構成されています：

1. **SU1**: 基本的な特徴量生成（生データからの変換）
2. **SU2**: 二次特徴量生成（SU1出力からの高度な統計特徴）
3. **GroupImputer**: 欠損値補完

## SU2 パイプライン

SU2パイプラインの詳細については、[SU2.md](./SU2.md)を参照してください。

### データフロー

```
生データ → SU1 → SU2 → GroupImputer → モデル学習
```

### 設定ファイル

パイプラインの設定は `configs/feature_generation.yaml` で管理されています。

## 実装ファイル

- `src/feature_generation/su2/`: SU2特徴量生成の実装
- `tests/feature_generation/`: 特徴量生成のテストコード
