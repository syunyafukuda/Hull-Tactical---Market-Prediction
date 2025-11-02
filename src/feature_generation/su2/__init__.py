"""SU2: 二次特徴量生成パッケージ

SU1の出力を入力として受け取り、ローリング統計、EWMA、遷移統計、
正規化などの高度な特徴量を生成します。

主要モジュール:
- feature_su2: 特徴量生成のコアロジック
- train_su2: 学習パイプライン
- predict_su2: 推論パイプライン
- sweep_oof: ハイパーパラメータスイープ
"""

from __future__ import annotations

__all__ = [
    "SU2FeatureGenerator",
    "train_su2_pipeline",
    "predict_su2_pipeline",
    "sweep_su2_policies",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "SU2FeatureGenerator":
        from .feature_su2 import SU2FeatureGenerator
        return SU2FeatureGenerator
    elif name == "train_su2_pipeline":
        from .train_su2 import train_su2_pipeline
        return train_su2_pipeline
    elif name == "predict_su2_pipeline":
        from .predict_su2 import predict_su2_pipeline
        return predict_su2_pipeline
    elif name == "sweep_su2_policies":
        from .sweep_oof import sweep_su2_policies
        return sweep_su2_policies
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
