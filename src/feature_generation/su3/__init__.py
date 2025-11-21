"""SU3 (Structure-based Unit 3) 特徴量生成モジュール。

SU3はSU1の出力を入力として、欠損パターンの時間的変化と代入影響を捕捉する特徴を生成します。

主な特徴カテゴリ:
- 遷移フラグ: NaN ↔ 観測の切り替わり検知
- 再出現パターン: 欠損後の観測復帰までの間隔
- 時間的バイアス: 曜日・月次の欠損パターン
- 祝日交差: 祝日と欠損の交差パターン
"""

from src.feature_generation.su3.feature_su3 import (
	SU3Config,
	SU3FeatureGenerator,
	load_su3_config,
)

__all__ = [
	"SU3Config",
	"SU3FeatureGenerator",
	"load_su3_config",
]
