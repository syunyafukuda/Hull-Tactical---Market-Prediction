"""SU8（ボラティリティ・レジーム特徴量）の生成ロジック。

本モジュールは局所ボラティリティとマーケットレジーム（低/中/高ボラ、トレンド/レンジなど）を
特徴量化し、「今どんな相場モードか」をモデルに渡す。

パイプライン上の位置:
  生データ → SU1 → SU5 → GroupImputers（M/E/I/P/S）
  → **SU8** → 前処理（スケーラー＋OneHot）→ LightGBM
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class SU8Config:
    """SU8（ボラティリティ・レジーム）特徴生成の設定を保持するデータクラス。"""

    # 代表インデックス列名（core_index_1に対応）
    core_index_col: str

    # ターゲット近傍リターン列名（ret_vol_adj 計算用）
    ret_base_col: str

    # EWMA パラメータ
    ewm_short_halflife: int = 5
    ewm_long_halflife: int = 20

    # 数値安定性用の epsilon
    eps: float = 1e-4

    # トレンド MA 窓サイズ
    trend_ma_short_window: int = 5
    trend_ma_long_window: int = 20

    # quantile 閾値
    vol_quantiles: Tuple[float, float] = (0.33, 0.66)
    trend_quantiles: Tuple[float, float] = (0.33, 0.66)

    # winsorize
    winsorize_ret_vol_adj_p: float = 0.99

    # カテゴリカル列（レジーム系）
    categorical_cols: Tuple[str, ...] = (
        "vol_regime_low",
        "vol_regime_mid",
        "vol_regime_high",
        "trend_regime_up",
        "trend_regime_down",
        "trend_regime_flat",
    )

    # 型
    dtype_float: np.dtype = np.dtype("float32")
    dtype_bool: np.dtype = np.dtype("bool")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SU8Config":
        """YAML設定から SU8Config を生成する。"""
        # 必須: core_index_col, ret_base_col
        core_index_col = mapping.get("core_index_col")
        if not core_index_col:
            raise ValueError("'core_index_col' is required and must not be empty")
        ret_base_col = mapping.get("ret_base_col")
        if not ret_base_col:
            raise ValueError("'ret_base_col' is required and must not be empty")

        # オプション
        ewm_short_halflife = int(mapping.get("ewm_short_halflife", 5))
        ewm_long_halflife = int(mapping.get("ewm_long_halflife", 20))
        eps = float(mapping.get("eps", 1e-4))
        trend_ma_short_window = int(mapping.get("trend_ma_short_window", 5))
        trend_ma_long_window = int(mapping.get("trend_ma_long_window", 20))

        vol_quantiles_raw = mapping.get("vol_quantiles", [0.33, 0.66])
        vol_quantiles = (float(vol_quantiles_raw[0]), float(vol_quantiles_raw[1]))
        trend_quantiles_raw = mapping.get("trend_quantiles", [0.33, 0.66])
        trend_quantiles = (float(trend_quantiles_raw[0]), float(trend_quantiles_raw[1]))

        winsorize_ret_vol_adj_p = float(mapping.get("winsorize_ret_vol_adj_p", 0.99))

        categorical_cols_raw = mapping.get(
            "categorical_cols",
            [
                "vol_regime_low",
                "vol_regime_mid",
                "vol_regime_high",
                "trend_regime_up",
                "trend_regime_down",
                "trend_regime_flat",
            ],
        )
        categorical_cols = tuple(str(c) for c in categorical_cols_raw)

        # データ型
        dtype_section = mapping.get("dtype", {})
        dtype_float = np.dtype(dtype_section.get("float", "float32"))
        dtype_bool = np.dtype(dtype_section.get("bool", "bool"))

        return cls(
            core_index_col=str(core_index_col),
            ret_base_col=str(ret_base_col),
            ewm_short_halflife=ewm_short_halflife,
            ewm_long_halflife=ewm_long_halflife,
            eps=eps,
            trend_ma_short_window=trend_ma_short_window,
            trend_ma_long_window=trend_ma_long_window,
            vol_quantiles=vol_quantiles,
            trend_quantiles=trend_quantiles,
            winsorize_ret_vol_adj_p=winsorize_ret_vol_adj_p,
            categorical_cols=categorical_cols,
            dtype_float=dtype_float,
            dtype_bool=dtype_bool,
        )


def load_su8_config(config_path: str | Path) -> SU8Config:
    """SU8 設定 YAML を読み込み :class:`SU8Config` を生成する。"""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as fh:
        full_cfg: Mapping[str, Any] = yaml.safe_load(fh) or {}

    try:
        su8_section = full_cfg["su8"]
    except KeyError as exc:
        raise KeyError("'su8' section is required in feature_generation.yaml") from exc

    return SU8Config.from_mapping(su8_section)


class SU8FeatureGenerator(BaseEstimator, TransformerMixin):
    """SU8 ボラティリティ・レジーム特徴量生成器。

    入力: GroupImputers 通過後の特徴行列（core_index_col, ret_base_col 列を含む）
    出力: ボラ指標 / ボラレジームタグ / トレンドレジームタグ / ボラ調整リターン

    特徴量の詳細:
    - ewmstd_short: EWMA 標準偏差（短期）
    - ewmstd_long: EWMA 標準偏差（長期）
    - vol_ratio: ewmstd_short / (ewmstd_long + eps)
    - vol_level: ewmstd_long
    - vol_regime_low/mid/high: ボラレジームタグ（bool）
    - trend_regime_up/down/flat: トレンドレジームタグ（bool）
    - ret_vol_adj: ボラ調整リターン
    """

    def __init__(self, config: SU8Config):
        self.config = config
        # fit 後に設定される属性
        self.quantiles_: Optional[Dict[str, float]] = None
        self.feature_names_: Optional[List[str]] = None
        self.winsorize_bounds_: Optional[Tuple[float, float]] = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SU8FeatureGenerator":
        """train データから quantile 閾値を学習する。

        Args:
            X: GroupImputers 通過後の特徴行列
            y: unused (sklearn互換のため)

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SU8FeatureGenerator expects a pandas.DataFrame input")

        # 対象列の存在確認
        if self.config.core_index_col not in X.columns:
            raise ValueError(
                f"core_index_col '{self.config.core_index_col}' not found in input. "
                f"Available columns: {list(X.columns)[:20]}..."
            )
        if self.config.ret_base_col not in X.columns:
            raise ValueError(
                f"ret_base_col '{self.config.ret_base_col}' not found in input. "
                f"Available columns: {list(X.columns)[:20]}..."
            )

        # ボラ系列を計算して quantile を fit
        core_series = pd.Series(X[self.config.core_index_col].astype(float))
        vol_level = self._compute_ewmstd(core_series, self.config.ewm_long_halflife)
        trend_indicator = self._compute_trend_indicator(core_series)

        # vol_level の quantile
        vol_level_valid = vol_level.dropna()
        q_low = float(vol_level_valid.quantile(self.config.vol_quantiles[0]))
        q_high = float(vol_level_valid.quantile(self.config.vol_quantiles[1]))

        # trend_indicator の quantile
        trend_valid = trend_indicator.dropna()
        tau_down = float(trend_valid.quantile(self.config.trend_quantiles[0]))
        tau_up = float(trend_valid.quantile(self.config.trend_quantiles[1]))

        self.quantiles_ = {
            "q_low": q_low,
            "q_high": q_high,
            "tau_down": tau_down,
            "tau_up": tau_up,
        }

        # ret_vol_adj の winsorize 閾値を fit
        ret_series = pd.Series(X[self.config.ret_base_col].astype(float))
        ewmstd_short = self._compute_ewmstd(core_series, self.config.ewm_short_halflife)
        denom = 1.0 + np.maximum(ewmstd_short, self.config.eps)
        ret_vol_adj = ret_series / denom
        ret_vol_adj_valid = ret_vol_adj.dropna()

        if len(ret_vol_adj_valid) > 0:
            p_low = 1.0 - self.config.winsorize_ret_vol_adj_p
            p_high = self.config.winsorize_ret_vol_adj_p
            clip_low = float(ret_vol_adj_valid.quantile(p_low))
            clip_high = float(ret_vol_adj_valid.quantile(p_high))
            self.winsorize_bounds_ = (clip_low, clip_high)
        else:
            self.winsorize_bounds_ = (float("-inf"), float("inf"))

        # feature_names_ を組み立て
        self.feature_names_ = self._build_feature_names()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """SU8 特徴を生成。

        前提: X は date_id で sort 済み。
        pandas の ewm/rolling は過去方向のみ参照するため、リークを防ぐ。

        Args:
            X: GroupImputers 通過後の特徴行列

        Returns:
            SU8 特徴のDataFrame（追加された特徴のみ）
        """
        if self.quantiles_ is None or self.feature_names_ is None:
            raise RuntimeError("The transformer must be fitted before calling transform().")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("SU8FeatureGenerator expects a pandas.DataFrame input")

        core_series = pd.Series(X[self.config.core_index_col].astype(float))
        ret_series = pd.Series(X[self.config.ret_base_col].astype(float))

        features: Dict[str, np.ndarray] = {}

        # 1. ボラティリティ指標
        vol_features = self._compute_volatility_features(core_series)
        features.update(vol_features)

        # 2. ボラレジームタグ
        vol_regime_features = self._compute_vol_regime(features["vol_level"])
        features.update(vol_regime_features)

        # 3. トレンドレジームタグ
        trend_features = self._compute_trend_regime(core_series)
        features.update(trend_features)

        # 4. ボラ調整リターン
        ret_vol_adj_features = self._compute_ret_vol_adj(ret_series, features["ewmstd_short"])
        features.update(ret_vol_adj_features)

        return pd.DataFrame(features, index=X.index)

    def _compute_ewmstd(self, series: pd.Series, halflife: int) -> pd.Series:
        """EWMA 標準偏差を計算する。"""
        return series.ewm(
            halflife=halflife, adjust=False, min_periods=halflife
        ).std()

    def _compute_trend_indicator(self, series: pd.Series) -> pd.Series:
        """トレンド指標（ma_short - ma_long）を計算する。"""
        ma_short = series.rolling(
            window=self.config.trend_ma_short_window,
            min_periods=self.config.trend_ma_short_window,
        ).mean()
        ma_long = series.rolling(
            window=self.config.trend_ma_long_window,
            min_periods=self.config.trend_ma_long_window,
        ).mean()
        return ma_short - ma_long

    def _compute_volatility_features(
        self, core_series: pd.Series
    ) -> Dict[str, np.ndarray]:
        """ボラティリティ指標を計算する。"""
        ewmstd_short = self._compute_ewmstd(core_series, self.config.ewm_short_halflife)
        ewmstd_long = self._compute_ewmstd(core_series, self.config.ewm_long_halflife)
        vol_ratio = ewmstd_short / (ewmstd_long + self.config.eps)
        vol_level = ewmstd_long

        return {
            "ewmstd_short": np.asarray(ewmstd_short.astype(self.config.dtype_float).values),
            "ewmstd_long": np.asarray(ewmstd_long.astype(self.config.dtype_float).values),
            "vol_ratio": np.asarray(vol_ratio.astype(self.config.dtype_float).values),
            "vol_level": np.asarray(vol_level.astype(self.config.dtype_float).values),
        }

    def _compute_vol_regime(self, vol_level: np.ndarray) -> Dict[str, np.ndarray]:
        """ボラレジームタグを計算する。"""
        if self.quantiles_ is None:
            raise RuntimeError("quantiles_ not set. Call fit() first.")

        q_low = self.quantiles_["q_low"]
        q_high = self.quantiles_["q_high"]

        vol_regime_low = (vol_level <= q_low).astype(self.config.dtype_bool)
        vol_regime_high = (vol_level > q_high).astype(self.config.dtype_bool)
        vol_regime_mid = (
            (vol_level > q_low) & (vol_level <= q_high)
        ).astype(self.config.dtype_bool)

        return {
            "vol_regime_low": vol_regime_low,
            "vol_regime_mid": vol_regime_mid,
            "vol_regime_high": vol_regime_high,
        }

    def _compute_trend_regime(self, core_series: pd.Series) -> Dict[str, np.ndarray]:
        """トレンドレジームタグを計算する。"""
        if self.quantiles_ is None:
            raise RuntimeError("quantiles_ not set. Call fit() first.")

        trend_indicator = self._compute_trend_indicator(core_series)
        trend_indicator_arr = np.asarray(trend_indicator.values)

        tau_down = self.quantiles_["tau_down"]
        tau_up = self.quantiles_["tau_up"]

        trend_regime_down = (trend_indicator_arr <= tau_down).astype(self.config.dtype_bool)
        trend_regime_up = (trend_indicator_arr >= tau_up).astype(self.config.dtype_bool)
        trend_regime_flat = (
            (trend_indicator_arr > tau_down) & (trend_indicator_arr < tau_up)
        ).astype(self.config.dtype_bool)

        return {
            "trend_regime_down": trend_regime_down,
            "trend_regime_up": trend_regime_up,
            "trend_regime_flat": trend_regime_flat,
        }

    def _compute_ret_vol_adj(
        self, ret_series: pd.Series, ewmstd_short: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """ボラ調整リターンを計算する。"""
        denom = 1.0 + np.maximum(ewmstd_short, self.config.eps)
        ret_vol_adj = np.asarray(ret_series.values, dtype=float) / denom

        # winsorize
        if self.winsorize_bounds_ is not None:
            clip_low, clip_high = self.winsorize_bounds_
            ret_vol_adj = np.clip(ret_vol_adj, clip_low, clip_high)

        return {
            "ret_vol_adj": ret_vol_adj.astype(self.config.dtype_float),
        }

    def _build_feature_names(self) -> List[str]:
        """生成される特徴名のリストを作成。"""
        return [
            "ewmstd_short",
            "ewmstd_long",
            "vol_ratio",
            "vol_level",
            "vol_regime_low",
            "vol_regime_mid",
            "vol_regime_high",
            "trend_regime_down",
            "trend_regime_up",
            "trend_regime_flat",
            "ret_vol_adj",
        ]

    def get_expected_column_count(self) -> int:
        """期待される列数を返す。

        Returns:
            生成される特徴列数 (11)
        """
        return 11


class SU8FeatureAugmenter(BaseEstimator, TransformerMixin):
    """SU8 特徴量を入力フレームへ追加するトランスフォーマー。

    入力: SU1+SU5+GroupImputers 通過後の特徴行列
    出力: 入力 + SU8 特徴

    使用例:
        augmenter = SU8FeatureAugmenter(su8_config)
        augmenter.fit(X)
        X_augmented = augmenter.transform(X)
    """

    def __init__(
        self,
        config: SU8Config,
        fill_value: float | None = None,
    ) -> None:
        """
        Args:
            config: SU8Config インスタンス
            fill_value: NaN を埋める値（None の場合は NaN のまま）
        """
        self.config = config
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SU8FeatureAugmenter":
        """SU8FeatureGenerator を内部で fit する。

        Args:
            X: SU1+SU5+GroupImputers 通過後の特徴行列
            y: unused (sklearn互換のため)

        Returns:
            self
        """
        frame = self._ensure_dataframe(X)

        # SU8 fit
        self.su8_generator_ = SU8FeatureGenerator(self.config)
        self.su8_generator_.fit(frame)

        # Store feature names
        self.su8_feature_names_ = list(self.su8_generator_.feature_names_ or [])
        self.input_columns_ = list(frame.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """SU8 特徴を生成し、元の DataFrame と結合する。

        Args:
            X: SU1+SU5+GroupImputers 通過後の特徴行列

        Returns:
            入力 + SU8 特徴の結合 DataFrame
        """
        if not hasattr(self, "su8_generator_"):
            raise RuntimeError("SU8FeatureAugmenter must be fitted before transform().")

        frame = self._ensure_dataframe(X)

        # Generate SU8 features
        su8_features = self.su8_generator_.transform(frame)
        su8_features = su8_features.reindex(columns=self.su8_feature_names_, copy=True)

        if self.fill_value is not None:
            su8_features = su8_features.fillna(self.fill_value)

        # Concatenate: original + SU8
        augmented = pd.concat([frame, su8_features], axis=1)
        augmented.index = frame.index
        return augmented

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SU8FeatureAugmenter expects a pandas.DataFrame input")
        return X.copy()
