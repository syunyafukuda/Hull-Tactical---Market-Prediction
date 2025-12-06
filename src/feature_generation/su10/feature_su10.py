"""SU10（外部レジーム特徴量）の生成ロジック。

本モジュールは SPY Historical Data から算出したボラティリティ・トレンドレジーム特徴を生成する。
train/test 内部データとは独立した情報軸を提供する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class SU10Config:
    """SU10 特徴量生成設定。

    scripts/su10/build_su10_external_regime.py の設定を移植。
    """

    # ボラティリティ設定
    ewm_halflife_short: int = 20
    ewm_halflife_long: int = 60
    vol_quantile_low: float = 0.33
    vol_quantile_high: float = 0.66

    # トレンド設定
    trend_ewm_halflife_short: int = 20
    trend_ewm_halflife_long: int = 60
    trend_quantile_down: float = 0.33
    trend_quantile_up: float = 0.66

    # リターン設定
    return_periods: tuple[int, ...] = (5, 20)
    vol_adj_period: int = 5
    winsorize_low: float = 0.01
    winsorize_high: float = 0.99

    # train/test 分割（date_id 基準）
    train_max_date_id: int = 8979  # train: 0〜8979, test: 8980〜8989

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SU10Config":
        """Mapping (e.g., YAML section) から SU10Config を生成する。"""
        return_periods = mapping.get("return_periods", (5, 20))
        if isinstance(return_periods, list):
            return_periods = tuple(return_periods)

        return cls(
            ewm_halflife_short=int(mapping.get("ewm_halflife_short", 20)),
            ewm_halflife_long=int(mapping.get("ewm_halflife_long", 60)),
            vol_quantile_low=float(mapping.get("vol_quantile_low", 0.33)),
            vol_quantile_high=float(mapping.get("vol_quantile_high", 0.66)),
            trend_ewm_halflife_short=int(mapping.get("trend_ewm_halflife_short", 20)),
            trend_ewm_halflife_long=int(mapping.get("trend_ewm_halflife_long", 60)),
            trend_quantile_down=float(mapping.get("trend_quantile_down", 0.33)),
            trend_quantile_up=float(mapping.get("trend_quantile_up", 0.66)),
            return_periods=return_periods,
            vol_adj_period=int(mapping.get("vol_adj_period", 5)),
            winsorize_low=float(mapping.get("winsorize_low", 0.01)),
            winsorize_high=float(mapping.get("winsorize_high", 0.99)),
            train_max_date_id=int(mapping.get("train_max_date_id", 8979)),
        )


def load_su10_config(config_path: str | Path) -> SU10Config:
    """SU10 設定 YAML を読み込み :class:`SU10Config` を生成する。"""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as fh:
        full_cfg: Mapping[str, Any] = yaml.safe_load(fh) or {}

    su10_section = full_cfg.get("su10", {})
    return SU10Config.from_mapping(su10_section)


# 特徴量カラム名リスト（順序固定）
SU10_FEATURE_COLUMNS = [
    "su10_spx_ewmstd_20d",
    "su10_spx_ewmstd_60d",
    "su10_spx_vol_ratio",
    "su10_spx_vol_level",
    "su10_spx_vol_regime_low",
    "su10_spx_vol_regime_mid",
    "su10_spx_vol_regime_high",
    "su10_spx_trend_indicator",
    "su10_spx_trend_regime_down",
    "su10_spx_trend_regime_flat",
    "su10_spx_trend_regime_up",
    "su10_spx_ret_5d",
    "su10_spx_ret_20d",
    "su10_spx_ret_vol_adj_5d",
]


class SU10FeatureGenerator(BaseEstimator, TransformerMixin):
    """外部レジーム特徴生成クラス。

    SPY Historical Data から SU10 特徴量を生成する。
    fit() で train 期間の分位点を記録し、transform() で特徴量を生成する。

    Parameters
    ----------
    config : SU10Config
        特徴量生成設定

    Attributes
    ----------
    vol_mean_ : float
        train期間のボラティリティ平均値
    vol_std_ : float
        train期間のボラティリティ標準偏差
    vol_quantiles_ : tuple[float, float]
        ボラティリティレジーム分類の分位点 (low, high)
    trend_quantiles_ : tuple[float, float]
        トレンドレジーム分類の分位点 (down, up)
    winsorize_bounds_ : tuple[float, float]
        Winsorize のクリップ範囲 (low, high)
    feature_names_ : list[str]
        生成される特徴量名のリスト
    """

    def __init__(self, config: SU10Config) -> None:
        self.config = config
        self.vol_mean_: float | None = None
        self.vol_std_: float | None = None
        self.vol_quantiles_: tuple[float, float] | None = None
        self.trend_quantiles_: tuple[float, float] | None = None
        self.winsorize_bounds_: tuple[float, float] | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, external_df: pd.DataFrame, y: Any = None) -> "SU10FeatureGenerator":
        """train 期間の分位点を記録する。

        Parameters
        ----------
        external_df : pd.DataFrame
            SPY Historical Data (date_id, Adj Close 等を含む)
        y : Any
            未使用 (sklearn 互換のため)

        Returns
        -------
        SU10FeatureGenerator
            self
        """
        if not isinstance(external_df, pd.DataFrame):
            raise TypeError("SU10FeatureGenerator expects a pandas.DataFrame input")

        df = external_df.copy()

        # date_id が必要
        if "date_id" not in df.columns:
            raise ValueError("'date_id' column is required in external_df")

        # Adj Close が必要
        if "Adj Close" not in df.columns:
            raise ValueError("'Adj Close' column is required in external_df")

        # 日次リターン
        df["ret"] = df["Adj Close"].pct_change()

        # ボラティリティ計算
        df["ewmstd_short"] = (
            df["ret"].ewm(halflife=self.config.ewm_halflife_short, adjust=False).std()
        )

        # train 期間マスク
        train_mask = df["date_id"] <= self.config.train_max_date_id

        # ボラティリティ標準化パラメータ
        vol_train = df.loc[train_mask, "ewmstd_short"]
        self.vol_mean_ = float(vol_train.mean())
        self.vol_std_ = float(vol_train.std())

        # ボラティリティ水準（標準化）
        df["vol_level"] = (df["ewmstd_short"] - self.vol_mean_) / (self.vol_std_ + 1e-8)

        # ボラティリティレジーム分位点
        vol_level_train = df.loc[train_mask, "vol_level"]
        q_low = float(vol_level_train.quantile(self.config.vol_quantile_low))
        q_high = float(vol_level_train.quantile(self.config.vol_quantile_high))
        self.vol_quantiles_ = (q_low, q_high)

        # トレンド指標計算
        df["ma_short"] = (
            df["Adj Close"]
            .ewm(halflife=self.config.trend_ewm_halflife_short, adjust=False)
            .mean()
        )
        df["ma_long"] = (
            df["Adj Close"]
            .ewm(halflife=self.config.trend_ewm_halflife_long, adjust=False)
            .mean()
        )
        df["trend_indicator"] = df["ma_short"] - df["ma_long"]

        # トレンドレジーム分位点
        trend_train = df.loc[train_mask, "trend_indicator"]
        tau_down = float(trend_train.quantile(self.config.trend_quantile_down))
        tau_up = float(trend_train.quantile(self.config.trend_quantile_up))
        self.trend_quantiles_ = (tau_down, tau_up)

        # Winsorize 境界（ボラ調整リターン用）
        ret_col = f"ret_{self.config.vol_adj_period}d"
        df[ret_col] = df["Adj Close"].pct_change(periods=self.config.vol_adj_period)
        denom = 1 + df["ewmstd_short"].clip(lower=1e-4)
        df["ret_vol_adj"] = df[ret_col] / denom

        ret_vol_adj_train = df.loc[train_mask, "ret_vol_adj"].dropna()
        clip_low = float(ret_vol_adj_train.quantile(self.config.winsorize_low))
        clip_high = float(ret_vol_adj_train.quantile(self.config.winsorize_high))
        self.winsorize_bounds_ = (clip_low, clip_high)

        self.feature_names_ = SU10_FEATURE_COLUMNS.copy()

        return self

    def transform(self, external_df: pd.DataFrame) -> pd.DataFrame:
        """特徴量を生成する。

        Parameters
        ----------
        external_df : pd.DataFrame
            SPY Historical Data (date_id, Adj Close 等を含む)

        Returns
        -------
        pd.DataFrame
            date_id と SU10 特徴量のみを含む DataFrame
        """
        if (
            self.vol_mean_ is None
            or self.vol_std_ is None
            or self.vol_quantiles_ is None
            or self.trend_quantiles_ is None
            or self.winsorize_bounds_ is None
        ):
            raise RuntimeError(
                "SU10FeatureGenerator must be fitted before calling transform()."
            )

        df = external_df.copy()

        # date_id が必要
        if "date_id" not in df.columns:
            raise ValueError("'date_id' column is required in external_df")

        # Adj Close が必要
        if "Adj Close" not in df.columns:
            raise ValueError("'Adj Close' column is required in external_df")

        # 日次リターン
        df["ret"] = df["Adj Close"].pct_change()

        # ========================================
        # ボラティリティ指標
        # ========================================
        df["su10_spx_ewmstd_20d"] = (
            df["ret"].ewm(halflife=self.config.ewm_halflife_short, adjust=False).std()
        )
        df["su10_spx_ewmstd_60d"] = (
            df["ret"].ewm(halflife=self.config.ewm_halflife_long, adjust=False).std()
        )
        df["su10_spx_vol_ratio"] = df["su10_spx_ewmstd_20d"] / (
            df["su10_spx_ewmstd_60d"] + 1e-8
        )

        # ボラ水準（標準化）- fit で算出した mean/std を使用
        df["su10_spx_vol_level"] = (df["su10_spx_ewmstd_20d"] - self.vol_mean_) / (
            self.vol_std_ + 1e-8
        )

        # ========================================
        # ボラティリティレジーム（3区分）
        # ========================================
        q_low, q_high = self.vol_quantiles_
        df["su10_spx_vol_regime_low"] = (df["su10_spx_vol_level"] <= q_low).astype(
            np.uint8
        )
        df["su10_spx_vol_regime_mid"] = (
            (df["su10_spx_vol_level"] > q_low) & (df["su10_spx_vol_level"] <= q_high)
        ).astype(np.uint8)
        df["su10_spx_vol_regime_high"] = (df["su10_spx_vol_level"] > q_high).astype(
            np.uint8
        )

        # ========================================
        # トレンド指標
        # ========================================
        df["ma_short"] = (
            df["Adj Close"]
            .ewm(halflife=self.config.trend_ewm_halflife_short, adjust=False)
            .mean()
        )
        df["ma_long"] = (
            df["Adj Close"]
            .ewm(halflife=self.config.trend_ewm_halflife_long, adjust=False)
            .mean()
        )
        df["su10_spx_trend_indicator"] = df["ma_short"] - df["ma_long"]

        # ========================================
        # トレンドレジーム（3区分）
        # ========================================
        tau_down, tau_up = self.trend_quantiles_
        df["su10_spx_trend_regime_down"] = (
            df["su10_spx_trend_indicator"] <= tau_down
        ).astype(np.uint8)
        df["su10_spx_trend_regime_flat"] = (
            (df["su10_spx_trend_indicator"] > tau_down)
            & (df["su10_spx_trend_indicator"] <= tau_up)
        ).astype(np.uint8)
        df["su10_spx_trend_regime_up"] = (
            df["su10_spx_trend_indicator"] > tau_up
        ).astype(np.uint8)

        # ========================================
        # 期間リターン
        # ========================================
        for period in self.config.return_periods:
            df[f"su10_spx_ret_{period}d"] = df["Adj Close"].pct_change(periods=period)

        # ========================================
        # ボラ調整リターン
        # ========================================
        ret_col = f"su10_spx_ret_{self.config.vol_adj_period}d"
        denom = 1 + df["su10_spx_ewmstd_20d"].clip(lower=1e-4)
        df["su10_spx_ret_vol_adj_5d"] = df[ret_col] / denom

        # Winsorize（fit で算出した分位点でクリップ）
        clip_low, clip_high = self.winsorize_bounds_
        df["su10_spx_ret_vol_adj_5d"] = df["su10_spx_ret_vol_adj_5d"].clip(
            lower=clip_low, upper=clip_high
        )

        # ========================================
        # SU10 列のみ抽出
        # ========================================
        su10_cols = ["date_id"] + SU10_FEATURE_COLUMNS
        result = pd.DataFrame(df[su10_cols].copy())

        # float32 に変換（メモリ削減）
        for col in result.columns:
            if result[col].dtype == np.float64:
                result[col] = result[col].astype(np.float32)

        return result

    def fit_transform(  # type: ignore[override]
        self, external_df: pd.DataFrame, y: Any = None
    ) -> pd.DataFrame:
        """fit と transform を連続して実行する。

        Parameters
        ----------
        external_df : pd.DataFrame
            SPY Historical Data (date_id, Adj Close 等を含む)
        y : Any
            未使用 (sklearn 互換のため)

        Returns
        -------
        pd.DataFrame
            date_id と SU10 特徴量のみを含む DataFrame
        """
        return self.fit(external_df, y).transform(external_df)
