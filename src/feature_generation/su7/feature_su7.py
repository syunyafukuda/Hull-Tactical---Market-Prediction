"""SU7（モメンタム・リバーサル特徴量）の生成ロジック。

本モジュールは価格・リターン系列のモメンタム/リバーサル構造を特徴量化し、
短期〜中期の値動きパターンを明示的にモデルへ渡す。

パイプライン上の位置:
  生データ → SU1 → SU5 → GroupImputers（M/E/I/P/S）
  → **SU7** → 前処理（スケーラー＋OneHot）→ LightGBM
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
class SU7Config:
    """SU7（モメンタム・リバーサル）特徴生成の設定を保持するデータクラス。"""

    # 対象列（1日リターン系列 r_t を前提）
    su7_base_cols: Tuple[str, ...]

    # lag/diff の k 値
    lags: Tuple[int, ...] = (1, 5, 20)

    # ローリング窓サイズ
    windows: Tuple[int, ...] = (5, 20)

    # RSI の halflife
    halflife_rsi: int = 5

    # 数値安定性用の epsilon
    eps: float = 1e-8

    # RS のクリップ上限
    rs_max: float = 100.0

    # 型
    dtype_float: np.dtype = np.dtype("float32")
    dtype_int: np.dtype = np.dtype("int8")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SU7Config":
        """YAML設定から SU7Config を生成する。"""
        # 必須: su7_base_cols
        base_cols_raw = mapping.get("su7_base_cols", [])
        if not base_cols_raw:
            raise ValueError("'su7_base_cols' is required and must not be empty")
        su7_base_cols = tuple(str(c) for c in base_cols_raw)

        # オプション: lags
        lags_raw = mapping.get("lags", [1, 5, 20])
        lags = tuple(int(k) for k in lags_raw)

        # オプション: windows
        windows_raw = mapping.get("windows", [5, 20])
        windows = tuple(int(w) for w in windows_raw)

        # オプション: halflife_rsi
        halflife_rsi = int(mapping.get("halflife_rsi", 5))

        # オプション: eps
        eps = float(mapping.get("eps", 1e-8))

        # オプション: rs_max
        rs_max = float(mapping.get("rs_max", 100.0))

        # データ型
        dtype_section = mapping.get("dtype", {})
        dtype_float = np.dtype(dtype_section.get("float", "float32"))
        dtype_int = np.dtype(dtype_section.get("int", "int8"))

        return cls(
            su7_base_cols=su7_base_cols,
            lags=lags,
            windows=windows,
            halflife_rsi=halflife_rsi,
            eps=eps,
            rs_max=rs_max,
            dtype_float=dtype_float,
            dtype_int=dtype_int,
        )


def load_su7_config(config_path: str | Path) -> SU7Config:
    """SU7 設定 YAML を読み込み :class:`SU7Config` を生成する。"""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as fh:
        full_cfg: Mapping[str, Any] = yaml.safe_load(fh) or {}

    try:
        su7_section = full_cfg["su7"]
    except KeyError as exc:
        raise KeyError("'su7' section is required in feature_generation.yaml") from exc

    return SU7Config.from_mapping(su7_section)


class SU7FeatureGenerator(BaseEstimator, TransformerMixin):
    """SU7 モメンタム・リバーサル特徴量生成器。

    入力: GroupImputers 通過後の特徴行列（su7_base_cols 列を含む）
    出力: diff/lag/rolling/RSI/sign 特徴

    特徴量の詳細:
    - diff_k/<col>: r_t[t] - r_t[t-k] (k ∈ lags)
    - lag_k/<col>: r_t[t-k] (k ∈ lags)
    - roll_ret_W/<col>: 過去W日リターン和
    - roll_mean_diff_W/<col>: 過去W日 diff_1 の平均
    - rsi_5/<col>: RSI ライク指標
    - sign_r_t/<col>: sign(r_t) ∈ {-1, 0, 1}
    """

    def __init__(self, config: SU7Config):
        self.config = config
        self.base_cols_: Optional[List[str]] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SU7FeatureGenerator":
        """対象列の存在確認と列順固定。

        SU7 は deterministic transform であり、学習データからパラメータを推定しない。
        fit は su7_base_cols の存在確認と列順固定のみを行う。

        Args:
            X: GroupImputers 通過後の特徴行列
            y: unused (sklearn互換のため)

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SU7FeatureGenerator expects a pandas.DataFrame input")

        # 対象列の存在確認
        missing_cols = [c for c in self.config.su7_base_cols if c not in X.columns]
        if missing_cols:
            raise ValueError(
                f"SU7 base columns not found in input: {missing_cols}. "
                f"Available columns: {list(X.columns)[:20]}..."
            )

        # 列順を固定
        self.base_cols_ = list(self.config.su7_base_cols)

        # feature_names_ を組み立て
        self.feature_names_ = self._build_feature_names()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """SU7特徴を生成。

        前提: X は date_id で sort 済み。
        pandas の shift/rolling/ewm は過去方向のみ参照するため、リークを防ぐ。

        Args:
            X: GroupImputers 通過後の特徴行列

        Returns:
            SU7特徴のDataFrame（追加された特徴のみ）
        """
        if self.base_cols_ is None:
            raise RuntimeError("The transformer must be fitted before calling transform().")

        features: Dict[str, np.ndarray] = {}

        # 各ベース列に対して特徴を生成
        for col in self.base_cols_:
            r_t = pd.Series(X[col].astype(float))

            # 1. diff / lag
            diff_lag_features = self._compute_diff_lag(r_t, col)
            features.update(diff_lag_features)

            # 2. ローリング・モメンタム
            rolling_features = self._compute_rolling(r_t, col)
            features.update(rolling_features)

            # 3. RSI ライク指標
            rsi_features = self._compute_rsi(r_t, col)
            features.update(rsi_features)

            # 4. 方向フラグ
            sign_features = self._compute_sign(r_t, col)
            features.update(sign_features)

        return pd.DataFrame(features, index=X.index)

    def _compute_diff_lag(self, r_t: pd.Series, col: str) -> Dict[str, np.ndarray]:
        """diff_k と lag_k を計算する。"""
        features: Dict[str, np.ndarray] = {}

        for k in self.config.lags:
            # diff_k = r_t[t] - r_t[t-k]
            diff_k = r_t - r_t.shift(k)
            features[f"diff_{k}/{col}"] = np.asarray(
                diff_k.astype(self.config.dtype_float).values
            )

            # lag_k = r_t[t-k]
            lag_k = r_t.shift(k)
            features[f"lag_{k}/{col}"] = np.asarray(
                lag_k.astype(self.config.dtype_float).values
            )

        return features

    def _compute_rolling(self, r_t: pd.Series, col: str) -> Dict[str, np.ndarray]:
        """ローリング・モメンタム特徴を計算する。"""
        features: Dict[str, np.ndarray] = {}

        # diff_1 は roll_mean_diff で使用
        diff_1 = r_t - r_t.shift(1)

        for w in self.config.windows:
            # roll_ret_W = 過去W日リターン和
            roll_ret = r_t.rolling(window=w, min_periods=w).sum()
            features[f"roll_ret_{w}/{col}"] = np.asarray(
                roll_ret.astype(self.config.dtype_float).values
            )

            # roll_mean_diff_W = 過去W日 diff_1 の平均
            roll_mean_diff = diff_1.rolling(window=w, min_periods=w).mean()
            features[f"roll_mean_diff_{w}/{col}"] = np.asarray(
                roll_mean_diff.astype(self.config.dtype_float).values
            )

        return features

    def _compute_rsi(self, r_t: pd.Series, col: str) -> Dict[str, np.ndarray]:
        """RSI ライク指標を計算する。

        Note: This is an RSI-like indicator that operates directly on the return
        series (r_t), not on price differences. This is intentional as specified
        in SU7.md - the input is already a 1-day return series, so we use the
        returns directly rather than computing differences of differences.

        手順 (as per SU7.md specification):
        1. gains = max(r_t, 0)
        2. losses = max(-r_t, 0)
        3. ema_gain = ema(gains, halflife)
        4. ema_loss = ema(losses, halflife)
        5. rs = ema_gain / (ema_loss + eps)
        6. rsi = rs / (1 + rs)

        数値安定性:
        - eps で ema_loss ≈ 0 のときの発散を抑える
        - rs を rs_max でクリップし、NaN/Inf を防ぐ
        """
        gains = r_t.clip(lower=0)
        losses = (-r_t).clip(lower=0)

        # EMA with halflife
        ema_gain = gains.ewm(halflife=self.config.halflife_rsi, min_periods=1).mean()
        ema_loss = losses.ewm(halflife=self.config.halflife_rsi, min_periods=1).mean()

        # RS with numerical stability
        rs = ema_gain / (ema_loss + self.config.eps)
        rs = rs.clip(upper=self.config.rs_max)

        # RSI = rs / (1 + rs)
        rsi = rs / (1 + rs)

        return {
            f"rsi_{self.config.halflife_rsi}/{col}": np.asarray(
                rsi.astype(self.config.dtype_float).values
            )
        }

    def _compute_sign(self, r_t: pd.Series, col: str) -> Dict[str, np.ndarray]:
        """方向フラグ sign(r_t) ∈ {-1, 0, 1} を計算する。"""
        sign_values = np.sign(np.asarray(r_t.values)).astype(self.config.dtype_int)
        return {f"sign_r_t/{col}": sign_values}

    def _build_feature_names(self) -> List[str]:
        """生成される特徴名のリストを作成。"""
        names: List[str] = []

        if self.base_cols_ is None:
            return names

        for col in self.base_cols_:
            # diff / lag
            for k in self.config.lags:
                names.append(f"diff_{k}/{col}")
                names.append(f"lag_{k}/{col}")

            # rolling
            for w in self.config.windows:
                names.append(f"roll_ret_{w}/{col}")
                names.append(f"roll_mean_diff_{w}/{col}")

            # RSI
            names.append(f"rsi_{self.config.halflife_rsi}/{col}")

            # sign
            names.append(f"sign_r_t/{col}")

        return names

    def get_expected_column_count(self, n_base_cols: Optional[int] = None) -> int:
        """期待される列数を計算する。

        Args:
            n_base_cols: ベース列数（Noneの場合はconfig.su7_base_colsから取得）

        Returns:
            生成される特徴列数

        計算式:
        - diff/lag: 2 * len(lags) * B = 6B (lags=[1,5,20]の場合)
        - rolling: 2 * len(windows) * B = 4B (windows=[5,20]の場合)
        - rsi: 1 * B = B
        - sign: 1 * B = B
        - 合計: (6 + 4 + 1 + 1) * B = 12B
        """
        if n_base_cols is None:
            n_base_cols = len(self.config.su7_base_cols)

        n_lags = len(self.config.lags)
        n_windows = len(self.config.windows)

        # diff + lag: 2 features per lag
        diff_lag_count = 2 * n_lags * n_base_cols

        # rolling: 2 features per window
        rolling_count = 2 * n_windows * n_base_cols

        # rsi: 1 feature per base col
        rsi_count = n_base_cols

        # sign: 1 feature per base col
        sign_count = n_base_cols

        return diff_lag_count + rolling_count + rsi_count + sign_count


class SU7FeatureAugmenter(BaseEstimator, TransformerMixin):
    """SU7 特徴量を入力フレームへ追加するトランスフォーマー。

    入力: SU1+SU5+GroupImputers 通過後の特徴行列
    出力: 入力 + SU7 特徴

    使用例:
        augmenter = SU7FeatureAugmenter(su7_config)
        augmenter.fit(X)
        X_augmented = augmenter.transform(X)
    """

    def __init__(
        self,
        config: SU7Config,
        fill_value: float | None = None,
    ) -> None:
        """
        Args:
            config: SU7Config インスタンス
            fill_value: NaN を埋める値（None の場合は NaN のまま）
        """
        self.config = config
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SU7FeatureAugmenter":
        """SU7FeatureGenerator を内部で fit する。

        Args:
            X: SU1+SU5+GroupImputers 通過後の特徴行列
            y: unused (sklearn互換のため)

        Returns:
            self
        """
        frame = self._ensure_dataframe(X)

        # SU7 fit
        self.su7_generator_ = SU7FeatureGenerator(self.config)
        self.su7_generator_.fit(frame)

        # Store feature names
        self.su7_feature_names_ = list(self.su7_generator_.feature_names_ or [])
        self.input_columns_ = list(frame.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """SU7 特徴を生成し、元の DataFrame と結合する。

        Args:
            X: SU1+SU5+GroupImputers 通過後の特徴行列

        Returns:
            入力 + SU7 特徴の結合 DataFrame
        """
        if not hasattr(self, "su7_generator_"):
            raise RuntimeError("SU7FeatureAugmenter must be fitted before transform().")

        frame = self._ensure_dataframe(X)

        # Generate SU7 features
        su7_features = self.su7_generator_.transform(frame)
        su7_features = su7_features.reindex(columns=self.su7_feature_names_, copy=True)

        if self.fill_value is not None:
            su7_features = su7_features.fillna(self.fill_value)

        # Concatenate: original + SU7
        augmented = pd.concat([frame, su7_features], axis=1)
        augmented.index = frame.index
        return augmented

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SU7FeatureAugmenter expects a pandas.DataFrame input")
        return X.copy()
