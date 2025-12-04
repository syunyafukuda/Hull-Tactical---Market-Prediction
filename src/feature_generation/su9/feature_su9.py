"""SU9（カレンダー・季節性特徴量）の生成ロジック。

本モジュールは決定可能な日付情報（曜日・月・祝日など）を特徴量化し、
時間構造軸の情報をモデルに渡す。

パイプライン上の位置:
  生データ → SU1 → SU5 → GroupImputers
  → **SU9** → 前処理（スケーラー＋OneHot）→ LightGBM

特徴量（32列）:
  - dow_0〜dow_6: 曜日 one-hot (7列)
  - dom_early, dom_mid, dom_late: 月内位置ビン (3列)
  - month_1〜month_12: 月 one-hot (12列)
  - is_month_start, is_month_end, is_qtr_start, is_qtr_end: 月末・期末フラグ (4列)
  - is_holiday, is_holiday_eve, is_holiday_next, is_bridge_day: 祝日・ブリッジ (4列)
  - yday_norm, days_to_year_end: 年内ポジション (2列)
  - 合計: 32列 (7+3+12+4+4+2、全特徴が有効な場合)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class SU9Config:
    """SU9（カレンダー・季節性）特徴生成の設定を保持するデータクラス。"""

    # 日付ID列名（営業日インデックス）
    id_column: str

    # 祝日カレンダーファイルパス（Optional）
    holiday_calendar_path: Optional[str] = None

    # 特徴量ON/OFFフラグ
    include_dow: bool = True
    include_dom: bool = True
    include_month: bool = True
    include_month_flags: bool = True
    include_holiday: bool = True
    include_year_position: bool = True

    # データ型
    dtype_flag: np.dtype[Any] = np.dtype("uint8")
    dtype_float: np.dtype[Any] = np.dtype("float32")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SU9Config":
        """YAML設定から SU9Config を生成する。"""
        id_column = mapping.get("id_column")
        if not id_column:
            raise ValueError("'id_column' is required and must not be empty")

        holiday_calendar_path = mapping.get("holiday_calendar_path")
        if holiday_calendar_path is not None:
            holiday_calendar_path = str(holiday_calendar_path)

        # 特徴量ON/OFFフラグ
        include_dow = bool(mapping.get("include_dow", True))
        include_dom = bool(mapping.get("include_dom", True))
        include_month = bool(mapping.get("include_month", True))
        include_month_flags = bool(mapping.get("include_month_flags", True))
        include_holiday = bool(mapping.get("include_holiday", True))
        include_year_position = bool(mapping.get("include_year_position", True))

        # データ型
        dtype_section = mapping.get("dtype", {})
        dtype_flag = np.dtype(dtype_section.get("flag", "uint8"))
        dtype_float = np.dtype(dtype_section.get("float", "float32"))

        return cls(
            id_column=str(id_column),
            holiday_calendar_path=holiday_calendar_path,
            include_dow=include_dow,
            include_dom=include_dom,
            include_month=include_month,
            include_month_flags=include_month_flags,
            include_holiday=include_holiday,
            include_year_position=include_year_position,
            dtype_flag=dtype_flag,
            dtype_float=dtype_float,
        )


def load_su9_config(config_path: str | Path) -> SU9Config:
    """SU9 設定 YAML を読み込み :class:`SU9Config` を生成する。"""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as fh:
        full_cfg: Mapping[str, Any] = yaml.safe_load(fh) or {}

    try:
        su9_section = full_cfg["su9"]
    except KeyError as exc:
        raise KeyError("'su9' section is required in feature_generation.yaml") from exc

    return SU9Config.from_mapping(su9_section)


class SU9FeatureGenerator(BaseEstimator, TransformerMixin):
    """SU9 カレンダー・季節性特徴量生成器。

    入力: GroupImputers 通過後の特徴行列（id_column を含む）
    出力: カレンダー・季節性特徴量

    特徴量の詳細:
    - dow_0〜dow_6: 曜日 one-hot (7列)
    - dom_early/mid/late: 月内位置ビン (3列)
    - month_1〜month_12: 月 one-hot (12列)
    - is_month_start/end, is_qtr_start/end: 月末・期末フラグ (4列)
    - is_holiday/holiday_eve/holiday_next/bridge_day: 祝日・ブリッジ (4列)
    - yday_norm, days_to_year_end: 年内ポジション (2列)
    合計: 32列 (7+3+12+4+4+2)
    """

    def __init__(self, config: SU9Config):
        self.config = config
        # fit 後に設定される属性
        self.feature_names_: Optional[List[str]] = None
        self._holiday_set: Optional[set[int]] = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SU9FeatureGenerator":
        """fit（sklearn互換のため実装、実際には何もしない）。

        Note:
            SU9 は決定可能な日付情報のみを使用するため、
            学習データからの統計量推定は不要。
            祝日カレンダーは fit 時に読み込み、後の transform で使用する。

        Args:
            X: 特徴行列
            y: unused (sklearn互換のため)

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SU9FeatureGenerator expects a pandas.DataFrame input")

        # 対象列の存在確認
        if self.config.id_column not in X.columns:
            raise ValueError(
                f"id_column '{self.config.id_column}' not found in input. "
                f"Available columns: {list(X.columns)[:20]}..."
            )

        # 祝日カレンダーの読み込み
        if self.config.holiday_calendar_path:
            self._holiday_set = self._load_holiday_calendar(
                self.config.holiday_calendar_path
            )
        else:
            # カレンダーが指定されていない場合は空セット
            self._holiday_set = set()

        # feature_names_ を組み立て
        self.feature_names_ = self._build_feature_names()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """SU9 特徴を生成。

        Args:
            X: 特徴行列（id_column を含む）

        Returns:
            SU9 特徴のDataFrame（追加された特徴のみ）
        """
        if self.feature_names_ is None:
            raise RuntimeError(
                "The transformer must be fitted before calling transform()."
            )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("SU9FeatureGenerator expects a pandas.DataFrame input")

        # id_column から date_id を取得
        date_ids_col = X[self.config.id_column]
        if not isinstance(date_ids_col, pd.Series):
            date_ids_col = pd.Series(date_ids_col)
        date_ids = date_ids_col.astype(int)

        features: dict[str, np.ndarray] = {}

        # 1. 曜日 (DOW)
        if self.config.include_dow:
            dow_features = self._compute_dow_features(date_ids)
            features.update(dow_features)

        # 2. 月内位置 (DOM)
        if self.config.include_dom:
            dom_features = self._compute_dom_features(date_ids)
            features.update(dom_features)

        # 3. 月 (Month)
        if self.config.include_month:
            month_features = self._compute_month_features(date_ids)
            features.update(month_features)

        # 4. 月末・期末フラグ
        if self.config.include_month_flags:
            flag_features = self._compute_month_flag_features(date_ids)
            features.update(flag_features)

        # 5. 祝日・ブリッジ
        if self.config.include_holiday:
            holiday_features = self._compute_holiday_features(date_ids)
            features.update(holiday_features)

        # 6. 年内ポジション
        if self.config.include_year_position:
            year_position_features = self._compute_year_position_features(date_ids)
            features.update(year_position_features)

        return pd.DataFrame(features, index=X.index)

    def _load_holiday_calendar(self, path: str) -> set[int]:
        """祝日カレンダーファイルを読み込む。

        Args:
            path: 祝日カレンダーファイルパス（CSV形式、date_id列を含む）

        Returns:
            祝日の date_id のセット
        """
        calendar_path = Path(path)
        if not calendar_path.exists():
            raise FileNotFoundError(f"Holiday calendar file not found: {path}")

        df = pd.read_csv(calendar_path)
        if "date_id" not in df.columns:
            raise ValueError(
                f"Holiday calendar must contain 'date_id' column. "
                f"Found columns: {list(df.columns)}"
            )

        return set(df["date_id"].astype(int).tolist())

    def _compute_dow_features(self, date_ids: pd.Series) -> dict[str, np.ndarray]:
        """曜日 one-hot 特徴を計算する。

        date_id を営業日インデックスとして解釈し、曜日を推定する。
        営業日ベースなので、date_id % 5 で曜日（月〜金）を推定。
        ただし、完全な曜日情報（日曜日含む）を表現するため 7 列で one-hot 化。
        """
        # 営業日ベース: date_id % 5 で月(0)〜金(4)を推定
        # 週末（土日）は営業日データには含まれないが、
        # 汎用性のため 7 列で表現（土日は常に 0）
        date_id_array = np.asarray(date_ids.values, dtype=np.int64)
        dow_values = (date_id_array % 5).astype(int)

        features: dict[str, np.ndarray] = {}
        for day in range(7):
            col_name = f"dow_{day}"
            if day < 5:
                # 月〜金
                features[col_name] = (dow_values == day).astype(self.config.dtype_flag)
            else:
                # 土日（営業日データには含まれない）
                features[col_name] = np.zeros(
                    len(date_ids), dtype=self.config.dtype_flag
                )

        return features

    def _compute_dom_features(self, date_ids: pd.Series) -> dict[str, np.ndarray]:
        """月内位置ビン特徴を計算する。

        営業日ベースで月内位置を推定:
        - early: 月初（1-10日相当、営業日で約1-7営業日目）
        - mid: 月中（11-20日相当、営業日で約8-14営業日目）
        - late: 月末（21日以降、営業日で約15営業日目以降）

        1ヶ月の営業日数を約22日と仮定し、date_id % 22 で月内位置を推定。
        """
        # 1ヶ月の営業日数を約22日と仮定
        TRADING_DAYS_PER_MONTH = 22

        date_id_array = np.asarray(date_ids.values, dtype=np.int64)
        dom_values = (date_id_array % TRADING_DAYS_PER_MONTH).astype(int)

        # early: 0-7 (約1-10日)
        # mid: 8-14 (約11-20日)
        # late: 15-21 (約21-月末)
        dom_early = (dom_values <= 7).astype(self.config.dtype_flag)
        dom_mid = ((dom_values > 7) & (dom_values <= 14)).astype(self.config.dtype_flag)
        dom_late = (dom_values > 14).astype(self.config.dtype_flag)

        return {
            "dom_early": dom_early,
            "dom_mid": dom_mid,
            "dom_late": dom_late,
        }

    def _compute_month_features(self, date_ids: pd.Series) -> dict[str, np.ndarray]:
        """月 one-hot 特徴を計算する。

        1年の営業日数を約252日と仮定し、date_id から月を推定。
        """
        # 1年の営業日数を約252日と仮定
        TRADING_DAYS_PER_YEAR = 252

        # 年内の営業日インデックス（0-251）
        date_id_array = np.asarray(date_ids.values, dtype=np.int64)
        yday = (date_id_array % TRADING_DAYS_PER_YEAR).astype(int)

        # 月を推定（各月約21営業日）
        # month = 1 + (yday // 21) の範囲を 1-12 にクリップ
        month_values = 1 + (yday // 21)
        month_values = np.clip(month_values, 1, 12)

        features: dict[str, np.ndarray] = {}
        for month in range(1, 13):
            col_name = f"month_{month}"
            features[col_name] = (month_values == month).astype(self.config.dtype_flag)

        return features

    def _compute_month_flag_features(
        self, date_ids: pd.Series
    ) -> dict[str, np.ndarray]:
        """月末・期末フラグ特徴を計算する。

        営業日ベースでフラグを推定:
        - is_month_start: 月初（各月の最初の営業日）
        - is_month_end: 月末（各月の最後の営業日）
        - is_qtr_start: 四半期初（1,4,7,10月の月初）
        - is_qtr_end: 四半期末（3,6,9,12月の月末）
        """
        TRADING_DAYS_PER_MONTH = 22

        date_id_array = np.asarray(date_ids.values, dtype=np.int64)
        dom_values = (date_id_array % TRADING_DAYS_PER_MONTH).astype(int)

        # 月初: dom == 0
        is_month_start = (dom_values == 0).astype(self.config.dtype_flag)

        # 月末: dom == TRADING_DAYS_PER_MONTH - 1
        is_month_end = (dom_values == TRADING_DAYS_PER_MONTH - 1).astype(
            self.config.dtype_flag
        )

        # 四半期判定のため月を計算
        TRADING_DAYS_PER_YEAR = 252
        yday = (date_id_array % TRADING_DAYS_PER_YEAR).astype(int)
        month_values = 1 + (yday // 21)
        month_values = np.clip(month_values, 1, 12)

        # is_qtr_start: 1,4,7,10月の月初
        qtr_start_months = {1, 4, 7, 10}
        is_qtr_month_start = np.isin(month_values, list(qtr_start_months))
        is_qtr_start = (is_month_start.astype(bool) & is_qtr_month_start).astype(
            self.config.dtype_flag
        )

        # is_qtr_end: 3,6,9,12月の月末
        qtr_end_months = {3, 6, 9, 12}
        is_qtr_month_end = np.isin(month_values, list(qtr_end_months))
        is_qtr_end = (is_month_end.astype(bool) & is_qtr_month_end).astype(
            self.config.dtype_flag
        )

        return {
            "is_month_start": is_month_start,
            "is_month_end": is_month_end,
            "is_qtr_start": is_qtr_start,
            "is_qtr_end": is_qtr_end,
        }

    def _compute_holiday_features(self, date_ids: pd.Series) -> dict[str, np.ndarray]:
        """祝日・ブリッジ特徴を計算する。

        営業日ベースで祝日関連フラグを計算:
        - is_holiday: 祝日（祝日カレンダーに含まれる日）
        - is_holiday_eve: 祝日前日（翌日が祝日）
        - is_holiday_next: 祝日翌日（前日が祝日）
        - is_bridge_day: ブリッジデー（祝日の前後両方が営業日）

        Note:
            祝日カレンダーが指定されていない場合、全て 0 になる。
        """
        holiday_set = self._holiday_set or set()
        date_id_values = date_ids.values.astype(int)
        n = len(date_id_values)

        is_holiday = np.zeros(n, dtype=self.config.dtype_flag)
        is_holiday_eve = np.zeros(n, dtype=self.config.dtype_flag)
        is_holiday_next = np.zeros(n, dtype=self.config.dtype_flag)
        is_bridge_day = np.zeros(n, dtype=self.config.dtype_flag)

        for i, date_id in enumerate(date_id_values):
            # is_holiday
            if date_id in holiday_set:
                is_holiday[i] = 1

            # is_holiday_eve: 翌日（date_id + 1）が祝日
            if (date_id + 1) in holiday_set:
                is_holiday_eve[i] = 1

            # is_holiday_next: 前日（date_id - 1）が祝日
            if (date_id - 1) in holiday_set:
                is_holiday_next[i] = 1

            # is_bridge_day: 前後両方が祝日
            # （祝日に挟まれた営業日）
            if (date_id - 1) in holiday_set and (date_id + 1) in holiday_set:
                if date_id not in holiday_set:  # 自身は営業日
                    is_bridge_day[i] = 1

        return {
            "is_holiday": is_holiday,
            "is_holiday_eve": is_holiday_eve,
            "is_holiday_next": is_holiday_next,
            "is_bridge_day": is_bridge_day,
        }

    def _compute_year_position_features(
        self, date_ids: pd.Series
    ) -> dict[str, np.ndarray]:
        """年内ポジション特徴を計算する。

        - yday_norm: 年内日数を [0, 1] に正規化
        - days_to_year_end: 年末までの残り営業日数（正規化）
        """
        TRADING_DAYS_PER_YEAR = 252

        # 年内の営業日インデックス（0-251）
        date_id_array = np.asarray(date_ids.values, dtype=np.int64)
        yday = (date_id_array % TRADING_DAYS_PER_YEAR).astype(int)

        # yday_norm: [0, 1] に正規化
        yday_norm = (yday / (TRADING_DAYS_PER_YEAR - 1)).astype(self.config.dtype_float)

        # days_to_year_end: 年末までの残り営業日数（正規化）
        days_to_year_end = (
            (TRADING_DAYS_PER_YEAR - 1 - yday) / (TRADING_DAYS_PER_YEAR - 1)
        ).astype(self.config.dtype_float)

        return {
            "yday_norm": yday_norm,
            "days_to_year_end": days_to_year_end,
        }

    def _build_feature_names(self) -> List[str]:
        """生成される特徴名のリストを作成。"""
        names: List[str] = []

        if self.config.include_dow:
            names.extend([f"dow_{i}" for i in range(7)])

        if self.config.include_dom:
            names.extend(["dom_early", "dom_mid", "dom_late"])

        if self.config.include_month:
            names.extend([f"month_{i}" for i in range(1, 13)])

        if self.config.include_month_flags:
            names.extend(
                ["is_month_start", "is_month_end", "is_qtr_start", "is_qtr_end"]
            )

        if self.config.include_holiday:
            names.extend(
                ["is_holiday", "is_holiday_eve", "is_holiday_next", "is_bridge_day"]
            )

        if self.config.include_year_position:
            names.extend(["yday_norm", "days_to_year_end"])

        return names

    def get_expected_column_count(self) -> int:
        """期待される列数を返す。

        Returns:
            生成される特徴列数（全機能有効時は32）
        """
        count = 0
        if self.config.include_dow:
            count += 7
        if self.config.include_dom:
            count += 3
        if self.config.include_month:
            count += 12
        if self.config.include_month_flags:
            count += 4
        if self.config.include_holiday:
            count += 4
        if self.config.include_year_position:
            count += 2
        return count


class SU9FeatureAugmenter(BaseEstimator, TransformerMixin):
    """SU9 特徴量を入力フレームへ追加するトランスフォーマー。

    入力: SU1+SU5+GroupImputers 通過後の特徴行列
    出力: 入力 + SU9 特徴

    使用例:
        augmenter = SU9FeatureAugmenter(su9_config)
        augmenter.fit(X)
        X_augmented = augmenter.transform(X)
    """

    def __init__(
        self,
        config: SU9Config,
        fill_value: float | None = None,
    ) -> None:
        """
        Args:
            config: SU9Config インスタンス
            fill_value: NaN を埋める値（None の場合は NaN のまま）
        """
        self.config = config
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SU9FeatureAugmenter":
        """SU9FeatureGenerator を内部で fit する。

        Args:
            X: SU1+SU5+GroupImputers 通過後の特徴行列
            y: unused (sklearn互換のため)

        Returns:
            self
        """
        frame = self._ensure_dataframe(X)

        # SU9 fit
        self.su9_generator_ = SU9FeatureGenerator(self.config)
        self.su9_generator_.fit(frame)

        # Store feature names
        self.su9_feature_names_ = list(self.su9_generator_.feature_names_ or [])
        self.input_columns_ = list(frame.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """SU9 特徴を生成し、元の DataFrame と結合する。

        Args:
            X: SU1+SU5+GroupImputers 通過後の特徴行列

        Returns:
            入力 + SU9 特徴の結合 DataFrame
        """
        if not hasattr(self, "su9_generator_"):
            raise RuntimeError("SU9FeatureAugmenter must be fitted before transform().")

        frame = self._ensure_dataframe(X)

        # Generate SU9 features
        su9_features = self.su9_generator_.transform(frame)
        su9_features = su9_features.reindex(columns=self.su9_feature_names_, copy=True)

        if self.fill_value is not None:
            su9_features = su9_features.fillna(self.fill_value)

        # Concatenate: original + SU9
        augmented = pd.concat([frame, su9_features], axis=1)
        augmented.index = frame.index
        return augmented

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SU9FeatureAugmenter expects a pandas.DataFrame input")
        return X.copy()
