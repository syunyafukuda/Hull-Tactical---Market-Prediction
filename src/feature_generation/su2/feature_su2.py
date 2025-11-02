"""SU2特徴量生成のコアロジック

SU1出力を入力として、以下の二次特徴量を生成:
- ローリング統計（移動平均、標準偏差、最小値、最大値、中央値）
- 指数加重移動平均（EWMA）
- 遷移統計（差分、変化率、符号変化）
- 正規化（Z-score、Min-Max）

未来参照を防ぐため、t時点ではt-1までのデータのみを使用。
Fold境界で状態をリセット。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class SU2FeatureGenerator:
    """SU2特徴量生成器
    
    SU1出力から二次特徴量を生成。状態を保持し、fold境界でリセット可能。
    
    Attributes:
        config: 設定辞書（ローリングウィンドウ、EWMAパラメータなど）
        feature_columns: 生成された特徴量のカラムリスト
        _state: 内部状態（ローリング統計用の履歴など）
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """初期化
        
        Args:
            config: SU2設定（configs/feature_generation.yamlのsu2セクション）
        """
        self.config = config
        self.feature_columns: list[str] = []
        self._state: dict[str, Any] = {}
        self._reset_state()
    
    def _reset_state(self) -> None:
        """内部状態をリセット（fold境界で呼び出される）"""
        self._state = {
            'rolling_history': {},  # ローリング統計用の履歴
            'ewma_state': {},       # EWMA計算用の状態
            'last_values': {},      # 前回の値（遷移統計用）
        }
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        fold_id: int | None = None,
    ) -> pd.DataFrame:
        """学習データに対する特徴量生成
        
        Args:
            df: SU1出力のDataFrame
            fold_id: Fold ID（指定時は状態をリセット）
        
        Returns:
            SU2特徴量を追加したDataFrame
        """
        # Fold境界で状態リセット
        if fold_id is not None and self.config.get('fold', {}).get('reset_on_boundary', True):
            self._reset_state()
        
        # 入力のコピーを作成
        result = df.copy()
        
        # 元の数値カラムのみを対象とする（生成された特徴量に対しては再帰的に適用しない）
        original_numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        
        # 各特徴量タイプを生成
        result = self._add_rolling_features(result, original_numeric_cols)
        result = self._add_ewma_features(result, original_numeric_cols)
        result = self._add_transition_features(result, original_numeric_cols)
        result = self._add_normalization_features(result, original_numeric_cols)
        
        # データ型の最適化
        result = self._optimize_dtypes(result)
        
        return result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """推論データに対する特徴量生成（状態を使用）
        
        Args:
            df: SU1出力のDataFrame
        
        Returns:
            SU2特徴量を追加したDataFrame
        """
        # fit_transformと同じロジックだが、状態はリセットしない
        result = df.copy()
        
        # 元の数値カラムのみを対象とする
        original_numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        
        result = self._add_rolling_features(result, original_numeric_cols)
        result = self._add_ewma_features(result, original_numeric_cols)
        result = self._add_transition_features(result, original_numeric_cols)
        result = self._add_normalization_features(result, original_numeric_cols)
        result = self._optimize_dtypes(result)
        
        return result
    
    def _add_rolling_features(self, df: pd.DataFrame, target_cols: list[str] | None = None) -> pd.DataFrame:
        """ローリング統計特徴量を追加
        
        Args:
            df: 入力DataFrame
            target_cols: 対象カラムリスト（None時は全数値カラム）
        
        Returns:
            ローリング統計を追加したDataFrame
        """
        rolling_config = self.config.get('rolling', {})
        windows = rolling_config.get('windows', [7, 14, 30])
        functions = rolling_config.get('functions', ['mean', 'std', 'min', 'max'])
        
        # 数値カラムのみを対象
        if target_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = target_cols
        
        for col in numeric_cols:
            for window in windows:
                for func in functions:
                    feature_name = f"{col}_rolling_{window}_{func}"
                    
                    if func == 'mean':
                        # 未来参照防止: shift(1)でt-1までのデータを使用
                        df[feature_name] = df[col].shift(1).rolling(window=window, min_periods=1).mean()
                    elif func == 'std':
                        df[feature_name] = df[col].shift(1).rolling(window=window, min_periods=1).std()
                    elif func == 'min':
                        df[feature_name] = df[col].shift(1).rolling(window=window, min_periods=1).min()
                    elif func == 'max':
                        df[feature_name] = df[col].shift(1).rolling(window=window, min_periods=1).max()
                    elif func == 'median':
                        df[feature_name] = df[col].shift(1).rolling(window=window, min_periods=1).median()
                    
                    self.feature_columns.append(feature_name)
        
        return df
    
    def _add_ewma_features(self, df: pd.DataFrame, target_cols: list[str] | None = None) -> pd.DataFrame:
        """EWMA（指数加重移動平均）特徴量を追加
        
        Args:
            df: 入力DataFrame
            target_cols: 対象カラムリスト（None時は全数値カラム）
        
        Returns:
            EWMA特徴量を追加したDataFrame
        """
        ewma_config = self.config.get('ewma', {})
        spans = ewma_config.get('spans', [5, 10, 20])
        adjust = ewma_config.get('adjust', False)
        
        if target_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = target_cols
        
        for col in numeric_cols:
            for span in spans:
                feature_name = f"{col}_ewma_{span}"
                # 未来参照防止: shift(1)
                df[feature_name] = df[col].shift(1).ewm(span=span, adjust=adjust, min_periods=1).mean()
                self.feature_columns.append(feature_name)
        
        return df
    
    def _add_transition_features(self, df: pd.DataFrame, target_cols: list[str] | None = None) -> pd.DataFrame:
        """遷移統計特徴量を追加（前期との比較）
        
        Args:
            df: 入力DataFrame
            target_cols: 対象カラムリスト（None時は全数値カラム）
        
        Returns:
            遷移統計を追加したDataFrame
        """
        transition_config = self.config.get('transitions', {})
        lags = transition_config.get('lags', [1, 7, 30])
        methods = transition_config.get('methods', ['diff', 'pct_change'])
        
        if target_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = target_cols
        
        for col in numeric_cols:
            for lag in lags:
                # 差分
                if 'diff' in methods:
                    feature_name = f"{col}_diff_{lag}"
                    df[feature_name] = df[col] - df[col].shift(lag)
                    self.feature_columns.append(feature_name)
                
                # 変化率
                if 'pct_change' in methods:
                    feature_name = f"{col}_pct_change_{lag}"
                    df[feature_name] = df[col].pct_change(periods=lag, fill_method=None)
                    # inf/-inf を NaN に置き換え
                    df[feature_name] = df[feature_name].replace([np.inf, -np.inf], np.nan)
                    self.feature_columns.append(feature_name)
                
                # 符号変化
                if 'sign_change' in methods:
                    feature_name = f"{col}_sign_change_{lag}"
                    current_sign = np.sign(df[col])
                    past_sign = np.sign(df[col].shift(lag))
                    df[feature_name] = (current_sign != past_sign).astype(np.uint8)
                    self.feature_columns.append(feature_name)
        
        return df
    
    def _add_normalization_features(self, df: pd.DataFrame, target_cols: list[str] | None = None) -> pd.DataFrame:
        """正規化特徴量を追加
        
        Args:
            df: 入力DataFrame
            target_cols: 対象カラムリスト（None時は全数値カラム）
        
        Returns:
            正規化特徴量を追加したDataFrame
        """
        norm_config = self.config.get('normalization', {})
        methods = norm_config.get('methods', ['zscore'])
        rolling_window = norm_config.get('rolling_window', 30)
        
        if target_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = target_cols
        
        for col in numeric_cols:
            # Z-score正規化（ローリング平均・標準偏差使用）
            if 'zscore' in methods:
                feature_name = f"{col}_zscore"
                # 未来参照防止: shift(1)
                rolling_mean = df[col].shift(1).rolling(window=rolling_window, min_periods=1).mean()
                rolling_std = df[col].shift(1).rolling(window=rolling_window, min_periods=1).std()
                # ゼロ除算を防ぐ
                df[feature_name] = np.where(
                    rolling_std > 0,
                    (df[col] - rolling_mean) / rolling_std,
                    0.0
                )
                self.feature_columns.append(feature_name)
            
            # Min-Max正規化（過去データのみ使用）
            if 'minmax' in methods:
                feature_name = f"{col}_minmax"
                # 未来参照防止: expanding().min/max()を使用し、shift(1)
                expanding_min = df[col].shift(1).expanding(min_periods=1).min()
                expanding_max = df[col].shift(1).expanding(min_periods=1).max()
                # ゼロ除算を防ぐ
                df[feature_name] = np.where(
                    expanding_max > expanding_min,
                    (df[col] - expanding_min) / (expanding_max - expanding_min),
                    0.5  # 範囲が0の場合は中央値
                )
                self.feature_columns.append(feature_name)
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ型の最適化
        
        Args:
            df: 入力DataFrame
        
        Returns:
            データ型を最適化したDataFrame
        """
        dtypes_config = self.config.get('dtypes', {})
        flag_dtype = dtypes_config.get('flag', 'uint8')
        small_int_dtype = dtypes_config.get('small_int', 'int16')
        float_dtype = dtypes_config.get('float', 'float32')
        
        for col in df.columns:
            # フラグ列（0/1のみ）- NaNがある場合はスキップ
            if 'sign_change' in col:
                # NaNが無く、0/1のみの場合のみ変換
                has_nan = bool(df[col].isna().any())
                if not has_nan:
                    unique_vals = df[col].unique()
                    if set(unique_vals).issubset({0.0, 1.0}):
                        df[col] = df[col].astype(flag_dtype)
            # 小さな整数列
            elif df[col].dtype in ['int32', 'int64']:
                if df[col].min() >= np.iinfo(np.int16).min and df[col].max() <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(small_int_dtype)
            # 浮動小数点列
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype(float_dtype)
        
        return df
    
    def get_feature_columns(self) -> list[str]:
        """生成された特徴量のカラムリストを取得
        
        Returns:
            特徴量カラムのリスト
        """
        return self.feature_columns.copy()
    
    def get_state(self) -> dict[str, Any]:
        """内部状態を取得（推論用のシリアライズ）
        
        Returns:
            内部状態の辞書
        """
        return self._state.copy()
    
    def set_state(self, state: dict[str, Any]) -> None:
        """内部状態を設定（推論用のデシリアライズ）
        
        Args:
            state: 内部状態の辞書
        """
        self._state = state.copy()
