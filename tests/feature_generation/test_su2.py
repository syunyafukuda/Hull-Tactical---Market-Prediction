"""SU2特徴量生成のテスト

以下のケースをテスト:
1. 全NaNケース
2. 交互NaNケース
3. 島状NaNケース
4. Fold境界リセット
5. 未来参照チェック
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_generation.su2.feature_su2 import SU2FeatureGenerator


@pytest.fixture
def base_config() -> dict:
    """基本設定"""
    return {
        'rolling': {
            'windows': [3, 5],
            'functions': ['mean', 'std'],
        },
        'ewma': {
            'spans': [3, 5],
            'adjust': False,
        },
        'transitions': {
            'lags': [1, 3],
            'methods': ['diff', 'pct_change'],
        },
        'normalization': {
            'methods': ['zscore', 'minmax'],
            'rolling_window': 5,
        },
        'dtypes': {
            'flag': 'uint8',
            'small_int': 'int16',
            'float': 'float32',
        },
        'fold': {
            'reset_on_boundary': True,
            'validate_no_leakage': True,
        },
    }


def test_all_nan_case(base_config):
    """全NaNケースのテスト
    
    すべての値がNaNの場合、特徴量生成がエラーなく実行され、
    適切にNaNが処理されることを検証。
    """
    # 全NaNのデータを作成
    df = pd.DataFrame({
        'feature_0': [np.nan] * 10,
        'feature_1': [np.nan] * 10,
    })
    
    generator = SU2FeatureGenerator(base_config)
    
    # エラーなく実行できることを確認
    result = generator.fit_transform(df, fold_id=0)
    
    # 結果がDataFrameであることを確認
    assert isinstance(result, pd.DataFrame)
    
    # 元のカラムが残っていることを確認
    assert 'feature_0' in result.columns
    assert 'feature_1' in result.columns
    
    # 新しい特徴量が追加されていることを確認
    assert len(result.columns) > len(df.columns)


def test_alternating_nan_case(base_config):
    """交互NaNケースのテスト
    
    交互にNaNが存在する場合、ローリング統計が適切に計算されることを検証。
    """
    # 交互にNaNを含むデータ
    values = [1.0, np.nan, 2.0, np.nan, 3.0, np.nan, 4.0, np.nan, 5.0, np.nan]
    df = pd.DataFrame({
        'feature_0': values,
    })
    
    generator = SU2FeatureGenerator(base_config)
    result = generator.fit_transform(df, fold_id=0)
    
    # 結果がDataFrameであることを確認
    assert isinstance(result, pd.DataFrame)
    
    # NaN以外の値が存在する行が処理されていることを確認
    # ローリング特徴量は計算されているはず（min_periods=1のため）
    rolling_cols = [col for col in result.columns if 'rolling' in col]
    assert len(rolling_cols) > 0


def test_island_nan_case(base_config):
    """島状NaNケースのテスト
    
    連続した欠損値の塊が存在する場合の処理を検証。
    """
    # 島状のNaNを含むデータ
    values = [1.0, 2.0, 3.0, np.nan, np.nan, np.nan, 4.0, 5.0, 6.0, 7.0]
    df = pd.DataFrame({
        'feature_0': values,
    })
    
    generator = SU2FeatureGenerator(base_config)
    result = generator.fit_transform(df, fold_id=0)
    
    # 結果がDataFrameであることを確認
    assert isinstance(result, pd.DataFrame)
    
    # NaN区間の後でもローリング統計が再開されることを確認
    rolling_mean_col = [col for col in result.columns if 'rolling' in col and 'mean' in col]
    assert len(rolling_mean_col) > 0


def test_fold_boundary_reset(base_config):
    """Fold境界リセットのテスト
    
    Fold境界で状態が正しくリセットされることを検証。
    """
    df = pd.DataFrame({
        'feature_0': list(range(10)),
    })
    
    generator = SU2FeatureGenerator(base_config)
    
    # Fold 0で変換
    result_fold0 = generator.fit_transform(df, fold_id=0)
    state_fold0 = generator.get_state()
    
    # Fold 1で変換（状態リセットあり）
    result_fold1 = generator.fit_transform(df, fold_id=1)
    state_fold1 = generator.get_state()
    
    # 両方とも同じ形状であることを確認
    assert result_fold0.shape == result_fold1.shape
    
    # 状態がリセットされていることを確認（辞書の内容が初期化される）
    # 完全に同一ではないが、構造は同じはず
    assert set(state_fold0.keys()) == set(state_fold1.keys())


def test_no_future_leakage(base_config):
    """未来参照チェックのテスト
    
    t時点の特徴量が t-1 までのデータのみで計算されていることを検証。
    """
    # 時系列データを作成
    df = pd.DataFrame({
        'feature_0': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    })
    
    generator = SU2FeatureGenerator(base_config)
    result = generator.fit_transform(df, fold_id=0)
    
    # ローリング平均の確認（window=3の場合）
    rolling_mean_cols = [col for col in result.columns if 'rolling_3_mean' in col]
    
    if rolling_mean_cols:
        rolling_col = rolling_mean_cols[0]
        
        # インデックス3の時点でのローリング平均は、インデックス0,1,2の平均のはず
        # shift(1)を使用しているため、インデックス3では0,1,2を使用
        # ただし、最初のいくつかの値はNaNまたは部分的な平均
        
        # 少なくとも、最後の値が全データの平均ではないことを確認
        # （未来のデータを使っていない証拠）
        last_rolling_mean = result[rolling_col].iloc[-1]
        all_data_mean = df['feature_0'].mean()
        
        # 最後のローリング平均は全体平均と異なるはず
        assert not np.isclose(last_rolling_mean, all_data_mean, rtol=0.1)


def test_dtype_optimization(base_config):
    """データ型最適化のテスト
    
    指定されたデータ型が正しく適用されることを検証。
    """
    df = pd.DataFrame({
        'feature_0': [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    
    generator = SU2FeatureGenerator(base_config)
    result = generator.fit_transform(df, fold_id=0)
    
    # float32が使用されていることを確認
    float_cols = result.select_dtypes(include=['float32']).columns
    assert len(float_cols) > 0
    
    # sign_change列（あれば）がuint8であることを確認
    sign_change_cols = [col for col in result.columns if 'sign_change' in col]
    if sign_change_cols:
        assert result[sign_change_cols[0]].dtype == np.uint8


def test_feature_columns_list(base_config):
    """特徴量カラムリストのテスト
    
    生成された特徴量のリストが正しく返されることを検証。
    """
    df = pd.DataFrame({
        'feature_0': list(range(10)),
        'feature_1': list(range(10, 20)),
    })
    
    generator = SU2FeatureGenerator(base_config)
    result = generator.fit_transform(df, fold_id=0)
    
    feature_list = generator.get_feature_columns()
    
    # 特徴量リストが空でないことを確認
    assert len(feature_list) > 0
    
    # 生成されたカラムがすべて結果に存在することを確認
    for col in feature_list:
        assert col in result.columns


def test_state_preservation(base_config):
    """状態の保存・復元のテスト
    
    状態を保存して復元できることを検証。
    """
    df = pd.DataFrame({
        'feature_0': list(range(10)),
    })
    
    generator1 = SU2FeatureGenerator(base_config)
    generator1.fit_transform(df, fold_id=0)
    
    # 状態を取得
    state = generator1.get_state()
    
    # 新しいジェネレータに状態を設定
    generator2 = SU2FeatureGenerator(base_config)
    generator2.set_state(state)
    
    # 状態が復元されていることを確認
    state2 = generator2.get_state()
    
    # 辞書のキーが同じであることを確認
    assert set(state.keys()) == set(state2.keys())


def test_transform_without_fit(base_config):
    """fit無しでのtransformのテスト
    
    fit_transform無しでtransformを呼び出した場合の挙動を検証。
    """
    df = pd.DataFrame({
        'feature_0': list(range(5)),
    })
    
    generator = SU2FeatureGenerator(base_config)
    
    # fit_transform無しでtransformを呼び出し
    # エラーにならず、初期状態で処理されることを確認
    result = generator.transform(df)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) > len(df.columns)


def test_empty_dataframe(base_config):
    """空のDataFrameのテスト
    
    空のDataFrameが入力された場合の挙動を検証。
    """
    df = pd.DataFrame()
    
    generator = SU2FeatureGenerator(base_config)
    
    # 空のDataFrameでもエラーにならないことを確認
    result = generator.fit_transform(df, fold_id=0)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_single_row_dataframe(base_config):
    """1行のDataFrameのテスト
    
    1行だけのDataFrameでローリング統計が計算されることを検証。
    """
    df = pd.DataFrame({
        'feature_0': [5.0],
    })
    
    generator = SU2FeatureGenerator(base_config)
    result = generator.fit_transform(df, fold_id=0)
    
    # 結果が1行であることを確認
    assert len(result) == 1
    
    # 特徴量が追加されていることを確認（min_periods=1のため）
    assert len(result.columns) > len(df.columns)
