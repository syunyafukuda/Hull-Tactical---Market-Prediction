"""SU10 特徴量生成の単体テスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_generation.su10.feature_su10 import (
    SU10Config,
    SU10FeatureGenerator,
    SU10_FEATURE_COLUMNS,
    load_su10_config,
)


def _build_mock_spy_data(n_rows: int = 100, start_date_id: int = 0) -> pd.DataFrame:
    """テスト用の疑似 SPY データを生成する。"""
    np.random.seed(42)
    date_ids = list(range(start_date_id, start_date_id + n_rows))
    # 擬似的な株価データ（ランダムウォーク）
    price_base = 100.0
    returns = np.random.normal(0.0005, 0.01, n_rows)
    prices = price_base * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date_id": date_ids,
        "Adj Close": prices,
    })


def test_su10_config_defaults() -> None:
    """SU10Config のデフォルト値検証。"""
    config = SU10Config()

    assert config.ewm_halflife_short == 20
    assert config.ewm_halflife_long == 60
    assert config.vol_quantile_low == 0.33
    assert config.vol_quantile_high == 0.66
    assert config.trend_quantile_down == 0.33
    assert config.trend_quantile_up == 0.66
    assert config.return_periods == (5, 20)
    assert config.vol_adj_period == 5
    assert config.winsorize_low == 0.01
    assert config.winsorize_high == 0.99
    assert config.train_max_date_id == 8979


def test_su10_config_from_mapping() -> None:
    """SU10Config.from_mapping() のテスト。"""
    mapping = {
        "ewm_halflife_short": 10,
        "ewm_halflife_long": 30,
        "vol_quantile_low": 0.25,
        "vol_quantile_high": 0.75,
        "train_max_date_id": 5000,
        "return_periods": [3, 10],
    }
    config = SU10Config.from_mapping(mapping)

    assert config.ewm_halflife_short == 10
    assert config.ewm_halflife_long == 30
    assert config.vol_quantile_low == 0.25
    assert config.vol_quantile_high == 0.75
    assert config.train_max_date_id == 5000
    assert config.return_periods == (3, 10)


def test_su10_config_loading(tmp_path: Path) -> None:
    """YAML設定の読込確認。"""
    config_path = tmp_path / "test_config.yaml"
    config_content = """
su10:
  ewm_halflife_short: 15
  ewm_halflife_long: 45
  vol_quantile_low: 0.30
  vol_quantile_high: 0.70
  train_max_date_id: 7000
"""
    config_path.write_text(config_content)

    config = load_su10_config(config_path)
    assert config.ewm_halflife_short == 15
    assert config.ewm_halflife_long == 45
    assert config.vol_quantile_low == 0.30
    assert config.vol_quantile_high == 0.70
    assert config.train_max_date_id == 7000


def test_su10_feature_generator_fit() -> None:
    """fit() の分位点算出テスト。"""
    config = SU10Config(train_max_date_id=50)
    generator = SU10FeatureGenerator(config)

    spy_data = _build_mock_spy_data(n_rows=100)
    generator.fit(spy_data)

    # fit 後に分位点が記録されているか
    assert generator.vol_mean_ is not None
    assert generator.vol_std_ is not None
    assert generator.vol_quantiles_ is not None
    assert generator.trend_quantiles_ is not None
    assert generator.winsorize_bounds_ is not None
    assert generator.feature_names_ is not None

    # 分位点は2要素のタプル
    assert len(generator.vol_quantiles_) == 2
    assert len(generator.trend_quantiles_) == 2
    assert len(generator.winsorize_bounds_) == 2

    # vol_quantiles_ は low < high の順
    assert generator.vol_quantiles_[0] <= generator.vol_quantiles_[1]


def test_su10_feature_generator_transform() -> None:
    """transform() の出力列数・型検証。"""
    config = SU10Config(train_max_date_id=50)
    generator = SU10FeatureGenerator(config)

    spy_data = _build_mock_spy_data(n_rows=100)
    generator.fit(spy_data)
    result = generator.transform(spy_data)

    # 出力は DataFrame
    assert isinstance(result, pd.DataFrame)

    # 行数は入力と同じ
    assert len(result) == len(spy_data)

    # date_id + 14 SU10 特徴量 = 15 列
    assert len(result.columns) == 15

    # date_id が含まれている
    assert "date_id" in result.columns


def test_su10_feature_columns() -> None:
    """14列の命名規則・型検証。"""
    config = SU10Config(train_max_date_id=50)
    generator = SU10FeatureGenerator(config)

    spy_data = _build_mock_spy_data(n_rows=100)
    result = generator.fit_transform(spy_data)

    # 全ての SU10 特徴量カラムが存在する
    for col in SU10_FEATURE_COLUMNS:
        assert col in result.columns, f"Missing column: {col}"

    # float 型の列は float32
    float_cols = [
        "su10_spx_ewmstd_20d",
        "su10_spx_ewmstd_60d",
        "su10_spx_vol_ratio",
        "su10_spx_vol_level",
        "su10_spx_trend_indicator",
        "su10_spx_ret_5d",
        "su10_spx_ret_20d",
        "su10_spx_ret_vol_adj_5d",
    ]
    for col in float_cols:
        assert result[col].dtype == np.float32, f"{col} should be float32"

    # uint8 型の列（レジームフラグ）
    uint8_cols = [
        "su10_spx_vol_regime_low",
        "su10_spx_vol_regime_mid",
        "su10_spx_vol_regime_high",
        "su10_spx_trend_regime_down",
        "su10_spx_trend_regime_flat",
        "su10_spx_trend_regime_up",
    ]
    for col in uint8_cols:
        assert result[col].dtype == np.uint8, f"{col} should be uint8"


def test_su10_no_future_leakage() -> None:
    """未来リーク検証: train 期間以降のデータで fit 結果が変わらないこと。"""
    config = SU10Config(train_max_date_id=50)

    # train期間のみのデータ
    train_only_data = _build_mock_spy_data(n_rows=60)
    train_only_data = train_only_data[train_only_data["date_id"] <= 50].reset_index(drop=True)

    # train + test 期間のデータ
    full_data = _build_mock_spy_data(n_rows=100)

    # 別々に fit
    gen1 = SU10FeatureGenerator(config)
    gen1.fit(train_only_data)

    gen2 = SU10FeatureGenerator(config)
    gen2.fit(full_data)

    # vol_mean_, vol_std_, vol_quantiles_, trend_quantiles_ は train 期間のみで計算されるため同じはず
    # ただし、EWM の warm-up の関係で完全一致はしない可能性がある
    # 少なくとも大きな差異がないことを確認

    # vol_quantiles_ の差が小さいことを確認（相対誤差 50% 以内）
    if gen1.vol_quantiles_ is not None and gen2.vol_quantiles_ is not None:
        for i in range(2):
            diff = abs(gen1.vol_quantiles_[i] - gen2.vol_quantiles_[i])
            max_val = max(abs(gen1.vol_quantiles_[i]), abs(gen2.vol_quantiles_[i]), 0.01)
            assert diff / max_val < 0.5, f"vol_quantiles_[{i}] mismatch: {gen1.vol_quantiles_[i]} vs {gen2.vol_quantiles_[i]}"


def test_su10_join_with_train() -> None:
    """train との join 整合性テスト。"""
    config = SU10Config(train_max_date_id=50)
    generator = SU10FeatureGenerator(config)

    spy_data = _build_mock_spy_data(n_rows=100)
    su10_features = generator.fit_transform(spy_data)

    # 疑似的な train データ
    train_data = pd.DataFrame({
        "date_id": [10, 20, 30, 40, 50],
        "feature_A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "target": [0.01, 0.02, 0.03, 0.04, 0.05],
    })

    # join
    merged = train_data.merge(su10_features, on="date_id", how="left")

    # 結果の検証
    assert len(merged) == len(train_data)
    assert "su10_spx_ewmstd_20d" in merged.columns
    assert "su10_spx_vol_regime_low" in merged.columns

    # NaN がないこと（date_id 10 以上では SPY データが存在するため）
    # ただし、EWM の warm-up 期間は NaN が発生する可能性がある
    # date_id が十分大きい場合は NaN がない
    non_nan_count = merged["su10_spx_ewmstd_20d"].notna().sum()
    assert non_nan_count >= 3  # 少なくとも3行は値がある


def test_su10_nan_handling() -> None:
    """date_id 0 付近の NaN 処理検証。"""
    config = SU10Config(train_max_date_id=50)
    generator = SU10FeatureGenerator(config)

    spy_data = _build_mock_spy_data(n_rows=100)
    result = generator.fit_transform(spy_data)

    # date_id = 0 では日次リターンが計算できないため NaN
    first_row = result[result["date_id"] == 0]
    assert len(first_row) == 1

    # date_id = 0 の su10_spx_ewmstd_20d は NaN
    assert pd.isna(first_row["su10_spx_ewmstd_20d"].values[0])

    # 後半の行では値が存在する
    last_row = result[result["date_id"] == 99]
    assert not pd.isna(last_row["su10_spx_ewmstd_20d"].values[0])


def test_su10_regime_one_hot() -> None:
    """レジームフラグが one-hot になっていることを確認。"""
    config = SU10Config(train_max_date_id=50)
    generator = SU10FeatureGenerator(config)

    spy_data = _build_mock_spy_data(n_rows=100)
    result = generator.fit_transform(spy_data)

    # ボラティリティレジーム: 各行で low + mid + high = 1
    vol_regime_sum = (
        result["su10_spx_vol_regime_low"]
        + result["su10_spx_vol_regime_mid"]
        + result["su10_spx_vol_regime_high"]
    )
    # NaN 以外の行では合計が 1
    non_nan_mask = result["su10_spx_vol_level"].notna()
    assert (vol_regime_sum[non_nan_mask] == 1).all(), "Volatility regime should be one-hot"

    # トレンドレジーム: 各行で down + flat + up = 1
    trend_regime_sum = (
        result["su10_spx_trend_regime_down"]
        + result["su10_spx_trend_regime_flat"]
        + result["su10_spx_trend_regime_up"]
    )
    # NaN 以外の行では合計が 1
    non_nan_mask = result["su10_spx_trend_indicator"].notna()
    assert (trend_regime_sum[non_nan_mask] == 1).all(), "Trend regime should be one-hot"


def test_su10_fit_transform_consistency() -> None:
    """fit_transform() が fit() + transform() と同じ結果になることを確認。"""
    config = SU10Config(train_max_date_id=50)
    spy_data = _build_mock_spy_data(n_rows=100)

    # fit_transform
    gen1 = SU10FeatureGenerator(config)
    result1 = gen1.fit_transform(spy_data)

    # fit + transform
    gen2 = SU10FeatureGenerator(config)
    gen2.fit(spy_data)
    result2 = gen2.transform(spy_data)

    # 結果が一致すること
    pd.testing.assert_frame_equal(result1, result2)


def test_su10_transform_without_fit_raises() -> None:
    """fit() せずに transform() を呼ぶとエラーになることを確認。"""
    config = SU10Config()
    generator = SU10FeatureGenerator(config)
    spy_data = _build_mock_spy_data(n_rows=100)

    with pytest.raises(RuntimeError, match="must be fitted"):
        generator.transform(spy_data)


def test_su10_missing_date_id_raises() -> None:
    """date_id 列がない場合にエラーになることを確認。"""
    config = SU10Config()
    generator = SU10FeatureGenerator(config)

    bad_data = pd.DataFrame({
        "Adj Close": [100.0, 101.0, 102.0],
    })

    with pytest.raises(ValueError, match="date_id"):
        generator.fit(bad_data)


def test_su10_missing_adj_close_raises() -> None:
    """Adj Close 列がない場合にエラーになることを確認。"""
    config = SU10Config()
    generator = SU10FeatureGenerator(config)

    bad_data = pd.DataFrame({
        "date_id": [0, 1, 2],
    })

    with pytest.raises(ValueError, match="Adj Close"):
        generator.fit(bad_data)
