#!/usr/bin/env python3
"""
SU10 External Regime CSV 生成スクリプト

SPY Historical Data から外部レジーム特徴量を計算し、
以下のCSVを生成する：
- su10_external_regime.csv: date_id → SU10特徴のマッピングテーブル
- train_with_su10.csv: train.csv + SU10特徴（ローカル学習用）
- test_with_su10.csv: test.csv + SU10特徴（ローカルテスト用）

Usage:
    uv run python scripts/su10/build_su10_external_regime.py

Output:
    data/su10/su10_external_regime.csv
    data/su10/train_with_su10.csv
    data/su10/test_with_su10.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SU10Config:
    """SU10 特徴量生成設定"""

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


def build_su10_features(
    spy_df: pd.DataFrame,
    config: SU10Config,
) -> pd.DataFrame:
    """
    SPY データから SU10 特徴量を生成する

    Parameters
    ----------
    spy_df : pd.DataFrame
        SPY Historical Data (date_id, Adj Close 等を含む)
    config : SU10Config
        特徴量生成設定

    Returns
    -------
    pd.DataFrame
        date_id と SU10 特徴量のみを含む DataFrame
    """
    df = spy_df.copy()

    # 日次リターン
    df["ret"] = df["Adj Close"].pct_change()

    # ========================================
    # ボラティリティ指標
    # ========================================
    df["su10_spx_ewmstd_20d"] = (
        df["ret"].ewm(halflife=config.ewm_halflife_short, adjust=False).std()
    )
    df["su10_spx_ewmstd_60d"] = (
        df["ret"].ewm(halflife=config.ewm_halflife_long, adjust=False).std()
    )
    df["su10_spx_vol_ratio"] = df["su10_spx_ewmstd_20d"] / (
        df["su10_spx_ewmstd_60d"] + 1e-8
    )

    # ボラ水準（標準化）- train 期間のみで mean/std を算出
    train_mask = df["date_id"] <= config.train_max_date_id
    vol_train = df.loc[train_mask, "su10_spx_ewmstd_20d"]
    vol_mean = vol_train.mean()
    vol_std = vol_train.std()
    df["su10_spx_vol_level"] = (df["su10_spx_ewmstd_20d"] - vol_mean) / (vol_std + 1e-8)

    # ========================================
    # ボラティリティレジーム（3区分）
    # ========================================
    vol_level_train = df.loc[train_mask, "su10_spx_vol_level"]
    q_low = vol_level_train.quantile(config.vol_quantile_low)
    q_high = vol_level_train.quantile(config.vol_quantile_high)

    df["su10_spx_vol_regime_low"] = (df["su10_spx_vol_level"] <= q_low).astype(np.uint8)
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
        df["Adj Close"].ewm(halflife=config.trend_ewm_halflife_short, adjust=False).mean()
    )
    df["ma_long"] = (
        df["Adj Close"].ewm(halflife=config.trend_ewm_halflife_long, adjust=False).mean()
    )
    df["su10_spx_trend_indicator"] = df["ma_short"] - df["ma_long"]

    # ========================================
    # トレンドレジーム（3区分）
    # ========================================
    trend_train = df.loc[train_mask, "su10_spx_trend_indicator"]
    tau_down = trend_train.quantile(config.trend_quantile_down)
    tau_up = trend_train.quantile(config.trend_quantile_up)

    df["su10_spx_trend_regime_down"] = (
        df["su10_spx_trend_indicator"] <= tau_down
    ).astype(np.uint8)
    df["su10_spx_trend_regime_flat"] = (
        (df["su10_spx_trend_indicator"] > tau_down)
        & (df["su10_spx_trend_indicator"] <= tau_up)
    ).astype(np.uint8)
    df["su10_spx_trend_regime_up"] = (df["su10_spx_trend_indicator"] > tau_up).astype(
        np.uint8
    )

    # ========================================
    # 期間リターン
    # ========================================
    for period in config.return_periods:
        df[f"su10_spx_ret_{period}d"] = df["Adj Close"].pct_change(periods=period)

    # ========================================
    # ボラ調整リターン
    # ========================================
    ret_col = f"su10_spx_ret_{config.vol_adj_period}d"
    denom = 1 + df["su10_spx_ewmstd_20d"].clip(lower=1e-4)
    df["su10_spx_ret_vol_adj_5d"] = df[ret_col] / denom

    # Winsorize（train 期間の分位点でクリップ）
    ret_vol_adj_train = df.loc[train_mask, "su10_spx_ret_vol_adj_5d"].dropna()
    clip_low = ret_vol_adj_train.quantile(config.winsorize_low)
    clip_high = ret_vol_adj_train.quantile(config.winsorize_high)
    df["su10_spx_ret_vol_adj_5d"] = df["su10_spx_ret_vol_adj_5d"].clip(
        lower=clip_low, upper=clip_high
    )

    # ========================================
    # SU10 列のみ抽出
    # ========================================
    su10_cols = ["date_id"] + [c for c in df.columns if c.startswith("su10_")]
    result = df[su10_cols].copy()

    # float32 に変換（メモリ削減）
    for col in result.columns:
        if result[col].dtype == np.float64:
            result[col] = result[col].astype(np.float32)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SU10 External Regime CSV")
    parser.add_argument(
        "--spy-path",
        type=Path,
        default=Path("data/histolical/spy-historical.csv"),
        help="Path to SPY historical data",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Path to train.csv",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/raw/test.csv"),
        help="Path to test.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/su10"),
        help="Output directory for generated CSVs",
    )
    args = parser.parse_args()

    # 出力ディレクトリ作成
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SU10 External Regime CSV 生成")
    print("=" * 60)

    # ========================================
    # Step 1: SPY データ読み込み
    # ========================================
    print(f"\n[1/4] SPY データ読み込み: {args.spy_path}")
    spy = pd.read_csv(args.spy_path)
    print(f"  - 行数: {len(spy)}")
    print(f"  - date_id 範囲: {spy['date_id'].min()} 〜 {spy['date_id'].max()}")

    # ========================================
    # Step 2: SU10 特徴量生成
    # ========================================
    print("\n[2/4] SU10 特徴量生成")
    config = SU10Config()
    su10 = build_su10_features(spy, config)

    su10_path = args.out_dir / "su10_external_regime.csv"
    su10.to_csv(su10_path, index=False)
    print(f"  - 出力: {su10_path}")
    print(f"  - 行数: {len(su10)}")
    print(f"  - 列数: {len(su10.columns)}")
    print(f"  - 列名: {list(su10.columns)}")

    # ========================================
    # Step 3: train_with_su10.csv 生成
    # ========================================
    print(f"\n[3/4] train_with_su10.csv 生成: {args.train_path}")
    train = pd.read_csv(args.train_path)
    train_with_su10 = train.merge(su10, on="date_id", how="left")

    train_out_path = args.out_dir / "train_with_su10.csv"
    train_with_su10.to_csv(train_out_path, index=False)
    print(f"  - 出力: {train_out_path}")
    print(f"  - 行数: {len(train_with_su10)}")
    print(f"  - 列数: {len(train_with_su10.columns)}")
    print(f"  - SU10 欠損行数 (date_id < 780): {train_with_su10['su10_spx_ret_5d'].isna().sum()}")

    # ========================================
    # Step 4: test_with_su10.csv 生成
    # ========================================
    print(f"\n[4/4] test_with_su10.csv 生成: {args.test_path}")
    test = pd.read_csv(args.test_path)
    test_with_su10 = test.merge(su10, on="date_id", how="left")

    test_out_path = args.out_dir / "test_with_su10.csv"
    test_with_su10.to_csv(test_out_path, index=False)
    print(f"  - 出力: {test_out_path}")
    print(f"  - 行数: {len(test_with_su10)}")
    print(f"  - 列数: {len(test_with_su10.columns)}")
    print(f"  - SU10 欠損行数: {test_with_su10['su10_spx_ret_5d'].isna().sum()}")

    # ========================================
    # 完了サマリ
    # ========================================
    print("\n" + "=" * 60)
    print("完了サマリ")
    print("=" * 60)
    print(f"  su10_external_regime.csv : {su10_path} ({len(su10)} 行, {len(su10.columns)} 列)")
    print(f"  train_with_su10.csv      : {train_out_path} ({len(train_with_su10)} 行)")
    print(f"  test_with_su10.csv       : {test_out_path} ({len(test_with_su10)} 行)")
    print("\nKaggle Dataset にアップロードするファイル:")
    print(f"  ✅ {su10_path}")
    print(f"  ❌ {train_out_path} (ローカル学習用、アップロード不要)")
    print(f"  ❌ {test_out_path} (ローカルテスト用、アップロード不要)")


if __name__ == "__main__":
    main()
