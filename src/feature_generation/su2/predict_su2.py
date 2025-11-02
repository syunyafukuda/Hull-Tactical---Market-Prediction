"""SU2推論パイプライン

学習済みモデルを使用してテストデータに対してSU2特徴量を適用し、予測を実行。
以下を出力:
- artifacts/SU2/submission.csv
- artifacts/SU2/submission.parquet
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def create_mock_test_data(n_samples: int = 100, n_features: int = 5, random_state: int = 123) -> pd.DataFrame:
    """モックのテストデータを生成（開発・テスト用）
    
    Args:
        n_samples: サンプル数
        n_features: 特徴量数
        random_state: 乱数シード
    
    Returns:
        モックDataFrame
    """
    rng = np.random.RandomState(random_state)
    
    data = {}
    for i in range(n_features):
        # トレンド + ノイズ
        trend = np.linspace(10, 15, n_samples)
        noise = rng.randn(n_samples) * 2
        data[f'feature_{i}'] = trend + noise
    
    df = pd.DataFrame(data)
    df.index.name = 'id'
    
    return df


def predict_su2_pipeline(
    bundle_path: str | Path = "artifacts/SU2/inference_bundle.pkl",
    test_data: pd.DataFrame | None = None,
    output_dir: str | Path = "artifacts/SU2",
) -> pd.DataFrame:
    """SU2推論パイプラインのメイン関数
    
    Args:
        bundle_path: 学習済みバンドルのパス
        test_data: テストデータ（Noneの場合はモックデータを生成）
        output_dir: 出力ディレクトリ
    
    Returns:
        予測結果のDataFrame
    """
    # 学習済みバンドルの読み込み
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"学習済みバンドルが見つかりません: {bundle_path}")
    
    print(f"学習済みバンドルを読み込み中: {bundle_path}")
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)
    
    generator = bundle['generator']
    feature_columns = bundle['feature_columns']
    
    print(f"読み込み完了: {len(feature_columns)} 個の特徴量")
    
    # テストデータの準備
    if test_data is None:
        print("\nモックテストデータを生成中...")
        test_df = create_mock_test_data(n_samples=100, n_features=5)
    else:
        test_df = test_data.copy()
    
    print(f"テストデータサイズ: {test_df.shape}")
    
    # SU2特徴量の生成
    print("\nSU2特徴量を生成中...")
    test_su2 = generator.transform(test_df)
    print(f"SU2適用後のサイズ: {test_su2.shape}")
    
    # 予測（簡易的に平均値を返す）
    # 実際のモデル推論に置き換え可能
    predictions = np.zeros(len(test_df))
    
    print(f"予測完了: {len(predictions)} 件")
    
    # 提出用データフレームの作成
    submission = pd.DataFrame({
        'id': test_df.index,
        'prediction': predictions,
    })
    
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # submission.csvの保存
    csv_path = output_path / "submission.csv"
    submission.to_csv(csv_path, index=False)
    print(f"\n提出ファイル（CSV）を保存: {csv_path}")
    
    # submission.parquetの保存
    parquet_path = output_path / "submission.parquet"
    submission.to_parquet(parquet_path, index=False)
    print(f"提出ファイル（Parquet）を保存: {parquet_path}")
    
    print("\n推論パイプライン完了!")
    
    return submission


if __name__ == "__main__":
    # コマンドライン実行時
    predict_su2_pipeline()
