"""SU2学習パイプライン

SU1出力を読み込み、SU2特徴量を生成してモデルを学習。
以下を出力:
- artifacts/SU2/feature_list.json
- artifacts/SU2/inference_bundle.pkl
- artifacts/SU2/model_meta.json
- artifacts/SU2/cv_fold_logs.csv
- artifacts/SU2/oof_predictions.csv
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from .feature_su2 import SU2FeatureGenerator


def load_config(config_path: str | Path = "configs/feature_generation.yaml") -> dict[str, Any]:
    """設定ファイルを読み込む
    
    Args:
        config_path: 設定ファイルのパス
    
    Returns:
        設定辞書
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_mock_su1_data(n_samples: int = 100, n_features: int = 3, random_state: int = 42) -> pd.DataFrame:
    """モックのSU1データを生成（開発・テスト用）
    
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
        trend = np.linspace(0, 10, n_samples)
        noise = rng.randn(n_samples) * 2
        data[f'feature_{i}'] = trend + noise
    
    # ターゲット変数（特徴量の線形結合 + ノイズ）
    data['target'] = sum(data[f'feature_{i}'] for i in range(n_features)) / n_features + rng.randn(n_samples)
    
    df = pd.DataFrame(data)
    df.index.name = 'id'
    
    return df


def train_su2_pipeline(
    config: dict[str, Any] | None = None,
    input_data: pd.DataFrame | None = None,
    output_dir: str | Path = "artifacts/SU2",
) -> None:
    """SU2学習パイプラインのメイン関数
    
    Args:
        config: SU2設定辞書（Noneの場合はデフォルト設定を使用）
        input_data: SU1出力データ（Noneの場合はモックデータを生成）
        output_dir: 出力ディレクトリ
    """
    # 設定の読み込み
    if config is None:
        try:
            full_config = load_config()
            config = full_config.get('su2', {})
        except FileNotFoundError:
            # 設定ファイルが無い場合はデフォルト値（最小構成）
            config = {
                'rolling': {'windows': [3], 'functions': ['mean']},
                'ewma': {'spans': [3], 'adjust': False},
                'transitions': {'lags': [1], 'methods': ['diff']},
                'normalization': {'methods': ['zscore'], 'rolling_window': 5},
                'dtypes': {'flag': 'uint8', 'small_int': 'int16', 'float': 'float32'},
                'fold': {'reset_on_boundary': True, 'validate_no_leakage': True},
            }
    
    # 入力データの準備
    if input_data is None:
        print("モックSU1データを生成中...")
        df = create_mock_su1_data(n_samples=100, n_features=3)
    else:
        df = input_data.copy()
    
    print(f"入力データサイズ: {df.shape}")
    
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ターゲット変数の分離
    if 'target' in df.columns:
        target = df['target'].values
        feature_df = df.drop(columns=['target'])
    else:
        # ターゲットが無い場合はダミーを作成
        target = np.zeros(len(df))
        feature_df = df.copy()
    
    # SU2特徴量生成器の初期化
    if config is None:
        config = {}
    generator = SU2FeatureGenerator(config)
    
    # クロスバリデーションの設定
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # OOF予測とログの保存用
    oof_predictions = np.zeros(len(df))
    fold_logs = []
    
    print(f"\n{n_folds}-Fold クロスバリデーション開始...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(feature_df)):
        print(f"\n=== Fold {fold_idx + 1}/{n_folds} ===")
        
        # データ分割
        X_train = feature_df.iloc[train_idx]
        X_val = feature_df.iloc[val_idx]
        y_train = target[train_idx]
        y_val = target[val_idx]
        
        # SU2特徴量生成（状態リセット付き）
        print("SU2特徴量を生成中...")
        X_train_su2 = generator.fit_transform(X_train, fold_id=fold_idx)
        X_val_su2 = generator.transform(X_val)
        
        print(f"学習データ: {X_train_su2.shape}, 検証データ: {X_val_su2.shape}")
        
        # 簡易モデル学習（平均値予測）
        # 実際のモデルに置き換え可能
        train_mean = y_train.mean()
        val_predictions = np.full(len(y_val), train_mean)
        
        # OOF予測を保存
        oof_predictions[val_idx] = val_predictions
        
        # 評価
        rmse = np.sqrt(np.mean((y_val - val_predictions) ** 2))  # type: ignore[operator]
        mae = np.mean(np.abs(y_val - val_predictions))  # type: ignore[operator]
        
        print(f"検証 RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # ログ記録
        fold_logs.append({
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'val_rmse': rmse,
            'val_mae': mae,
        })
    
    # 全体のOOF評価
    overall_rmse = float(np.sqrt(np.mean((np.asarray(target) - oof_predictions) ** 2)))
    overall_mae = float(np.mean(np.abs(np.asarray(target) - oof_predictions)))
    print(f"\n全体 OOF RMSE: {overall_rmse:.4f}, MAE: {overall_mae:.4f}")
    
    # feature_list.jsonの保存
    feature_list = generator.get_feature_columns()
    feature_list_path = output_path / "feature_list.json"
    with open(feature_list_path, 'w', encoding='utf-8') as f:
        json.dump(feature_list, f, indent=2, ensure_ascii=False)
    print(f"\n特徴量リストを保存: {feature_list_path}")
    print(f"生成された特徴量数: {len(feature_list)}")
    
    # inference_bundle.pklの保存
    inference_bundle = {
        'generator': generator,
        'config': config,
        'feature_columns': feature_list,
    }
    bundle_path = output_path / "inference_bundle.pkl"
    with open(bundle_path, 'wb') as f:
        pickle.dump(inference_bundle, f)
    print(f"推論バンドルを保存: {bundle_path}")
    
    # model_meta.jsonの保存
    model_meta = {
        'su2_config': config,
        'feature_columns': feature_list,
        'n_features': len(feature_list),
        'input_shape': list(feature_df.shape),
        'cv_folds': n_folds,
        'oof_rmse': float(overall_rmse),
        'oof_mae': float(overall_mae),
    }
    meta_path = output_path / "model_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(model_meta, f, indent=2, ensure_ascii=False)
    print(f"モデルメタデータを保存: {meta_path}")
    
    # cv_fold_logs.csvの保存
    fold_logs_df = pd.DataFrame(fold_logs)
    logs_path = output_path / "cv_fold_logs.csv"
    fold_logs_df.to_csv(logs_path, index=False)
    print(f"CVログを保存: {logs_path}")
    
    # oof_predictions.csvの保存
    oof_df = pd.DataFrame({
        'id': df.index,
        'target': target,
        'prediction': oof_predictions,
    })
    oof_path = output_path / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF予測を保存: {oof_path}")
    
    print("\n学習パイプライン完了!")


if __name__ == "__main__":
    # コマンドライン実行時
    train_su2_pipeline()
