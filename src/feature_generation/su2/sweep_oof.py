"""SU2ハイパーパラメータスイープ

configs/feature_generation.yamlに記載された各ポリシー候補をスイープし、
最適なパラメータを特定。

出力:
- results/ablation/SU2/sweep_result_{timestamp}.json (各スイープ結果)
- results/ablation/SU2/sweep_summary.csv (全結果のサマリ)
"""

from __future__ import annotations

import json
from datetime import datetime
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


def create_mock_data(n_samples: int = 1000, n_features: int = 5, random_state: int = 42) -> pd.DataFrame:
    """モックデータ生成
    
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
        trend = np.linspace(0, 10, n_samples)
        noise = rng.randn(n_samples) * 2
        data[f'feature_{i}'] = trend + noise
    
    data['target'] = sum(data[f'feature_{i}'] for i in range(n_features)) / n_features + rng.randn(n_samples)
    
    df = pd.DataFrame(data)
    df.index.name = 'id'
    
    return df


def evaluate_config(
    config: dict[str, Any],
    df: pd.DataFrame,
    n_folds: int = 5,
) -> dict[str, Any]:
    """指定された設定でOOF評価を実行
    
    Args:
        config: SU2設定
        df: 入力データ
        n_folds: クロスバリデーションのfold数
    
    Returns:
        評価結果の辞書
    """
    # ターゲット変数の分離
    if 'target' in df.columns:
        target = df['target'].values
        feature_df = df.drop(columns=['target'])
    else:
        target = np.zeros(len(df))
        feature_df = df.copy()
    
    # SU2特徴量生成器の初期化
    generator = SU2FeatureGenerator(config)
    
    # クロスバリデーション
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(df))
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(feature_df)):
        X_train = feature_df.iloc[train_idx]
        X_val = feature_df.iloc[val_idx]
        y_train = target[train_idx]
        y_val = target[val_idx]
        
        # SU2特徴量生成
        _ = generator.fit_transform(X_train, fold_id=fold_idx)
        _ = generator.transform(X_val)
        
        # 簡易モデル（平均値予測）
        train_mean = y_train.mean()
        val_predictions = np.full(len(y_val), train_mean)
        
        oof_predictions[val_idx] = val_predictions
        
        # 評価
        rmse = np.sqrt(np.mean((y_val - val_predictions) ** 2))  # type: ignore[operator]
        fold_scores.append(rmse)
    
    # 全体評価
    overall_rmse = float(np.sqrt(np.mean((np.asarray(target) - oof_predictions) ** 2)))
    
    return {
        'overall_rmse': float(overall_rmse),
        'mean_fold_rmse': float(np.mean(fold_scores)),
        'std_fold_rmse': float(np.std(fold_scores)),
        'fold_scores': [float(s) for s in fold_scores],
        'n_features': len(generator.get_feature_columns()),
    }


def sweep_su2_policies(
    config_path: str | Path = "configs/feature_generation.yaml",
    input_data: pd.DataFrame | None = None,
    output_dir: str | Path = "results/ablation/SU2",
) -> pd.DataFrame:
    """SU2ポリシーのスイープを実行
    
    Args:
        config_path: 設定ファイルのパス
        input_data: 入力データ（Noneの場合はモックデータ生成）
        output_dir: 出力ディレクトリ
    
    Returns:
        スイープ結果のサマリDataFrame
    """
    # 設定の読み込み
    try:
        full_config = load_config(config_path)
        sweep_config = full_config.get('sweep', {}).get('su2', {})
        base_su2_config = full_config.get('su2', {})
    except FileNotFoundError:
        print("設定ファイルが見つかりません。デフォルト設定を使用します。")
        sweep_config = {
            'rolling_windows': [[7, 14, 30], [5, 10, 20, 40]],
            'ewma_spans': [[5, 10, 20], [5, 10, 20, 40]],
            'normalization_window': [20, 30],
            'metric': 'rmse',
            'n_folds': 5,
        }
        base_su2_config = {
            'rolling': {'windows': [7, 14, 30], 'functions': ['mean', 'std']},
            'ewma': {'spans': [5, 10, 20], 'adjust': False},
            'transitions': {'lags': [1, 7, 30], 'methods': ['diff', 'pct_change']},
            'normalization': {'methods': ['zscore'], 'rolling_window': 30},
            'dtypes': {'flag': 'uint8', 'small_int': 'int16', 'float': 'float32'},
            'fold': {'reset_on_boundary': True},
        }
    
    # 入力データの準備
    if input_data is None:
        print("モックデータを生成中...")
        df = create_mock_data(n_samples=500, n_features=3)
    else:
        df = input_data.copy()
    
    print(f"入力データサイズ: {df.shape}")
    
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # スイープ実行
    n_folds = sweep_config.get('n_folds', 5)
    results = []
    
    # ローリングウィンドウのスイープ
    print("\n=== ローリングウィンドウのスイープ ===")
    for windows in sweep_config.get('rolling_windows', [[7, 14, 30]]):
        config = base_su2_config.copy()
        config['rolling'] = {'windows': windows, 'functions': config['rolling']['functions']}
        
        print(f"評価中: rolling_windows={windows}")
        eval_result = evaluate_config(config, df, n_folds=n_folds)
        
        result_entry = {
            'policy': 'rolling_windows',
            'value': str(windows),
            'overall_rmse': eval_result['overall_rmse'],
            'mean_fold_rmse': eval_result['mean_fold_rmse'],
            'std_fold_rmse': eval_result['std_fold_rmse'],
            'n_features': eval_result['n_features'],
            'timestamp': datetime.now().isoformat(),
        }
        results.append(result_entry)
        
        # 個別結果の保存
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = output_path / f"sweep_rolling_windows_{timestamp_str}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({**result_entry, 'fold_scores': eval_result['fold_scores']}, f, indent=2)
        
        print(f"  → RMSE: {eval_result['overall_rmse']:.4f} (保存: {result_path.name})")
    
    # EWMAスパンのスイープ
    print("\n=== EWMAスパンのスイープ ===")
    for spans in sweep_config.get('ewma_spans', [[5, 10, 20]]):
        config = base_su2_config.copy()
        config['ewma'] = {'spans': spans, 'adjust': config['ewma']['adjust']}
        
        print(f"評価中: ewma_spans={spans}")
        eval_result = evaluate_config(config, df, n_folds=n_folds)
        
        result_entry = {
            'policy': 'ewma_spans',
            'value': str(spans),
            'overall_rmse': eval_result['overall_rmse'],
            'mean_fold_rmse': eval_result['mean_fold_rmse'],
            'std_fold_rmse': eval_result['std_fold_rmse'],
            'n_features': eval_result['n_features'],
            'timestamp': datetime.now().isoformat(),
        }
        results.append(result_entry)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = output_path / f"sweep_ewma_spans_{timestamp_str}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({**result_entry, 'fold_scores': eval_result['fold_scores']}, f, indent=2)
        
        print(f"  → RMSE: {eval_result['overall_rmse']:.4f} (保存: {result_path.name})")
    
    # 正規化ウィンドウのスイープ
    print("\n=== 正規化ウィンドウのスイープ ===")
    for window in sweep_config.get('normalization_window', [30]):
        config = base_su2_config.copy()
        config['normalization'] = {
            'methods': config['normalization']['methods'],
            'rolling_window': window
        }
        
        print(f"評価中: normalization_window={window}")
        eval_result = evaluate_config(config, df, n_folds=n_folds)
        
        result_entry = {
            'policy': 'normalization_window',
            'value': str(window),
            'overall_rmse': eval_result['overall_rmse'],
            'mean_fold_rmse': eval_result['mean_fold_rmse'],
            'std_fold_rmse': eval_result['std_fold_rmse'],
            'n_features': eval_result['n_features'],
            'timestamp': datetime.now().isoformat(),
        }
        results.append(result_entry)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = output_path / f"sweep_normalization_window_{timestamp_str}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({**result_entry, 'fold_scores': eval_result['fold_scores']}, f, indent=2)
        
        print(f"  → RMSE: {eval_result['overall_rmse']:.4f} (保存: {result_path.name})")
    
    # サマリの作成と保存
    summary_df = pd.DataFrame(results)
    summary_path = output_path / "sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n全スイープ結果のサマリを保存: {summary_path}")
    
    # ベストパラメータの表示
    best_idx = summary_df['overall_rmse'].idxmin()
    best_result = summary_df.iloc[best_idx]
    print("\nベストパラメータ:")
    print(f"  Policy: {best_result['policy']}")
    print(f"  Value: {best_result['value']}")
    print(f"  RMSE: {best_result['overall_rmse']:.4f}")
    
    print("\nスイープ完了!")
    
    return summary_df


if __name__ == "__main__":
    # コマンドライン実行時
    sweep_su2_policies()
