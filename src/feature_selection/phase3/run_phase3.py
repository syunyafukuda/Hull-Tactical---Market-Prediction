#!/usr/bin/env python
"""Phase 3 orchestration script.

This script runs the complete Phase 3 pipeline:
1. Correlation clustering (Phase 3-1)
2. Representative selection (Phase 3-2)
3. Tier3 exclusion list creation (Phase 3-3a)
4. Tier3 evaluation (Phase 3-3b)
5. Feature sets configuration (Phase 3-4)
6. Documentation generation (Phase 3-6)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Run complete Phase 3 pipeline."
    )
    ap.add_argument(
        "--config-path",
        type=str,
        default="configs/feature_generation.yaml",
        help="Path to feature_generation.yaml",
    )
    ap.add_argument(
        "--preprocess-config",
        type=str,
        default="configs/preprocess.yaml",
        help="Path to preprocess.yaml",
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing train/test files",
    )
    ap.add_argument(
        "--tier2-excluded",
        type=str,
        default="configs/feature_selection/tier2/excluded.json",
        help="Path to Tier2 excluded.json",
    )
    ap.add_argument(
        "--tier2-importance",
        type=str,
        default="results/feature_selection/tier2/importance_summary.csv",
        help="Path to Tier2 importance_summary.csv",
    )
    ap.add_argument(
        "--tier2-evaluation",
        type=str,
        default="results/feature_selection/tier2/evaluation.json",
        help="Path to Tier2 evaluation.json",
    )
    ap.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Correlation threshold for clustering (default: 0.95)",
    )
    ap.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip correlation clustering and use Tier2 as final",
    )
    ap.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip Tier3 evaluation (use only for testing)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of top features for FS_topK (default: 50)",
    )
    return ap.parse_args(argv)


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with code {result.returncode}")
        return result.returncode
    
    print(f"\n✅ {description} completed successfully")
    return 0


def check_tier3_rmse(
    tier2_eval_path: str,
    tier3_eval_path: str,
    threshold: float = 0.0001,
) -> Dict[str, Any]:
    """Check if Tier3 meets RMSE criteria."""
    # Load evaluations
    with open(tier2_eval_path, "r") as f:
        tier2_eval = json.load(f)
    
    with open(tier3_eval_path, "r") as f:
        tier3_eval = json.load(f)
    
    tier2_rmse = tier2_eval.get("oof_rmse", 0.0)
    tier3_rmse = tier3_eval.get("oof_rmse", 0.0)
    
    delta = tier3_rmse - tier2_rmse
    acceptable = delta <= threshold
    
    result = {
        "tier2_rmse": tier2_rmse,
        "tier3_rmse": tier3_rmse,
        "delta": delta,
        "threshold": threshold,
        "acceptable": acceptable,
        "decision": "✅ Tier3 adopted" if acceptable else "❌ Tier2 maintained",
    }
    
    return result


def generate_phase3_report(
    clustering_path: str,
    representatives_path: str,
    tier3_excluded_path: str,
    tier2_eval_path: str,
    tier3_eval_path: str,
    feature_sets_path: str,
    output_path: str,
    skip_clustering: bool = False,
    skip_evaluation: bool = False,
) -> None:
    """Generate Phase 3 report markdown."""
    # Load data
    if not skip_clustering:
        with open(clustering_path, "r") as f:
            clustering = json.load(f)
        with open(representatives_path, "r") as f:
            representatives = json.load(f)
        with open(tier3_excluded_path, "r") as f:
            tier3_excluded = json.load(f)
    else:
        clustering = {"n_clusters": 0, "n_singletons": 0}
        representatives = {"total_removed": 0}
    
    with open(feature_sets_path, "r") as f:
        feature_sets = json.load(f)
    
    # Check RMSE if evaluation was run (both clustering and evaluation must be enabled)
    rmse_result = None
    if not skip_clustering and not skip_evaluation:
        try:
            rmse_result = check_tier3_rmse(tier2_eval_path, tier3_eval_path)
        except FileNotFoundError:
            print("Warning: Tier3 evaluation file not found, skipping RMSE check")
            rmse_result = None
    
    # Generate report
    report_lines = [
        "# Phase 3: 相関クラスタリングと最終特徴セット確定 実行レポート",
        "",
        f"**実行日時**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 概要",
        "",
        "Phase 3 では、Tier2 特徴セット（120列）を入力として、相関クラスタリングによる冗長性削減と",
        "最終特徴セットの確定を行いました。",
        "",
        "## 実行結果",
        "",
    ]
    
    if skip_clustering:
        report_lines.extend([
            "### Phase 3-1: 相関クラスタリング（スキップ）",
            "",
            "相関クラスタリングはスキップされました。Tier2 をそのまま最終セットとして採用します。",
            "",
        ])
    else:
        report_lines.extend([
            "### Phase 3-1: 相関クラスタリング",
            "",
            "- **相関閾値**: 0.95",
            f"- **検出クラスタ数**: {clustering['n_clusters']}",
            f"- **シングルトン特徴数**: {clustering['n_singletons']}",
            "",
            "### Phase 3-2: クラスタ代表選出",
            "",
            f"- **削除候補特徴数**: {representatives['total_removed']}",
            "",
            "各クラスタから mean_gain 最大の特徴を代表として選出しました。",
            "",
            "### Phase 3-3: Tier3 評価",
            "",
            f"- **Tier3 除外特徴数**: {tier3_excluded['summary']['total_exclusions']}",
            "",
        ])
        
        if rmse_result:
            report_lines.extend([
                "#### RMSE 評価結果",
                "",
                "| Tier | OOF RMSE | Delta |",
                "|------|----------|-------|",
                f"| Tier2 | {rmse_result['tier2_rmse']:.6f} | - |",
                f"| Tier3 | {rmse_result['tier3_rmse']:.6f} | {rmse_result['delta']:+.6f} |",
                "",
                f"**判定**: {rmse_result['decision']}",
                "",
                f"Delta が閾値 {rmse_result['threshold']:.6f} {'以内' if rmse_result['acceptable'] else 'を超過'}のため、",
                f"{'Tier3 を採用' if rmse_result['acceptable'] else 'Tier2 を維持'}します。",
                "",
            ])
        else:
            report_lines.extend([
                "評価はスキップされました。",
                "",
            ])
    
    report_lines.extend([
        "### Phase 3-4: Feature Set 定義",
        "",
        "以下の Feature Set バリエーションを定義しました:",
        "",
        "| Set Name | Description | Features | OOF RMSE |",
        "|----------|-------------|----------|----------|",
    ])
    
    for name, config in feature_sets["feature_sets"].items():
        rmse_str = f"{config['oof_rmse']:.6f}" if config['oof_rmse'] is not None else "N/A"
        report_lines.append(
            f"| {name} | {config['description']} | {config['n_features']} | {rmse_str} |"
        )
    
    report_lines.extend([
        "",
        f"**推奨セット**: {feature_sets['recommended']}",
        "",
        "## 次のステップ",
        "",
        "1. 推奨 Feature Set を使用してモデル選定フェーズに進む",
        "2. 複数モデル（XGBoost, CatBoost, Ridge 等）で性能評価",
        "3. ハイパーパラメータ最適化",
        "4. アンサンブル検討",
        "",
        "## ファイル出力",
        "",
        "- `results/feature_selection/phase3/correlation_clusters.json`: クラスタ結果",
        "- `results/feature_selection/phase3/cluster_representatives.json`: 代表特徴",
        "- `configs/feature_selection/tier3/excluded.json`: Tier3 除外リスト",
        "- `results/feature_selection/tier3/evaluation.json`: Tier3 評価結果",
        "- `configs/feature_selection/feature_sets.json`: Feature Set 定義",
        "- `docs/feature_selection/phase3_report.md`: 本レポート",
        "",
    ])
    
    # Write report
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n✅ Phase 3 report written to {output}")


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    # If skip_clustering is True, force skip_evaluation to True as well
    # (no Tier3 to evaluate without clustering)
    if args.skip_clustering:
        args.skip_evaluation = True
    
    print("=" * 80)
    print("Phase 3 Pipeline Orchestration")
    print("=" * 80)
    print(f"Config path: {args.config_path}")
    print(f"Preprocess config: {args.preprocess_config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Tier2 excluded: {args.tier2_excluded}")
    print(f"Tier2 importance: {args.tier2_importance}")
    print(f"Correlation threshold: {args.correlation_threshold}")
    print(f"Skip clustering: {args.skip_clustering}")
    print(f"Skip evaluation: {args.skip_evaluation}")
    print(f"Top K: {args.topk}")
    print()
    
    phase3_dir = THIS_DIR
    
    # Phase 3-1: Correlation Clustering
    if not args.skip_clustering:
        ret = run_command(
            [
                sys.executable,
                str(phase3_dir / "correlation_clustering.py"),
                "--config-path", args.config_path,
                "--preprocess-config", args.preprocess_config,
                "--data-dir", args.data_dir,
                "--exclude-features", args.tier2_excluded,
                "--correlation-threshold", str(args.correlation_threshold),
                "--out-dir", "results/feature_selection/phase3",
            ],
            "Phase 3-1: Correlation Clustering",
        )
        if ret != 0:
            return ret
        
        # Phase 3-2: Select Representatives
        ret = run_command(
            [
                sys.executable,
                str(phase3_dir / "select_representatives.py"),
                "--clusters-json", "results/feature_selection/phase3/correlation_clusters.json",
                "--importance-csv", args.tier2_importance,
                "--out-dir", "results/feature_selection/phase3",
            ],
            "Phase 3-2: Select Cluster Representatives",
        )
        if ret != 0:
            return ret
        
        # Phase 3-3a: Create Tier3 Excluded
        ret = run_command(
            [
                sys.executable,
                str(phase3_dir / "create_tier3_excluded.py"),
                "--tier2-excluded", args.tier2_excluded,
                "--representatives-json", "results/feature_selection/phase3/cluster_representatives.json",
                "--out-file", "configs/feature_selection/tier3/excluded.json",
            ],
            "Phase 3-3a: Create Tier3 Excluded List",
        )
        if ret != 0:
            return ret
        
        # Phase 3-3b: Evaluate Tier3 (optional)
        if not args.skip_evaluation:
            ret = run_command(
                [
                    sys.executable,
                    "-m", "src.feature_selection.common.evaluate_baseline",
                    "--config-path", args.config_path,
                    "--preprocess-config", args.preprocess_config,
                    "--data-dir", args.data_dir,
                    "--exclude-features", "configs/feature_selection/tier3/excluded.json",
                    "--out-dir", "results/feature_selection/tier3",
                ],
                "Phase 3-3b: Evaluate Tier3",
            )
            if ret != 0:
                return ret
    else:
        print("\n⏭️  Skipping correlation clustering (--skip-clustering)")
    
    # Determine recommended feature set based on RMSE evaluation
    # Default to FS_full if clustering was skipped or Tier3 not acceptable
    recommended = "FS_full"
    rmse_check_result = None
    
    if not args.skip_clustering and not args.skip_evaluation:
        # Check if Tier3 meets RMSE criteria
        try:
            rmse_check_result = check_tier3_rmse(
                args.tier2_evaluation,
                "results/feature_selection/tier3/evaluation.json",
            )
            if rmse_check_result["acceptable"]:
                recommended = "FS_compact"
                print(f"\n✅ Tier3 RMSE acceptable (delta={rmse_check_result['delta']:+.6f}), recommending FS_compact")
            else:
                print(f"\n❌ Tier3 RMSE not acceptable (delta={rmse_check_result['delta']:+.6f}), recommending FS_full")
        except FileNotFoundError:
            print("\n⚠️  Tier3 evaluation not found, defaulting to FS_full")
    elif args.skip_clustering:
        print("\n⏭️  Clustering skipped, recommending FS_full")
    
    # Phase 3-4: Create Feature Sets
    ret = run_command(
        [
            sys.executable,
            str(phase3_dir / "create_feature_sets.py"),
            "--tier2-excluded", args.tier2_excluded,
            "--tier3-excluded", "configs/feature_selection/tier3/excluded.json",
            "--tier2-evaluation", args.tier2_evaluation,
            "--tier3-evaluation", "results/feature_selection/tier3/evaluation.json",
            "--tier2-importance", args.tier2_importance,
            "--topk", str(args.topk),
            "--recommended", recommended,
            "--out-file", "configs/feature_selection/feature_sets.json",
        ],
        "Phase 3-4: Create Feature Sets Configuration",
    )
    if ret != 0:
        return ret
    
    # Phase 3-6: Generate Report
    print("\n" + "=" * 80)
    print("Generating Phase 3 Report")
    print("=" * 80)
    
    generate_phase3_report(
        clustering_path="results/feature_selection/phase3/correlation_clusters.json",
        representatives_path="results/feature_selection/phase3/cluster_representatives.json",
        tier3_excluded_path="configs/feature_selection/tier3/excluded.json",
        tier2_eval_path=args.tier2_evaluation,
        tier3_eval_path="results/feature_selection/tier3/evaluation.json",
        feature_sets_path="configs/feature_selection/feature_sets.json",
        output_path="docs/feature_selection/phase3_report.md",
        skip_clustering=args.skip_clustering,
        skip_evaluation=args.skip_evaluation,
    )
    
    print("\n" + "=" * 80)
    print("✅ Phase 3 Pipeline Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review: docs/feature_selection/phase3_report.md")
    print("  2. Check: configs/feature_selection/feature_sets.json")
    print("  3. Proceed to model selection phase")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
