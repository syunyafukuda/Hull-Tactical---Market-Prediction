#!/usr/bin/env python
"""SU7 バリアント・スイープ実行スクリプト。

本スクリプトは configs/su7_sweep.yaml で定義された複数の SU7 バリアントを
順次実行し、各バリアントの結果を artifacts/SU7/<variant_name>/ に保存する。

使用例:
    python scripts/su7/run_su7_sweep.py
    python scripts/su7/run_su7_sweep.py --variants case_a case_b
    python scripts/su7/run_su7_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

# Add project root to path
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_sweep_config(config_path: Path) -> Dict[str, Any]:
    """スイープ設定を読み込む。"""
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_base_config(config_path: Path) -> Dict[str, Any]:
    """ベース feature_generation.yaml を読み込む。"""
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_su7_config_for_variant(
    variant_def: Mapping[str, Any],
    base_su7_section: Mapping[str, Any],
) -> Dict[str, Any]:
    """バリアント定義からSU7設定を構築する。

    Args:
        variant_def: バリアントの定義（su7_sweep.yaml の variants.<name>）
        base_su7_section: ベースの su7 セクション（feature_generation.yaml の su7）

    Returns:
        マージされた su7 設定
    """
    # ベース設定をコピー
    merged = dict(base_su7_section)

    # バリアント定義で上書き
    override_keys = [
        "su7_base_cols",
        "lags",
        "windows",
        "halflife_rsi",
        "eps",
        "rs_max",
        "use_rsi",
        "use_sign",
    ]
    for key in override_keys:
        if key in variant_def:
            merged[key] = variant_def[key]

    return merged


def estimate_feature_count(variant_def: Mapping[str, Any]) -> int:
    """バリアント定義から特徴量数を推定する。

    計算式:
    - diff/lag: 2 * len(lags) * B
    - rolling: 2 * len(windows) * B
    - rsi: B if use_rsi else 0
    - sign: B if use_sign else 0
    """
    n_base_cols = len(variant_def.get("su7_base_cols", []))
    n_lags = len(variant_def.get("lags", [1, 5, 20]))
    n_windows = len(variant_def.get("windows", [5, 20]))
    use_rsi = variant_def.get("use_rsi", True)
    use_sign = variant_def.get("use_sign", True)

    diff_lag_count = 2 * n_lags * n_base_cols
    rolling_count = 2 * n_windows * n_base_cols
    rsi_count = n_base_cols if use_rsi else 0
    sign_count = n_base_cols if use_sign else 0

    return diff_lag_count + rolling_count + rsi_count + sign_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数をパースする。"""
    parser = argparse.ArgumentParser(
        description="Run SU7 sweep across multiple variant configurations."
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default="configs/su7_sweep.yaml",
        help="Path to su7_sweep.yaml",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/feature_generation.yaml",
        help="Path to feature_generation.yaml",
    )
    parser.add_argument(
        "--preprocess-config",
        type=str,
        default="configs/preprocess.yaml",
        help="Path to preprocess.yaml",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing train/test files",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="artifacts/SU7",
        help="Base output directory for sweep results",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        help="Specific variants to run (default: all enabled variants)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without running",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Override n_splits for CV (default: use sweep config)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override n_estimators for LightGBM (default: use sweep config)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip variants that already have results",
    )
    return parser.parse_args(argv)


def run_variant(
    variant_name: str,
    variant_def: Mapping[str, Any],
    args: argparse.Namespace,
    base_su7_section: Mapping[str, Any],
) -> Dict[str, Any]:
    """単一のバリアントを実行する。

    Args:
        variant_name: バリアント名
        variant_def: バリアント定義
        args: コマンドライン引数
        base_su7_section: ベースの su7 設定

    Returns:
        実行結果のメタデータ
    """
    import subprocess
    import tempfile

    # 出力ディレクトリ
    out_dir = Path(args.output_base_dir) / variant_name

    if args.skip_existing:
        meta_path = out_dir / "model_meta.json"
        if meta_path.exists():
            print(f"[skip] {variant_name}: results already exist at {out_dir}")
            with meta_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

    # SU7 設定をマージ
    merged_su7 = build_su7_config_for_variant(variant_def, base_su7_section)

    # 一時的な設定ファイルを作成
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp_file:
        # ベース設定を読み込み、su7セクションを上書き
        base_config = load_base_config(Path(args.base_config))
        base_config["su7"] = merged_su7

        yaml.dump(base_config, tmp_file, default_flow_style=False, allow_unicode=True)
        tmp_config_path = tmp_file.name

    try:
        # train_su7.py を実行
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "feature_generation" / "su7" / "train_su7.py"),
            "--config-path",
            tmp_config_path,
            "--preprocess-config",
            args.preprocess_config,
            "--data-dir",
            args.data_dir,
            "--out-dir",
            str(out_dir),
        ]

        # オプション引数
        if args.n_splits is not None:
            cmd.extend(["--n-splits", str(args.n_splits)])
        if args.n_estimators is not None:
            cmd.extend(["--n-estimators", str(args.n_estimators)])

        print(f"[run] {variant_name}: {' '.join(cmd)}")

        if args.dry_run:
            print(f"[dry-run] Would execute: {' '.join(cmd)}")
            return {"variant_name": variant_name, "status": "dry-run"}

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[error] {variant_name} failed:")
            print(result.stderr)
            return {
                "variant_name": variant_name,
                "status": "failed",
                "error": result.stderr,
            }

        # 結果を読み込む
        meta_path = out_dir / "model_meta.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        else:
            return {"variant_name": variant_name, "status": "completed_no_meta"}

    finally:
        # 一時ファイルを削除
        Path(tmp_config_path).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """メインエントリーポイント。"""
    args = parse_args(argv)

    # 設定を読み込む
    sweep_config_path = Path(args.sweep_config)
    base_config_path = Path(args.base_config)

    if not sweep_config_path.exists():
        print(f"[error] Sweep config not found: {sweep_config_path}")
        return 1

    if not base_config_path.exists():
        print(f"[error] Base config not found: {base_config_path}")
        return 1

    sweep_config = load_sweep_config(sweep_config_path)
    base_config = load_base_config(base_config_path)

    variants = sweep_config.get("variants", {})
    sweep_settings = sweep_config.get("sweep_settings", {})
    enabled_variants = sweep_settings.get("enabled_variants", list(variants.keys()))

    # 実行対象のバリアントを決定
    if args.variants:
        target_variants = args.variants
    else:
        target_variants = enabled_variants

    # 存在しないバリアント名をチェック
    invalid_variants = [v for v in target_variants if v not in variants]
    if invalid_variants:
        print(f"[error] Unknown variants: {invalid_variants}")
        print(f"[info] Available variants: {list(variants.keys())}")
        return 1

    print(f"[info] Running {len(target_variants)} variants: {target_variants}")
    print()

    # スイープの概要を表示
    for variant_name in target_variants:
        variant_def = variants[variant_name]
        feature_count = estimate_feature_count(variant_def)
        expected = variant_def.get("expected_feature_count", "N/A")
        print(
            f"  - {variant_name}: "
            f"{len(variant_def.get('su7_base_cols', []))} cols, "
            f"~{feature_count} features (expected: {expected})"
        )
    print()

    if args.dry_run:
        print("[dry-run] Exiting without execution.")
        return 0

    # 各バリアントを実行
    base_su7_section = base_config.get("su7", {})
    results: List[Dict[str, Any]] = []

    for variant_name in target_variants:
        print(f"\n{'='*60}")
        print(f"[sweep] Running variant: {variant_name}")
        print(f"{'='*60}")

        variant_def = variants[variant_name]
        result = run_variant(variant_name, variant_def, args, base_su7_section)
        result["variant_name"] = variant_name
        results.append(result)

    # サマリーを表示
    print(f"\n{'='*60}")
    print("[summary] Sweep Results")
    print(f"{'='*60}")

    for result in results:
        variant_name = result.get("variant_name", "unknown")
        status = result.get("status", "unknown")
        oof_rmse = result.get("oof_rmse", "N/A")
        oof_msr = result.get("oof_best_metrics", {}).get("msr", "N/A")

        if isinstance(oof_rmse, float):
            oof_rmse = f"{oof_rmse:.6f}"
        if isinstance(oof_msr, float):
            oof_msr = f"{oof_msr:.6f}"

        print(f"  {variant_name}: status={status}, RMSE={oof_rmse}, MSR={oof_msr}")

    # サマリーを JSON として保存
    summary_path = Path(args.output_base_dir) / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n[ok] Sweep summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
