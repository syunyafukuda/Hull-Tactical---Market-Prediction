#!/usr/bin/env python
"""SU7 バリアント比較レポート生成スクリプト。

本スクリプトは artifacts/SU7/<variant_name>/ に保存された各バリアントの
OOF 指標を比較する一覧表を生成する。

使用例:
    python scripts/su7/compare_su7_variants.py
    python scripts/su7/compare_su7_variants.py --output-format csv
    python scripts/su7/compare_su7_variants.py --baseline baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Add project root to path
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def collect_variant_results(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """artifacts/SU7 以下のバリアント結果を収集する。

    Args:
        artifacts_dir: artifacts/SU7 ディレクトリのパス

    Returns:
        各バリアントのメタデータを含む辞書のリスト
    """
    results = []

    if not artifacts_dir.exists():
        return results

    for variant_dir in sorted(artifacts_dir.iterdir()):
        if not variant_dir.is_dir():
            continue

        meta_path = variant_dir / "model_meta.json"
        if not meta_path.exists():
            continue

        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

        # バリアント名を追加
        meta["variant_name"] = variant_dir.name
        meta["variant_dir"] = str(variant_dir)

        results.append(meta)

    return results


def extract_metrics(meta: Dict[str, Any]) -> Dict[str, Any]:
    """メタデータから比較用の指標を抽出する。

    Args:
        meta: model_meta.json から読み込んだメタデータ

    Returns:
        比較用の指標辞書
    """
    oof_best = meta.get("oof_best_metrics", {})
    su7_config = meta.get("su7_config", {})

    return {
        "variant_name": meta.get("variant_name", "unknown"),
        "su7_base_cols": len(su7_config.get("su7_base_cols", [])),
        "su7_feature_count": meta.get("su7_feature_count", 0),
        "total_feature_count": meta.get("su1_feature_count", 0)
        + meta.get("su5_feature_count", 0)
        + meta.get("su7_feature_count", 0),
        "use_rsi": su7_config.get("use_rsi", True),
        "use_sign": su7_config.get("use_sign", True),
        "windows": su7_config.get("windows", []),
        "oof_rmse": meta.get("oof_rmse", float("nan")),
        "msr": oof_best.get("msr", float("nan")),
        "msr_down": oof_best.get("msr_down", float("nan")),
        "vmsr": oof_best.get("vmsr", float("nan")),
        "n_splits": meta.get("n_splits", 0),
    }


def compute_deltas(
    metrics_list: List[Dict[str, Any]], baseline_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """ベースラインとの差分を計算する。

    Args:
        metrics_list: 指標リスト
        baseline_name: ベースラインのバリアント名（None の場合は先頭を使用）

    Returns:
        差分列を追加した指標リスト
    """
    if not metrics_list:
        return metrics_list

    # ベースラインを見つける
    baseline = None
    for m in metrics_list:
        if baseline_name is None or m["variant_name"] == baseline_name:
            baseline = m
            break

    if baseline is None:
        baseline = metrics_list[0]

    baseline_rmse = baseline["oof_rmse"]
    baseline_msr = baseline["msr"]
    baseline_vmsr = baseline["vmsr"]

    for m in metrics_list:
        m["delta_rmse"] = m["oof_rmse"] - baseline_rmse
        m["delta_msr"] = m["msr"] - baseline_msr
        m["delta_vmsr"] = m["vmsr"] - baseline_vmsr
        m["is_baseline"] = m["variant_name"] == baseline["variant_name"]

    return metrics_list


def format_table(metrics_list: List[Dict[str, Any]], show_deltas: bool = True) -> str:
    """メトリクスを表形式でフォーマットする。

    Args:
        metrics_list: 指標リスト
        show_deltas: 差分列を表示するか

    Returns:
        フォーマットされた表文字列
    """
    if not metrics_list:
        return "No results found."

    # ヘッダー
    headers = [
        "Variant",
        "Cols",
        "SU7 Feats",
        "RSI",
        "Sign",
        "RMSE",
        "MSR",
        "vMSR",
    ]

    if show_deltas:
        headers.extend(["ΔRMSE", "ΔMSR", "ΔvMSR"])

    # データ行
    rows = []
    for m in metrics_list:
        row = [
            m["variant_name"],
            str(m["su7_base_cols"]),
            str(m["su7_feature_count"]),
            "Y" if m["use_rsi"] else "N",
            "Y" if m["use_sign"] else "N",
            f"{m['oof_rmse']:.6f}" if not _is_nan(m["oof_rmse"]) else "N/A",
            f"{m['msr']:.6f}" if not _is_nan(m["msr"]) else "N/A",
            f"{m['vmsr']:.6f}" if not _is_nan(m["vmsr"]) else "N/A",
        ]

        if show_deltas:
            delta_rmse = m.get("delta_rmse", float("nan"))
            delta_msr = m.get("delta_msr", float("nan"))
            delta_vmsr = m.get("delta_vmsr", float("nan"))

            row.extend(
                [
                    f"{delta_rmse:+.6f}" if not _is_nan(delta_rmse) else "N/A",
                    f"{delta_msr:+.6f}" if not _is_nan(delta_msr) else "N/A",
                    f"{delta_vmsr:+.6f}" if not _is_nan(delta_vmsr) else "N/A",
                ]
            )

        rows.append(row)

    # 列幅を計算
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # フォーマット
    lines = []

    # ヘッダー行
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # データ行
    for row in rows:
        line = " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(line)

    return "\n".join(lines)


def _is_nan(value: Any) -> bool:
    """NaN かどうかを判定する。"""
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def export_csv(metrics_list: List[Dict[str, Any]], output_path: Path) -> None:
    """メトリクスを CSV に出力する。

    Args:
        metrics_list: 指標リスト
        output_path: 出力先パス
    """
    if not metrics_list:
        return

    fieldnames = [
        "variant_name",
        "su7_base_cols",
        "su7_feature_count",
        "total_feature_count",
        "use_rsi",
        "use_sign",
        "windows",
        "oof_rmse",
        "msr",
        "msr_down",
        "vmsr",
        "delta_rmse",
        "delta_msr",
        "delta_vmsr",
        "is_baseline",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in metrics_list:
            # windows をカンマ区切りの文字列に変換
            m_copy = dict(m)
            m_copy["windows"] = ",".join(str(w) for w in m.get("windows", []))
            writer.writerow(m_copy)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数をパースする。"""
    parser = argparse.ArgumentParser(
        description="Compare SU7 variant results and generate a report."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/SU7",
        help="Directory containing SU7 variant results",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline variant name for delta calculation (default: first result)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["table", "csv", "json"],
        default="table",
        help="Output format",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output file path (default: stdout for table, auto-generated for csv/json)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["variant_name", "oof_rmse", "msr", "vmsr"],
        default="oof_rmse",
        help="Sort results by this column",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default: descending for msr/vmsr, ascending for rmse)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """メインエントリーポイント。"""
    args = parse_args(argv)

    artifacts_dir = Path(args.artifacts_dir)

    if not artifacts_dir.exists():
        print(f"[error] Artifacts directory not found: {artifacts_dir}")
        return 1

    # 結果を収集
    results = collect_variant_results(artifacts_dir)

    if not results:
        print(f"[info] No variant results found in {artifacts_dir}")
        return 0

    # 指標を抽出
    metrics_list = [extract_metrics(m) for m in results]

    # 差分を計算
    metrics_list = compute_deltas(metrics_list, baseline_name=args.baseline)

    # ソート
    sort_key = args.sort_by
    if sort_key == "variant_name":
        reverse = not args.ascending
    elif sort_key in ("msr", "vmsr"):
        reverse = not args.ascending  # MSR/vMSR は高い方が良い
    else:
        reverse = args.ascending  # RMSE は低い方が良い

    metrics_list.sort(key=lambda m: m.get(sort_key, float("inf")), reverse=reverse)

    # 出力
    if args.output_format == "table":
        table = format_table(metrics_list)
        if args.output_path:
            Path(args.output_path).write_text(table, encoding="utf-8")
            print(f"[ok] Table saved to: {args.output_path}")
        else:
            print(table)

    elif args.output_format == "csv":
        output_path = Path(args.output_path or artifacts_dir / "comparison.csv")
        export_csv(metrics_list, output_path)
        print(f"[ok] CSV saved to: {output_path}")

    elif args.output_format == "json":
        output_path = Path(args.output_path or artifacts_dir / "comparison.json")
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics_list, fh, indent=2, ensure_ascii=False)
        print(f"[ok] JSON saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
