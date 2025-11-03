#!/usr/bin/env python
"""SU2 OOF sweep script for hyperparameter tuning.

本スクリプトは configs/feature_generation.yaml の su2 セクションに記載された
各ポリシーの候補をスイープし、確定させるためのロジックを実装する。
すべてのスイープ結果jsonとサマリcsvは results/ablation/SU2 配下に出力する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
	if str(path) not in sys.path:
		sys.path.append(str(path))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Sweep SU2 hyperparameters using OOF validation.")
	ap.add_argument("--config-path", type=str, default="configs/feature_generation.yaml")
	ap.add_argument("--out-dir", type=str, default="results/ablation/SU2")
	ap.add_argument("--n-splits", type=int, default=5)
	ap.add_argument("--random-state", type=int, default=42)
	return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	"""Main entry point for SU2 hyperparameter sweep."""
	args = parse_args(argv)
	
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	
	print("[info] SU2 OOF sweep not yet implemented - placeholder created")
	print(f"[info] Output directory: {out_dir}")
	
	# Placeholder: Create empty summary file
	summary_path = out_dir / "sweep_summary.csv"
	summary_df = pd.DataFrame({
		"config_id": [0],
		"rolling_windows": ["[5, 10, 20, 60]"],
		"ewma_alpha": ["[0.1, 0.3, 0.5]"],
		"oof_rmse": [np.nan],
		"oof_msr": [np.nan],
	})
	summary_df.to_csv(summary_path, index=False)
	print(f"[ok] wrote placeholder summary: {summary_path}")
	
	return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
	sys.exit(main())
