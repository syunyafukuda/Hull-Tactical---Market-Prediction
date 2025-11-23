#!/usr/bin/env python
"""SU4 モデルの推論エントリーポイント。

学習済みの SU4 パイプラインを読み込み、テストデータに対して推論を実行する。

Usage:
    python predict_su4.py --bundle artifacts/SU4/inference_bundle.pkl --test data/raw/test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
	if str(path) not in sys.path:
		sys.path.append(str(path))


def main() -> int:
	"""SU4 推論パイプラインを実行する。"""
	parser = argparse.ArgumentParser(description="SU4 inference pipeline")
	parser.add_argument(
		"--bundle",
		type=str,
		default="artifacts/SU4/inference_bundle.pkl",
		help="Path to trained pipeline bundle",
	)
	parser.add_argument(
		"--test",
		type=str,
		default="data/raw/test.csv",
		help="Path to test data CSV",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="submission.csv",
		help="Path to output submission CSV",
	)
	args = parser.parse_args()

	bundle_path = Path(args.bundle)
	test_path = Path(args.test)

	print("="*80)
	print("SU4 Inference Pipeline - Development Status")
	print("="*80)
	print("\nCurrent Implementation:")
	print("✓ SU4FeatureGenerator with all feature categories")
	print("✓ SU4FeatureAugmenter for pipeline integration")
	print("✓ Configuration and testing infrastructure")
	print("\nFor Inference:")
	print("1. Train a complete pipeline using train_su4.py")
	print("2. Save the pipeline bundle with joblib")
	print("3. Load and apply to test data")
	print("\nExample usage in notebooks:")
	print("  from src.feature_generation.su4.feature_su4 import SU4FeatureGenerator")
	print("  generator = SU4FeatureGenerator(config)")
	print("  generator.fit(raw_train, imputed_train)")
	print("  features = generator.transform(raw_test, imputed_test)")
	print("="*80)

	# Check if bundle exists
	if bundle_path.exists():
		print(f"\nLoading bundle from {bundle_path}...")
		try:
			_pipeline = joblib.load(bundle_path)
			print("Bundle loaded successfully!")
		except Exception as e:
			print(f"Error loading bundle: {e}", file=sys.stderr)
			return 1
	else:
		print(f"\nBundle not found at {bundle_path}")
		print("Please run train_su4.py first to create the inference bundle.")
		return 1

	# Check if test data exists
	if test_path.exists():
		print(f"\nLoading test data from {test_path}...")
		try:
			test_data = pd.read_csv(test_path)
			print(f"Loaded {len(test_data)} rows")
		except Exception as e:
			print(f"Error loading test data: {e}", file=sys.stderr)
			return 1
	else:
		print(f"\nTest data not found at {test_path}")
		return 1

	return 0


if __name__ == "__main__":
	sys.exit(main())
