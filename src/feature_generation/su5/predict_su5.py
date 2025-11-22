"""Prediction script for SU5 feature generation.

This script demonstrates the SU5 inference pipeline structure.
Note: This is a minimal implementation for demonstration purposes.
Full implementation requires actual data and trained model.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from feature_generation.su5 import load_su5_config


def load_inference_bundle(bundle_path: Path) -> dict:
    """Load inference bundle containing pipeline and metadata.

    Args:
        bundle_path: Path to inference bundle pickle file.

    Returns:
        Dictionary with 'pipeline' and 'metadata' keys.
    """
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def create_submission(
    predictions: pd.Series, output_dir: Path, format: str = "csv"
) -> None:
    """Create submission file in specified format.

    Args:
        predictions: Prediction series with index as IDs.
        output_dir: Directory to save submission files.
        format: Output format ('csv', 'parquet', or 'both').
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {"id": predictions.index, "prediction": predictions.values}
    )

    # Save in requested format(s)
    if format in ("csv", "both"):
        csv_path = output_dir / "submission.csv"
        submission_df.to_csv(csv_path, index=False)
        print(f"✓ CSV submission saved: {csv_path}")

    if format in ("parquet", "both"):
        parquet_path = output_dir / "submission.parquet"
        submission_df.to_parquet(parquet_path, index=False)
        print(f"✓ Parquet submission saved: {parquet_path}")


def main() -> None:
    """Main prediction function.

    This is a skeleton implementation showing the inference flow.
    Full implementation would include:
    - Loading test data
    - Loading trained pipeline bundle
    - Applying transformations (SU1 + SU5 + preprocessing)
    - Making predictions
    - Post-processing (signal conversion, etc.)
    - Creating submission files
    """
    print("=== SU5 Prediction Pipeline ===\n")

    # Load configuration
    config_path = Path("configs/feature_generation.yaml")
    config = load_su5_config(config_path)

    print("Configuration loaded:")
    print(f"  - top_k_pairs: {config.top_k_pairs}")
    print(f"  - windows: {config.windows}\n")

    artifacts_dir = Path("artifacts/SU5")

    # Check if artifacts exist
    if not artifacts_dir.exists():
        print(f"⚠️  Artifacts directory not found: {artifacts_dir}")
        print("Please run train_su5.py first to generate artifacts.")
        return

    # Load feature list
    feature_list_path = artifacts_dir / "feature_list.json"
    if feature_list_path.exists():
        with open(feature_list_path) as f:
            feature_info = json.load(f)
        print(f"Loaded feature list: {len(feature_info['features'])} features")
    else:
        print("⚠️  Feature list not found")

    # In full implementation:
    # 1. Load test data
    # 2. Load inference bundle (pipeline + model)
    # 3. Apply same preprocessing as training:
    #    - SU1 features (if needed)
    #    - SU5 features
    #    - GroupImputers
    #    - Scaling
    # 4. Make predictions
    # 5. Post-process predictions
    # 6. Create submission files

    print("\nNote: This is a skeleton implementation.")
    print("Full implementation requires actual test data and trained model.\n")

    # Create dummy predictions for demonstration
    dummy_predictions = pd.Series(
        [0.5, 0.3, 0.7, 0.2, 0.8], index=[101, 102, 103, 104, 105]
    )

    # Save submission files
    create_submission(dummy_predictions, artifacts_dir, format="both")

    print("\n=== Prediction Complete ===")
    print(f"Submission files saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
