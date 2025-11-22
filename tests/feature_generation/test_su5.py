"""Tests for SU5 feature generation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_generation.su5 import SU5Config, SU5FeatureGenerator, load_su5_config


@pytest.fixture
def sample_config() -> SU5Config:
    """Create a sample SU5Config for testing."""
    return SU5Config(
        enabled=True,
        top_k_pairs=3,
        windows=[5, 10],
        reset_each_fold=True,
        dtype_flag="uint8",
        dtype_int="int16",
        dtype_float="float32",
    )


@pytest.fixture
def sample_data_with_missing() -> pd.DataFrame:
    """Create sample data with missing indicators (m/<col> columns).

    This simulates SU1 output with missing value indicators.
    """
    np.random.seed(42)
    n_rows = 100

    # Create 5 columns with missing indicators
    data = {
        "date_id": range(n_rows),
        "m/feature_A": np.random.randint(0, 2, n_rows),
        "m/feature_B": np.random.randint(0, 2, n_rows),
        "m/feature_C": np.random.randint(0, 2, n_rows),
        "m/feature_D": np.random.randint(0, 2, n_rows),
        "m/feature_E": np.random.randint(0, 2, n_rows),
    }

    return pd.DataFrame(data)


def test_su5_config_loading(tmp_path: Path) -> None:
    """Test loading SU5Config from YAML file."""
    config_file = tmp_path / "test_config.yaml"
    config_content = """
su5:
  enabled: true
  base_features: su1
  id_column: date_id
  output_prefix: su5
  top_k_pairs: 15
  windows: [5, 10, 20]
  reset_each_fold: false
  dtype:
    flag: uint8
    int: int16
    float: float32
"""
    config_file.write_text(config_content)

    config = load_su5_config(config_file)

    assert config.enabled is True
    assert config.base_features == "su1"
    assert config.id_column == "date_id"
    assert config.output_prefix == "su5"
    assert config.top_k_pairs == 15
    assert config.windows == [5, 10, 20]
    assert config.reset_each_fold is False
    assert config.dtype_flag == "uint8"
    assert config.dtype_int == "int16"
    assert config.dtype_float == "float32"


def test_su5_config_loading_missing_file() -> None:
    """Test that loading from non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_su5_config("nonexistent_config.yaml")


def test_su5_all_observed_columns() -> None:
    """Test SU5 with all observed columns (m/<col> == 0 everywhere).

    Expected behavior:
    - co_miss_now flags should be all 0
    - Rolling rates should be 0 or NaN
    """
    config = SU5Config(enabled=True, top_k_pairs=2, windows=[5])

    # Create data where all columns are observed (m==0)
    n_rows = 50
    data = {
        "date_id": range(n_rows),
        "m/feature_A": np.zeros(n_rows, dtype=int),
        "m/feature_B": np.zeros(n_rows, dtype=int),
        "m/feature_C": np.zeros(n_rows, dtype=int),
    }
    df = pd.DataFrame(data)

    generator = SU5FeatureGenerator(config)
    generator.fit(df)
    result = generator.transform(df)

    # Check that co_miss_now features are all 0
    co_miss_now_cols = [col for col in result.columns if col.startswith("co_miss_now/")]
    for col in co_miss_now_cols:
        assert result[col].sum() == 0, f"{col} should be all zeros"

    # Check that rolling rates are 0
    rollrate_cols = [
        col for col in result.columns if col.startswith("co_miss_rollrate_")
    ]
    for col in rollrate_cols:
        assert (
            result[col].max() <= 0.0
        ), f"{col} should have all zeros (no co-missing)"


def test_su5_all_nan_columns() -> None:
    """Test SU5 with all NaN columns (m/<col> == 1 everywhere).

    Expected behavior:
    - Co-missing scores should be maximal
    - Top-k pairs should include these columns
    - co_miss_now flags should be all 1
    """
    config = SU5Config(enabled=True, top_k_pairs=2, windows=[5])

    # Create data where all columns are missing (m==1)
    n_rows = 50
    data = {
        "date_id": range(n_rows),
        "m/feature_A": np.ones(n_rows, dtype=int),
        "m/feature_B": np.ones(n_rows, dtype=int),
        "m/feature_C": np.ones(n_rows, dtype=int),
    }
    df = pd.DataFrame(data)

    generator = SU5FeatureGenerator(config)
    generator.fit(df)

    # All pairs should have maximal co-missing score
    assert len(generator.top_pairs_) == config.top_k_pairs

    result = generator.transform(df)

    # Check that co_miss_now features are all 1
    co_miss_now_cols = [col for col in result.columns if col.startswith("co_miss_now/")]
    assert len(co_miss_now_cols) > 0, "Should generate co_miss_now features"

    for col in co_miss_now_cols:
        assert result[col].min() == 1, f"{col} should be all ones"
        assert result[col].max() == 1, f"{col} should be all ones"


def test_su5_single_co_miss_pair() -> None:
    """Test SU5 with a single pair that co-misses consistently.

    Expected behavior:
    - The specific pair should be selected as top-1
    - co_miss_now for that pair should match the co-missing pattern
    """
    config = SU5Config(enabled=True, top_k_pairs=1, windows=[])

    # Create data where feature_A and feature_B always co-miss
    # and other features are independent
    n_rows = 50
    co_miss_pattern = np.array([1, 0, 1, 0, 1] * 10)  # Alternating pattern

    data = {
        "date_id": range(n_rows),
        "m/feature_A": co_miss_pattern,
        "m/feature_B": co_miss_pattern,  # Same pattern as A
        "m/feature_C": np.random.randint(0, 2, n_rows),
        "m/feature_D": np.random.randint(0, 2, n_rows),
    }
    df = pd.DataFrame(data)

    generator = SU5FeatureGenerator(config)
    generator.fit(df)

    # The top pair should be (feature_A, feature_B)
    assert len(generator.top_pairs_) == 1
    pair = generator.top_pairs_[0]
    assert set(pair) == {"feature_A", "feature_B"}

    result = generator.transform(df)

    # Check co_miss_now for this pair
    co_miss_now_cols = [col for col in result.columns if col.startswith("co_miss_now/")]
    assert len(co_miss_now_cols) == 1

    # The co_miss_now should match the pattern
    col_name = co_miss_now_cols[0]
    expected_co_miss = co_miss_pattern.astype("uint8")
    np.testing.assert_array_equal(result[col_name].values, expected_co_miss)


def test_su5_fold_reset() -> None:
    """Test that fold boundary reset works correctly for rolling features.

    Expected behavior:
    - With reset_each_fold=True, rolling window should not cross fold boundaries
    - Values at the start of each fold should not depend on previous fold
    """
    config = SU5Config(enabled=True, top_k_pairs=1, windows=[5], reset_each_fold=True)

    # Create simple co-missing pattern
    n_rows = 30
    data = {
        "date_id": range(n_rows),
        "m/feature_A": np.ones(n_rows, dtype=int),
        "m/feature_B": np.ones(n_rows, dtype=int),
    }
    df = pd.DataFrame(data)

    # Create fold indices: 3 folds of 10 rows each
    fold_indices = np.repeat([0, 1, 2], 10)

    generator = SU5FeatureGenerator(config)
    generator.fit(df)
    result = generator.transform(df, fold_indices=fold_indices)

    # Get rolling rate column
    rollrate_cols = [
        col for col in result.columns if col.startswith("co_miss_rollrate_")
    ]
    assert len(rollrate_cols) > 0

    rollrate_col = rollrate_cols[0]
    rollrate_values = result[rollrate_col].values

    # At fold boundaries (indices 10, 20), the rolling rate should reset
    # The first value of each fold should not depend on the previous fold
    # Since all values are 1 (co-missing), rates should build up within each fold

    # First position of each fold should have rate based only on that position
    assert rollrate_values[0] == 1.0  # First position of fold 0
    assert rollrate_values[10] == 1.0  # First position of fold 1
    assert rollrate_values[20] == 1.0  # First position of fold 2


def test_su5_output_shape() -> None:
    """Test that SU5 generates the correct number of features.

    Expected behavior:
    - Number of co_miss_now features = top_k_pairs
    - Number of rollrate features = top_k_pairs * len(windows)
    - Number of degree features = number of unique columns in top pairs
    """
    config = SU5Config(enabled=True, top_k_pairs=3, windows=[5, 10])

    n_rows = 50
    data = {
        "date_id": range(n_rows),
        "m/feature_A": np.random.randint(0, 2, n_rows),
        "m/feature_B": np.random.randint(0, 2, n_rows),
        "m/feature_C": np.random.randint(0, 2, n_rows),
        "m/feature_D": np.random.randint(0, 2, n_rows),
    }
    df = pd.DataFrame(data)

    generator = SU5FeatureGenerator(config)
    generator.fit(df)
    result = generator.transform(df)

    # Count generated features
    co_miss_now_cols = [col for col in result.columns if col.startswith("co_miss_now/")]
    rollrate_cols = [
        col for col in result.columns if col.startswith("co_miss_rollrate_")
    ]
    degree_cols = [col for col in result.columns if col.startswith("co_miss_deg/")]

    # Check counts
    assert (
        len(co_miss_now_cols) == config.top_k_pairs
    ), f"Expected {config.top_k_pairs} co_miss_now features"
    assert config.windows is not None, "Windows should not be None"
    expected_rollrate = config.top_k_pairs * len(config.windows)
    assert (
        len(rollrate_cols) == expected_rollrate
    ), f"Expected {expected_rollrate} rollrate features"

    # Degree features depend on unique columns in pairs
    assert len(degree_cols) > 0, "Should generate at least some degree features"


def test_su5_dtype() -> None:
    """Test that generated features have correct data types.

    Expected behavior:
    - co_miss_now flags: uint8
    - co_miss_deg counts: int16
    - co_miss_rollrate rates: float32
    """
    config = SU5Config(
        enabled=True,
        top_k_pairs=2,
        windows=[5],
        dtype_flag="uint8",
        dtype_int="int16",
        dtype_float="float32",
    )

    n_rows = 50
    data = {
        "date_id": range(n_rows),
        "m/feature_A": np.random.randint(0, 2, n_rows),
        "m/feature_B": np.random.randint(0, 2, n_rows),
        "m/feature_C": np.random.randint(0, 2, n_rows),
    }
    df = pd.DataFrame(data)

    generator = SU5FeatureGenerator(config)
    generator.fit(df)
    result = generator.transform(df)

    # Check dtypes
    co_miss_now_cols = [col for col in result.columns if col.startswith("co_miss_now/")]
    for col in co_miss_now_cols:
        assert (
            result[col].dtype == np.dtype("uint8")
        ), f"{col} should be uint8, got {result[col].dtype}"

    rollrate_cols = [
        col for col in result.columns if col.startswith("co_miss_rollrate_")
    ]
    for col in rollrate_cols:
        assert (
            result[col].dtype == np.dtype("float32")
        ), f"{col} should be float32, got {result[col].dtype}"

    degree_cols = [col for col in result.columns if col.startswith("co_miss_deg/")]
    for col in degree_cols:
        assert (
            result[col].dtype == np.dtype("int16")
        ), f"{col} should be int16, got {result[col].dtype}"


def test_su5_no_missing_columns() -> None:
    """Test SU5 behavior when no m/<col> columns are present.

    Expected behavior:
    - Should not generate any features
    - Should return input DataFrame unchanged
    """
    config = SU5Config(enabled=True, top_k_pairs=5)

    # Create data without any m/<col> columns
    n_rows = 50
    data = {
        "date_id": range(n_rows),
        "feature_A": np.random.rand(n_rows),
        "feature_B": np.random.rand(n_rows),
    }
    df = pd.DataFrame(data)

    generator = SU5FeatureGenerator(config)
    generator.fit(df)
    result = generator.transform(df)

    # Should return same columns
    assert list(result.columns) == list(df.columns)
    assert len(generator.top_pairs_) == 0
    assert len(generator.feature_names_) == 0
