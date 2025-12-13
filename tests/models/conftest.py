# Model tests conftest
"""Shared fixtures for model tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_train_data() -> pd.DataFrame:
    """Create sample training data for model tests."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    data = {
        "Date": pd.date_range("2020-01-01", periods=n_samples, freq="W"),
        "TARGET": np.random.randn(n_samples) * 0.01,
    }
    
    for i in range(n_features):
        data[f"feature_{i}"] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def artifacts_dir(project_root: Path) -> Path:
    """Return artifacts directory."""
    return project_root / "artifacts" / "models"
