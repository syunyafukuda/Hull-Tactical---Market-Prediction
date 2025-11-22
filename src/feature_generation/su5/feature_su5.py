"""SU5 feature generation: Co-missing structure features.

This module generates features based on co-missing patterns between columns,
including:
- Single-day co-missing flags (co_miss_now)
- Rolling co-missing rates (co_miss_rollrate_W)
- Per-column co-missing degree (co_miss_deg)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class SU5Config:
    """Configuration for SU5 feature generation.

    Attributes:
        enabled: Whether SU5 features are enabled.
        base_features: Base feature set to use (e.g., 'su1').
        id_column: Column name for date/time identifier.
        output_prefix: Prefix for output feature names.
        top_k_pairs: Number of top co-missing pairs to select.
        top_k_pairs_per_group: Optional number of top pairs per group.
        windows: List of window sizes for rolling co-missing rate.
        reset_each_fold: Whether to reset rolling stats at fold boundaries.
        dtype_flag: Data type for flag features.
        dtype_int: Data type for integer features.
        dtype_float: Data type for float features.
    """

    enabled: bool = False
    base_features: str = "su1"
    id_column: str = "date_id"
    output_prefix: str = "su5"
    top_k_pairs: int = 10
    top_k_pairs_per_group: int | None = None
    windows: list[int] | None = None
    reset_each_fold: bool = True
    dtype_flag: str = "uint8"
    dtype_int: str = "int16"
    dtype_float: str = "float32"

    def __post_init__(self) -> None:
        """Validate and set defaults after initialization."""
        if self.windows is None:
            self.windows = [5, 20]


def load_su5_config(config_path: str | Path) -> SU5Config:
    """Load SU5 configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        SU5Config instance with loaded configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        KeyError: If 'su5' section is missing from config.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    if "su5" not in config_dict:
        raise KeyError("'su5' section not found in config file")

    su5_config = config_dict["su5"]

    # Extract dtype settings
    dtype_settings = su5_config.get("dtype", {})

    return SU5Config(
        enabled=su5_config.get("enabled", False),
        base_features=su5_config.get("base_features", "su1"),
        id_column=su5_config.get("id_column", "date_id"),
        output_prefix=su5_config.get("output_prefix", "su5"),
        top_k_pairs=su5_config.get("top_k_pairs", 10),
        top_k_pairs_per_group=su5_config.get("top_k_pairs_per_group"),
        windows=su5_config.get("windows", [5, 20]),
        reset_each_fold=su5_config.get("reset_each_fold", True),
        dtype_flag=dtype_settings.get("flag", "uint8"),
        dtype_int=dtype_settings.get("int", "int16"),
        dtype_float=dtype_settings.get("float", "float32"),
    )


class SU5FeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate co-missing structure features from SU1 output.

    This transformer analyzes missing value patterns (m/<col> columns from SU1)
    and generates features representing co-missing structure:
    - co_miss_now/<a>__<b>: Single-day co-missing flag
    - co_miss_rollrate_W/<a>__<b>: Rolling co-missing rate
    - co_miss_deg/<col>: Per-column co-missing degree

    Attributes:
        config: SU5Config instance with feature generation settings.
        top_pairs_: List of (col_a, col_b) tuples for top-k co-missing pairs.
        feature_names_: List of generated feature names.
        m_columns_: List of m/<col> column names found in input data.
    """

    def __init__(self, config: SU5Config | None = None) -> None:
        """Initialize SU5FeatureGenerator.

        Args:
            config: SU5Config instance. If None, uses default config.
        """
        self.config = config if config is not None else SU5Config()
        self.top_pairs_: list[tuple[str, str]] = []
        self.feature_names_: list[str] = []
        self.m_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> SU5FeatureGenerator:
        """Fit the transformer on training data.

        Identifies m/<col> columns, computes co-missing scores for all pairs,
        and selects top-k pairs based on co-missing frequency.

        Args:
            X: Input DataFrame containing m/<col> columns from SU1.
            y: Ignored (for sklearn compatibility).

        Returns:
            self: Fitted transformer.
        """
        # Identify m/<col> columns
        self.m_columns_ = [col for col in X.columns if col.startswith("m/")]

        if len(self.m_columns_) == 0:
            # No missing indicator columns found
            self.top_pairs_ = []
            self.feature_names_ = []
            return self

        # Compute co-missing scores for all pairs
        m_data = X[self.m_columns_]
        assert isinstance(m_data, pd.DataFrame), "Expected DataFrame"
        co_miss_scores = self._compute_comiss_scores(m_data)

        # Select top-k pairs
        self.top_pairs_ = self._select_top_k_pairs(
            co_miss_scores, self.config.top_k_pairs
        )

        # Build feature name list
        self.feature_names_ = self._build_feature_names()

        return self

    def transform(
        self, X: pd.DataFrame, fold_indices: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Transform input data by adding co-missing structure features.

        Args:
            X: Input DataFrame containing m/<col> columns from SU1.
            fold_indices: Optional array indicating fold membership for each row.
                         Used for fold boundary reset if config.reset_each_fold is True.

        Returns:
            DataFrame with original columns plus SU5 features.
        """
        if len(self.top_pairs_) == 0:
            # No features to generate
            return X.copy()

        X_out = X.copy()

        # Extract missing indicators for selected pairs
        m_data = X[self.m_columns_].values

        # Category A: Single-day co-missing flags
        for col_a, col_b in self.top_pairs_:
            idx_a = self.m_columns_.index(f"m/{col_a}")
            idx_b = self.m_columns_.index(f"m/{col_b}")

            # Co-missing flag: both are missing (m==1)
            col_a_vals = m_data[:, idx_a]  # type: ignore[index]
            col_b_vals = m_data[:, idx_b]  # type: ignore[index]
            co_miss_flag = (col_a_vals == 1) & (col_b_vals == 1)
            feature_name = f"co_miss_now/{col_a}__{col_b}"
            X_out[feature_name] = co_miss_flag.astype(self.config.dtype_flag)

        # Category B: Rolling co-missing rates
        if self.config.windows:
            for window in self.config.windows:
                for col_a, col_b in self.top_pairs_:
                    feature_name = f"co_miss_rollrate_{window}/{col_a}__{col_b}"
                    X_out[feature_name] = self._compute_rolling_comiss_rate(
                        X, col_a, col_b, window, fold_indices
                    )

        # Category C: Per-column co-missing degree (optional, simplified version)
        # Count how many top pairs each column participates in
        col_degree: dict[str, int] = {}
        for col_a, col_b in self.top_pairs_:
            col_degree[col_a] = col_degree.get(col_a, 0) + 1
            col_degree[col_b] = col_degree.get(col_b, 0) + 1

        for col, degree in col_degree.items():
            feature_name = f"co_miss_deg/{col}"
            X_out[feature_name] = np.full(
                len(X), degree, dtype=self.config.dtype_int
            )

        return X_out

    def _compute_comiss_scores(self, m_df: pd.DataFrame) -> pd.DataFrame:
        """Compute co-missing scores for all column pairs.

        Args:
            m_df: DataFrame with m/<col> columns (1=missing, 0=observed).

        Returns:
            DataFrame with co-missing scores (higher = more frequently co-missing).
        """
        n_cols = len(m_df.columns)
        scores = np.zeros((n_cols, n_cols))

        m_values = m_df.values

        # Compute pairwise co-missing counts
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                # Count rows where both columns are missing
                co_miss_count = np.sum((m_values[:, i] == 1) & (m_values[:, j] == 1))
                scores[i, j] = co_miss_count
                scores[j, i] = co_miss_count

        # Convert to DataFrame for easier handling
        score_df = pd.DataFrame(scores, index=m_df.columns, columns=m_df.columns)
        return score_df

    def _select_top_k_pairs(
        self, score_df: pd.DataFrame, k: int
    ) -> list[tuple[str, str]]:
        """Select top-k column pairs with highest co-missing scores.

        Args:
            score_df: DataFrame with co-missing scores.
            k: Number of top pairs to select.

        Returns:
            List of (col_a, col_b) tuples for top-k pairs.
        """
        # Get upper triangle indices (avoid duplicates)
        pairs = []
        for i in range(len(score_df)):
            for j in range(i + 1, len(score_df)):
                col_a = str(score_df.index[i]).replace("m/", "")
                col_b = str(score_df.columns[j]).replace("m/", "")
                score = score_df.iloc[i, j]
                pairs.append((score, col_a, col_b))

        # Sort by score (descending) and take top-k
        pairs.sort(reverse=True, key=lambda x: x[0])
        top_pairs = [(col_a, col_b) for _, col_a, col_b in pairs[:k]]

        return top_pairs

    def _compute_rolling_comiss_rate(
        self,
        X: pd.DataFrame,
        col_a: str,
        col_b: str,
        window: int,
        fold_indices: np.ndarray | None,
    ) -> np.ndarray:
        """Compute rolling co-missing rate for a column pair.

        Args:
            X: Input DataFrame.
            col_a: First column name (without m/ prefix).
            col_b: Second column name (without m/ prefix).
            window: Rolling window size.
            fold_indices: Optional fold boundary information.

        Returns:
            Array of rolling co-missing rates.
        """
        m_col_a = f"m/{col_a}"
        m_col_b = f"m/{col_b}"

        # Get co-missing flags
        co_miss = ((X[m_col_a] == 1) & (X[m_col_b] == 1)).astype(int)

        # Compute rolling sum
        if fold_indices is not None and self.config.reset_each_fold:
            # Reset at fold boundaries
            roll_rate = self._rolling_with_fold_reset(
                co_miss.values, window, fold_indices
            )
        else:
            # Simple rolling without reset
            roll_sum = co_miss.rolling(window=window, min_periods=1).sum()
            roll_count = co_miss.rolling(window=window, min_periods=1).count()
            roll_rate = (roll_sum / roll_count).fillna(0.0).values

        return roll_rate.astype(self.config.dtype_float)

    def _rolling_with_fold_reset(
        self, values: np.ndarray, window: int, fold_indices: np.ndarray
    ) -> np.ndarray:
        """Compute rolling statistics with fold boundary reset.

        Args:
            values: Array of values to compute rolling stats on.
            window: Rolling window size.
            fold_indices: Array indicating fold membership.

        Returns:
            Array of rolling rates with fold reset applied.
        """
        result = np.zeros_like(values, dtype=float)

        # Compute fold boundaries
        fold_boundaries = self._compute_fold_boundaries(fold_indices)

        for start, end in fold_boundaries:
            fold_values = values[start:end]
            n = len(fold_values)

            for i in range(n):
                # Window starts at max(0, i-window+1) within this fold
                win_start = max(0, i - window + 1)
                win_values = fold_values[win_start : i + 1]

                if len(win_values) > 0:
                    result[start + i] = np.mean(win_values)

        return result

    def _compute_fold_boundaries(
        self, fold_indices: np.ndarray
    ) -> list[tuple[int, int]]:
        """Compute fold boundaries from fold indices.

        Args:
            fold_indices: Array indicating fold membership for each row.

        Returns:
            List of (start, end) tuples for each fold.
        """
        boundaries = []
        unique_folds = np.unique(fold_indices)

        for fold_id in unique_folds:
            fold_mask = fold_indices == fold_id
            fold_positions = np.where(fold_mask)[0]
            if len(fold_positions) > 0:
                boundaries.append((fold_positions[0], fold_positions[-1] + 1))

        return boundaries

    def _build_feature_names(self) -> list[str]:
        """Build list of feature names that will be generated.

        Returns:
            List of feature names.
        """
        feature_names = []

        # Category A: co_miss_now
        for col_a, col_b in self.top_pairs_:
            feature_names.append(f"co_miss_now/{col_a}__{col_b}")

        # Category B: co_miss_rollrate
        if self.config.windows:
            for window in self.config.windows:
                for col_a, col_b in self.top_pairs_:
                    feature_names.append(f"co_miss_rollrate_{window}/{col_a}__{col_b}")

        # Category C: co_miss_deg
        col_set = set()
        for col_a, col_b in self.top_pairs_:
            col_set.add(col_a)
            col_set.add(col_b)

        for col in sorted(col_set):
            feature_names.append(f"co_miss_deg/{col}")

        return feature_names
