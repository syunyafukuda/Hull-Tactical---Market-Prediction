"""SU5 feature generation module for co-missing structure analysis.

This module provides functionality to generate features based on co-missing patterns
in the data, analyzing which columns tend to be missing together.
"""

from .feature_su5 import (
    SU5Config,
    SU5FeatureGenerator,
    load_su5_config,
)

__all__ = [
    "SU5Config",
    "SU5FeatureGenerator",
    "load_su5_config",
]
