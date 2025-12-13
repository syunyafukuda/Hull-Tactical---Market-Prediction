"""Ridge regression model training and evaluation.

This module provides Ridge regression (L2-regularized linear model) for ensemble diversity.
Ridge offers fundamentally different predictions from tree-based models, making it valuable
for model ensembling.

Key features:
- Linear model with L2 regularization
- High interpretability (coefficient analysis)
- Low prediction correlation with gradient boosting models
- Requires StandardScaler for proper feature scaling
"""
