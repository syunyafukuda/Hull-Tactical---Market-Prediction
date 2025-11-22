# Feature Generation Documentation

This directory contains documentation for feature generation modules used in the Hull Tactical Market Prediction project.

## Overview

Feature generation is organized into modular units (SU: Structure Units) that each handle specific aspects of feature engineering:

## Implemented Modules

### SU5: Co-Missing Structure Features

**Status**: ✅ Implemented (v0.1.0)

**Purpose**: Analyze and extract co-missing patterns between columns

**Features**:
- Top-k co-missing pair selection
- Single-day co-missing flags
- Rolling co-missing rates
- Per-column co-missing degree (network centrality)

**Documentation**: [SU5.md](./SU5.md)

**Key Metrics**:
- Feature count: ~50-80 columns (Stage 1)
- Code coverage: 99%
- Type safety: 100% (0 pyright errors)
- Test suite: 9 comprehensive tests

## Planned Modules

### SU1: Missing Structure Base Features

**Status**: 📝 Planned

**Purpose**: Generate basic missing value indicators and primary missing structure features

**Expected Features**:
- Missing indicators (`m/<col>`)
- Missing counts per row/column
- Missing patterns by group (M/E/I/P/S)

### SU2: Advanced Missing Features

**Status**: 📝 Planned (Deferred based on lessons learned)

**Purpose**: Extended missing value features

**Note**: May be skipped based on SU2/SU3 lessons learned (feature explosion risk)

### SU3: Time Series Missing Patterns

**Status**: ⚠️ Abandoned

**Purpose**: Temporal missing patterns

**Note**: Abandoned due to overfitting and feature explosion issues. Lessons learned incorporated into SU5 design.

### SU4: Interaction Features

**Status**: 📝 Planned

**Purpose**: Cross-feature interactions and derived features

### SU6: Domain-Specific Features

**Status**: 📝 Planned

**Purpose**: Market-specific and domain knowledge features

## Design Principles

Based on lessons learned from previous iterations, all feature generation modules follow these principles:

### 1. Feature Count Control

- **Strict limits**: Each module targets +50-100 features maximum
- **Top-k selection**: Use ranking and selection instead of generating all possible features
- **Configuration**: All limits configurable via YAML

### 2. Time Series Safety

- **No future leakage**: All features computed from past information only
- **Fold boundary reset**: Rolling statistics reset at CV fold boundaries (configurable)
- **Temporal validation**: Dedicated tests for time series correctness

### 3. Interpretability

- **Clear naming**: Descriptive feature names with consistent patterns
- **Simple logic**: Avoid overly complex transformations
- **Documentation**: Each feature type documented with examples

### 4. Type Safety and Quality

- **Strong typing**: Full type hints with pyright validation
- **Data types**: Explicit dtype specification (uint8/int16/float32)
- **Test coverage**: Minimum 95% coverage target
- **Linting**: Ruff + pyright clean

### 5. Configuration-Driven

- **YAML config**: Central configuration file (`configs/feature_generation.yaml`)
- **Easy toggling**: Enable/disable modules without code changes
- **Hyperparameter sweep**: Support for automated parameter search

## Module Dependencies

```
SU1 (Base) → SU5 (Co-Missing) → ... → Final Feature Set
              ↓
           SU4 (Interactions)
              ↓
           SU6 (Domain)
```

- **SU1**: Must be implemented first (generates base `m/<col>` indicators)
- **SU5**: Depends on SU1 output
- **SU4, SU6**: Can build on top of SU1 + SU5

## Configuration Structure

Each module has a dedicated section in `configs/feature_generation.yaml`:

```yaml
su1:
  enabled: true
  # SU1-specific settings...

su5:
  enabled: false
  base_features: su1
  top_k_pairs: 10
  windows: [5, 20]
  # More SU5 settings...

su4:
  enabled: false
  # SU4-specific settings...
```

## Testing Standards

Each feature generation module must include:

1. **Configuration Tests**
   - YAML loading
   - Default values
   - Validation

2. **Edge Case Tests**
   - Empty input
   - All missing / all observed
   - Single data point

3. **Functional Tests**
   - Feature generation correctness
   - Expected output shape
   - Correct data types

4. **Time Series Tests** (if applicable)
   - Fold boundary behavior
   - No future leakage
   - Rolling window correctness

5. **Integration Tests** (if applicable)
   - Interaction with other modules
   - Pipeline compatibility

## Development Workflow

### Adding a New Feature Module

1. **Design Phase**
   - Document module purpose and features in this README
   - Define configuration structure
   - Estimate feature count

2. **Implementation Phase**
   - Create `src/feature_generation/suN/` directory
   - Implement core transformer class
   - Add configuration loader

3. **Testing Phase**
   - Write comprehensive test suite
   - Achieve >95% coverage
   - Pass all quality checks

4. **Documentation Phase**
   - Create `docs/feature_generation/SUN.md`
   - Update this README
   - Add usage examples

5. **Integration Phase**
   - Add to pipeline
   - Run ablation studies
   - Compare with baseline

## Quality Checklist

Before merging a new feature module:

- [ ] All tests passing
- [ ] Coverage >95%
- [ ] Pyright: 0 errors
- [ ] Ruff: All checks passed
- [ ] Documentation complete
- [ ] Configuration added to YAML
- [ ] Feature count within limits
- [ ] No future leakage verified

## Benchmarking

### Performance Targets

- **SU1**: Baseline (LB 0.674 reference)
- **SU5**: ±0.0005 RMSE, non-degrading MSR
- **Future modules**: Similar constraints

### Evaluation Metrics

- **OOF RMSE**: Out-of-fold root mean squared error
- **OOF MSR**: Out-of-fold custom metric (if applicable)
- **Feature count**: Total number of generated features
- **Training time**: Time to fit and transform

### Ablation Studies

Results stored in `results/ablation/SUN/`:
- Parameter sweep results
- Feature importance rankings
- Performance comparisons

## References

- **Project Root**: `README.md`
- **Module Documentation**: `docs/feature_generation/SU*.md`
- **Configuration**: `configs/feature_generation.yaml`
- **Tests**: `tests/feature_generation/test_su*.py`

## Version History

### v0.1.0 (2025-11-22)

- Initial feature generation framework
- SU5 module implementation
- Documentation structure established

---

For module-specific documentation, see individual SU*.md files in this directory.
