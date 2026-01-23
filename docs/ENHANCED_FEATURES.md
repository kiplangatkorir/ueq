# UEQ v1.0.2 - Enhanced Evaluation & Visualization Features

This document describes the new features added in UEQ v1.0.2, addressing Issues #9, #10, and #17.

## Overview

UEQ v1.0.2 introduces comprehensive tools for evaluating and visualizing uncertainty quantification:

1. **Standardized UQ Evaluation Metrics** (Issue #9)
2. **Reliability & Calibration Diagnostics** (Issue #10)
3. **Enhanced Prediction Interval Visualizations** (Issue #17)

## 1. Standardized UQ Evaluation Metrics (Issue #9)

### Available Metrics

UEQ now provides 7 comprehensive metrics for evaluating uncertainty quality:

- **Coverage**: Fraction of true values within prediction intervals
- **Sharpness**: Average width of prediction intervals (tighter is better)
- **Interval Width**: Alias for sharpness
- **Expected Calibration Error (ECE)**: Measures calibration quality
- **Maximum Calibration Error (MCE)**: Worst-case calibration error
- **Interval Score**: Proper scoring rule that penalizes both width and miscoverage
- **Miscoverage Rate**: Fraction of points outside intervals (1 - coverage)

### Usage

#### Evaluate All Metrics

```python
from ueq.utils import evaluate_uncertainty

metrics = evaluate_uncertainty(
    y_true=y_test,
    y_pred=predictions,
    intervals=intervals,
    alpha=0.05,  # For 95% intervals
    n_bins=10
)

print(f"Coverage: {metrics['coverage']:.4f}")
print(f"Sharpness: {metrics['sharpness']:.4f}")
print(f"ECE: {metrics['ece']:.4f}")
```

#### Evaluate Selected Metrics

```python
metrics = evaluate_uncertainty(
    y_true=y_test,
    y_pred=predictions,
    intervals=intervals,
    metrics=['coverage', 'sharpness', 'interval_score']
)
```

#### Individual Metrics

```python
from ueq.utils import (
    coverage,
    sharpness,
    interval_score,
    expected_calibration_error
)

cov = coverage(y_true, intervals)
sharp = sharpness(intervals)
score = interval_score(y_true, intervals, alpha=0.05)
ece = expected_calibration_error(y_true, intervals, n_bins=10)
```

## 2. Reliability & Calibration Diagnostics (Issue #10)

### Automated Calibration Checking

The `check_calibration()` function provides automated diagnostics with warnings:

```python
from ueq.utils import check_calibration

diagnostics = check_calibration(
    y_true=y_test,
    intervals=intervals,
    confidence=0.95,  # Expected nominal coverage
    tolerance=0.05    # Acceptable deviation
)

print(f"Empirical Coverage: {diagnostics['empirical_coverage']:.4f}")
print(f"Well Calibrated: {diagnostics['is_well_calibrated']}")

# Check for warnings
if diagnostics['warnings']:
    for warning in diagnostics['warnings']:
        print(f"⚠️  {warning}")
```

### Detected Issues

The calibration checker automatically detects:

1. **Undercoverage**: Intervals too narrow (dangerous)
2. **Overcoverage**: Intervals too wide (conservative)
3. **Uncertainty Collapse**: Extremely narrow intervals (overconfidence)
4. **Constant Intervals**: Non-adaptive uncertainty

### Reliability Diagram

Visualize calibration with reliability diagrams:

```python
from ueq.utils import plot_reliability_diagram

plot_reliability_diagram(
    y_true=y_test,
    y_pred=predictions,
    intervals=intervals,
    n_bins=10,
    confidence=0.95,
    title="Model Reliability Diagram"
)
```

This creates a two-panel plot:
- Left: Nominal vs empirical coverage (should follow diagonal)
- Right: Calibration error by bin (should be near zero)

## 3. Enhanced Prediction Interval Visualizations (Issue #17)

### Plot Prediction Intervals

```python
from ueq.utils import plot_intervals

plot_intervals(
    x=X_test.flatten(),
    y_pred=predictions,
    lower=lower_bounds,
    upper=upper_bounds,
    y_true=y_test,  # Optional: show true values
    title="Prediction Intervals",
    xlabel="X",
    ylabel="Y",
    alpha=0.3,  # Transparency
    show_points=True  # Show prediction points
)
```

### Coverage vs Interval Width

Diagnostic plot showing relationship between interval width and coverage:

```python
from ueq.utils import plot_coverage_vs_confidence

plot_coverage_vs_confidence(
    y_true=y_test,
    intervals=intervals,
    n_bins=20,
    title="Coverage vs Interval Width"
)
```

This helps identify if wider intervals achieve better coverage (as expected).

## Complete Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from ueq import UQ
from ueq.utils import (
    evaluate_uncertainty,
    check_calibration,
    plot_intervals,
    plot_reliability_diagram
)

# Generate data
X = np.linspace(0, 10, 200).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.randn(200) * 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train UQ model
model = LinearRegression()
uq = UQ(model, method='bootstrap', n_models=50)
uq.fit(X_train, y_train)

# Get predictions
y_pred, intervals = uq.predict(X_test, return_interval=True)

# Evaluate metrics
metrics = evaluate_uncertainty(y_test, y_pred, intervals)
print(f"Coverage: {metrics['coverage']:.4f}")
print(f"Sharpness: {metrics['sharpness']:.4f}")
print(f"Interval Score: {metrics['interval_score']:.4f}")

# Check calibration
diagnostics = check_calibration(y_test, intervals, confidence=0.95)
if diagnostics['warnings']:
    for warning in diagnostics['warnings']:
        print(f"⚠️  {warning}")

# Visualize
plot_intervals(X_test.flatten(), y_pred, 
              [iv[0] for iv in intervals],
              [iv[1] for iv in intervals],
              y_true=y_test)

plot_reliability_diagram(y_test, y_pred, intervals)
```

## API Reference

### Metrics

- `coverage(y_true, intervals)` → float
- `sharpness(intervals)` → float
- `interval_width(intervals)` → float (alias for sharpness)
- `interval_score(y_true, intervals, alpha=0.05)` → float
- `miscoverage_rate(y_true, intervals)` → float
- `expected_calibration_error(y_true, intervals, n_bins=10)` → float
- `maximum_calibration_error(y_true, intervals, n_bins=10)` → float
- `evaluate_uncertainty(y_true, y_pred, intervals, metrics=None, alpha=0.05, n_bins=10)` → dict
- `check_calibration(y_true, intervals, confidence=0.95, tolerance=0.05)` → dict

### Visualizations

- `plot_intervals(x, y_pred, lower, upper, y_true=None, **kwargs)` → None
- `plot_reliability_diagram(y_true, y_pred, intervals, n_bins=10, confidence=0.95, **kwargs)` → None
- `plot_coverage_vs_confidence(y_true, intervals, n_bins=20, **kwargs)` → None

## Integration with Existing Code

All new features are backward compatible. Existing code continues to work unchanged.

```python
# Old way (still works)
from ueq.utils.metrics import coverage, sharpness
cov = coverage(y_true, intervals)

# New way (recommended)
from ueq.utils import evaluate_uncertainty
metrics = evaluate_uncertainty(y_true, y_pred, intervals)
cov = metrics['coverage']
```

## Next Steps

See `examples/demo_enhanced_evaluation.py` for a complete working example demonstrating all features.

For more information on future releases and roadmap, see the main README.md.
