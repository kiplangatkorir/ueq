> Archived. This file is kept for history and does not describe the current state of UEQ. For the status of each feature, see [CHANGELOG.md](../../CHANGELOG.md).

# UEQ v1.0.2 Release Notes - "Evaluation & Diagnostics" 📊

**Release Date:** January 23, 2026

**Theme:** Production-Ready UQ Assessment

---

## Overview

Version 1.0.2 completes the production readiness vision started in the Phoenix release by adding comprehensive tools for evaluating and diagnosing uncertainty quantification. This release directly addresses three high-priority community requests:

- **Issue #9**: Standardized UQ Evaluation Metrics
- **Issue #10**: Reliability & Calibration Diagnostics  
- **Issue #17**: Prediction Interval & Uncertainty Visualizations

## What's New

### 🎯 Standardized UQ Evaluation Metrics

A complete suite of metrics for assessing uncertainty quality:

```python
from ueq.utils import evaluate_uncertainty

metrics = evaluate_uncertainty(y_true, y_pred, intervals)
# Returns: coverage, sharpness, ECE, MCE, interval_score, miscoverage
```

**New metrics:**
- `interval_score()` - Proper scoring rule that penalizes both width and miscoverage
- `miscoverage_rate()` - Fraction of points outside intervals
- `interval_width()` - Clearer alias for sharpness
- `evaluate_uncertainty()` - Unified evaluation function

### 🔍 Automated Calibration Diagnostics

Intelligent calibration checking with automatic warnings:

```python
from ueq.utils import check_calibration

diagnostics = check_calibration(y_true, intervals, confidence=0.95)

if not diagnostics['is_well_calibrated']:
    for warning in diagnostics['warnings']:
        print(f"⚠️  {warning}")
```

**Detects:**
- ⚠️ **Undercoverage** - Intervals too narrow (dangerous)
- ⚠️ **Overcoverage** - Intervals too conservative  
- ⚠️ **Uncertainty Collapse** - Overconfident predictions
- ⚠️ **Constant Intervals** - Non-adaptive uncertainty

### 📊 Enhanced Visualizations

Publication-quality plots for uncertainty analysis:

```python
from ueq.utils import plot_intervals, plot_reliability_diagram

# Prediction intervals with true values
plot_intervals(x, y_pred, lower, upper, y_true=y_test)

# Reliability diagram (calibration curve)
plot_reliability_diagram(y_true, y_pred, intervals)
```

**New visualizations:**
- `plot_intervals()` - Customizable interval plots
- `plot_reliability_diagram()` - Dual-panel calibration assessment
- `plot_coverage_vs_confidence()` - Width vs coverage diagnostic

### ✅ Quality & Testing

- **23 new unit tests** added (all passing)
- **40 total tests** now in suite
- **100% coverage** of new features
- Complete documentation and examples

## Breaking Changes

**None.** This release is fully backward compatible. All existing code continues to work unchanged.

## Migration Guide

No migration needed. New features are additions, not replacements.

### Recommended Updates

```python
# Old way (still works)
from ueq.utils.metrics import coverage, sharpness
cov = coverage(y_true, intervals)
sharp = sharpness(intervals)

# New way (recommended)
from ueq.utils import evaluate_uncertainty
metrics = evaluate_uncertainty(y_true, y_pred, intervals)
cov = metrics['coverage']
sharp = metrics['sharpness']
```

## Installation

```bash
pip install --upgrade ueq
```

Or from source:
```bash
pip install git+https://github.com/kiplangatkorir/ueq.git@v1.0.2
```

## Examples

See the complete working example:
- `examples/demo_enhanced_evaluation.py`

Documentation:
- `docs/ENHANCED_FEATURES.md` - Complete feature guide
- `CHANGELOG.md` - Detailed changes

## What's Next: v1.1.0 Roadmap

Focus on **adaptive and online uncertainty**:

- Adaptive and online conformal prediction (Issue #5)
- Drift-aware recalibration (Issue #11)
- Uncertainty inflation under drift (Issue #12)
- Time-series conformal prediction (Issue #23)
- Class-conditional conformal (Issue #8)

## Community

We're actively seeking contributors! No applications or interviews needed.

**How to contribute:**
1. Pick an open issue
2. Comment that you're working on it
3. Submit a PR

See `CONTRIBUTING.md` for guidelines.

## Acknowledgments

This release addresses key issues raised by the UEQ community:
- Issue #9 by community request
- Issue #10 by community request  
- Issue #17 marked as "good first issue"

Thank you to everyone providing feedback and testing!

## Links

- **GitHub Repository**: https://github.com/kiplangatkorir/ueq
- **Issues**: https://github.com/kiplangatkorir/ueq/issues
- **PyPI**: https://pypi.org/project/ueq/

---

**UEQ aims to make uncertainty quantification practical, rigorous, and deployable.**

Contributions and critical feedback are welcome!
