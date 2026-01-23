# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-01-23 - "Evaluation & Diagnostics" 📊
**"Production-Ready UQ Assessment"**

### Added
- **Standardized UQ evaluation metrics** (Issue #9) - Complete suite of 7 metrics for uncertainty quality assessment:
  - `interval_score()` - Proper scoring rule for prediction intervals
  - `interval_width()` - Alias for sharpness for clarity
  - `miscoverage_rate()` - Complement of coverage
  - `evaluate_uncertainty()` - Unified function to compute all metrics at once
- **Reliability and calibration diagnostics** (Issue #10):
  - `check_calibration()` - Automated calibration checking with warnings
  - Detects undercoverage, overcoverage, uncertainty collapse, and constant intervals
  - `plot_reliability_diagram()` - Visual calibration assessment with dual plots
  - `plot_coverage_vs_confidence()` - Diagnostic plot for interval width vs coverage
- **Enhanced prediction interval visualizations** (Issue #17):
  - `plot_intervals()` - Comprehensive interval plotting with customization
  - Support for true values overlay and custom styling
  - Completed and fixed incomplete `visualization.py` file
- **Comprehensive documentation**:
  - New `ENHANCED_FEATURES.md` guide with complete API reference
  - Example notebook demonstrating all new features
  - 23 new unit tests (100% coverage of new features)

### Changed
- Enhanced `ueq.utils` module to export all new functions
- Main package `__init__.py` now exports visualization and diagnostic functions
- Improved error messages in metric evaluation

### Fixed
- Completed incomplete `visualization.py` (was cut off at line 75)
- Fixed calibration curve function that was broken

### Technical Details
- Added type hints to metrics module
- All new functions include comprehensive docstrings
- Total of 40 tests now passing (up from 17)
- Backward compatible - all existing code continues to work

## [1.0.1] - 2025-09-29 - "Phoenix" 🔥
**"Rising from Research to Production"**

### Added
- **Auto-detection system** - Automatically detects model types and selects optimal UQ methods
- **Cross-framework ensembles** - Combine models from different frameworks (sklearn + PyTorch) in unified uncertainty estimates
- **Smart method selection** - `UQ(model)` now auto-selects the best method based on model type:
  - sklearn regressors → bootstrap
  - sklearn classifiers → conformal prediction
  - PyTorch models → MC dropout
  - constructor functions → deep ensembles
  - no model → Bayesian linear regression
- **Enhanced UQ class** with new `get_info()` method for debugging and introspection
- **Cross-framework ensemble class** (`CrossFrameworkEnsembleUQ`) with weighted aggregation
- **Comprehensive examples** showcasing new auto-detection and cross-framework capabilities
- **Backward compatibility** - All existing code continues to work unchanged

### Changed
- Default `method` parameter changed from `"bootstrap"` to `"auto"` for intelligent method selection
- Improved error messages and user feedback
- Enhanced documentation with auto-detection examples

### Fixed
- Fixed MC Dropout constructor handling in `UQ` class
- Fixed parameter name inconsistency (`n_samples` → `n_models` for bootstrap)
- Fixed README examples to match current API
- Fixed visualization plotting functions

### Technical Details
- Added `_detect_model_type()` method for automatic framework detection
- Added `_auto_select_method()` method for intelligent method selection
- Added `CrossFrameworkEnsembleUQ` class for multi-framework ensembles
- Enhanced `get_info()` method with detailed model information
- Improved error handling and validation

## [0.1.0] - 2025-09-26

### Added
- Initial release of Uncertainty Everywhere (UEQ)
- Core UQ methods: Bootstrap, Conformal Prediction, MC Dropout, Deep Ensembles, Bayesian Linear Regression
- Unified `UQ` class interface
- Comprehensive metrics: coverage, sharpness, ECE, MCE
- Visualization utilities for uncertainty plots
- Example demos and benchmarks
- Full test suite
- Documentation and README

### Features
- Support for scikit-learn models
- Support for PyTorch models
- Multiple uncertainty quantification methods
- Production-ready metrics and evaluation
- Extensible architecture for new UQ methods
