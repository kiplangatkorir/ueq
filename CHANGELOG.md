# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
