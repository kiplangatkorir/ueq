# UEQ v1.0.1 Release Notes

**Release Date:** September 29, 2025

## 🚀 Major New Features

### Auto-Detection System
UEQ now automatically detects your model type and selects the optimal uncertainty quantification method:

- **sklearn regressors** → Bootstrap ensembles
- **sklearn classifiers** → Conformal prediction  
- **PyTorch models** → MC Dropout
- **Constructor functions** → Deep ensembles
- **No model** → Bayesian linear regression

```python
# Zero configuration - just works!
from ueq import UQ
uq = UQ(your_model)  # Auto-detects and selects best method
```

### Cross-Framework Ensembles
Combine models from different frameworks in unified uncertainty estimates:

```python
# Mix sklearn + PyTorch models
models = [sklearn_model, pytorch_model, xgboost_model]
uq = UQ(models)  # Creates cross-framework ensemble
```

## 🔧 Enhanced Features

- **Smart method selection** based on model type
- **Weighted ensemble aggregation** for cross-framework models
- **Enhanced error handling** with better user feedback
- **Backward compatibility** - all existing code works unchanged
- **Rich introspection** with `get_info()` method

## 🐛 Bug Fixes

- Fixed MC Dropout constructor handling
- Fixed parameter name inconsistencies (`n_samples` → `n_models`)
- Fixed README examples to match current API
- Fixed conformal prediction return values
- Fixed visualization plotting functions

## 📚 Documentation Updates

- Updated README with auto-detection examples
- Added cross-framework ensemble documentation
- Enhanced code examples and tutorials
- Created comprehensive changelog

## 🧪 Testing

- All existing tests pass
- Added tests for auto-detection functionality
- Added tests for cross-framework ensembles
- Improved test coverage and reliability

## 🔄 Migration Guide

**No breaking changes!** All existing code continues to work:

```python
# Old way (still works)
uq = UQ(model, method="bootstrap")

# New way (auto-detection)
uq = UQ(model)  # Automatically selects bootstrap for sklearn regressors
```

## 🎯 What's Next

- TensorFlow/Keras support
- XGBoost/LightGBM integration
- Advanced ensemble methods
- Production deployment features
- Performance optimizations

---

**Full Changelog:** https://github.com/kiplangatkorir/ueq/blob/main/CHANGELOG.md
