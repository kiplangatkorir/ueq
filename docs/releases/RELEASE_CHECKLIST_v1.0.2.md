> Archived. This file is kept for history and does not describe the current state of UEQ. For the status of each feature, see [CHANGELOG.md](../../CHANGELOG.md).

# v1.0.2 Release Checklist

**Release Date:** January 23, 2026  
**Theme:** Production-Ready UQ Assessment - "Evaluation & Diagnostics"

## ✅ Development Complete

All development work for v1.0.2 has been completed and merged into main via PR #28.

### Implemented Features

#### 1. Standardized UQ Evaluation Metrics (Issue #9)
- ✅ `interval_score()` - Proper scoring rule for prediction intervals
- ✅ `miscoverage_rate()` - Fraction of points outside intervals  
- ✅ `interval_width()` - Clearer alias for sharpness
- ✅ `evaluate_uncertainty()` - Unified evaluation function
- ✅ 7 comprehensive metrics total
- ✅ Complete API documentation
- ✅ 12 unit tests

#### 2. Reliability & Calibration Diagnostics (Issue #10)
- ✅ `check_calibration()` - Automated calibration checking
- ✅ Intelligent warnings for:
  - Undercoverage (dangerous - intervals too narrow)
  - Overcoverage (too conservative)
  - Uncertainty collapse (overconfident)
  - Constant intervals (non-adaptive)
- ✅ `plot_reliability_diagram()` - Dual-panel calibration assessment
- ✅ `plot_coverage_vs_confidence()` - Diagnostic plots
- ✅ 11 unit tests for visualizations

#### 3. Enhanced Prediction Interval Visualizations (Issue #17)
- ✅ `plot_intervals()` - Customizable interval plots
- ✅ True values overlay support
- ✅ Publication-quality defaults
- ✅ Works with all UQ methods (bootstrap, conformal, ensembles, etc.)
- ✅ Example gallery in documentation

### Quality Assurance
- ✅ **40 total tests passing** (23 new tests for v1.0.2)
- ✅ **100% coverage** of new features
- ✅ **CodeQL**: 0 security alerts
- ✅ **Demo**: examples/demo_enhanced_evaluation.py runs successfully
- ✅ **Backward compatible**: No breaking changes

### Documentation
- ✅ `CHANGELOG.md` updated with v1.0.2 changes
- ✅ `RELEASE_NOTES_v1.0.2.md` created with complete release notes
- ✅ `docs/ENHANCED_FEATURES.md` with full API reference
- ✅ `examples/demo_enhanced_evaluation.py` working example
- ✅ `setup.py` version bumped to 1.0.2

## 📦 Release Process (Maintainer Actions Required)

### Step 1: Verify Code Quality
```bash
# Clone/pull latest main branch
git checkout main
git pull origin main

# Verify all tests pass
pip install -e .
python -m pytest tests/ -v

# Run the demo
python examples/demo_enhanced_evaluation.py

# Verify no security issues
# (CodeQL already passed in PR #28)
```

### Step 2: Create GitHub Release

1. Go to https://github.com/kiplangatkorir/ueq/releases/new
2. Tag version: `v1.0.2`
3. Target: `main` branch
4. Release title: `v1.0.2 - Evaluation & Diagnostics`
5. Description: Copy content from `RELEASE_NOTES_v1.0.2.md`
6. Add this footer to reference issues:

```markdown
## Issues Resolved
Closes #9
Closes #10  
Closes #17
```

7. Click "Publish release"

### Step 3: Publish to PyPI

```bash
# Ensure you're on main with latest changes
git checkout main
git pull

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI (requires PyPI credentials)
twine upload dist/*
```

### Step 4: Verify Publication

```bash
# Test installation from PyPI
pip install --upgrade ueq

# Verify version
python -c "import ueq; print(ueq.__version__)"
# Should print: 1.0.2

# Test a quick example
python -c "
from ueq import UQ
from ueq.utils import evaluate_uncertainty
print('✓ v1.0.2 successfully installed from PyPI')
"
```

### Step 5: Close Issues

Issues #9, #10, and #17 should automatically close when the release is published with "Closes #X" in the release notes. If they don't auto-close:

1. Go to each issue:
   - https://github.com/kiplangatkorir/ueq/issues/9
   - https://github.com/kiplangatkorir/ueq/issues/10
   - https://github.com/kiplangatkorir/ueq/issues/17

2. Add a comment referencing the release:
   ```
   Fixed in v1.0.2: https://github.com/kiplangatkorir/ueq/releases/tag/v1.0.2
   ```

3. Close the issue

### Step 6: Announce Release

Consider announcing on:
- GitHub Discussions
- Project README (update installation instructions)
- Any relevant communities or mailing lists

## 📊 Release Statistics

- **Features Added**: 3 major feature sets (Issues #9, #10, #17)
- **New Functions**: 10+ new functions
- **Tests Added**: 23 new unit tests (40 total)
- **Lines of Code**: ~1,700 additions
- **Files Changed**: 12 files
- **Breaking Changes**: 0 (fully backward compatible)
- **Security Issues**: 0

## 🎯 Post-Release

After v1.0.2 is published:

1. Update project roadmap
2. Consider next release focus (v1.1.0 roadmap already outlined in RELEASE_NOTES_v1.0.2.md)
3. Monitor for any issues reported by users
4. Update documentation website (if applicable)

## 📝 Notes

- All development work is complete and merged
- No code changes needed
- Repository is ready for immediate release
- This is a minor version bump with new features
- Fully backward compatible with v1.0.1

---

**Status**: ✅ READY FOR RELEASE

**Prepared by**: GitHub Copilot Agent  
**Date**: January 23, 2026
