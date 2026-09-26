# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with the versioning policy stated under 1.0.2.

## [1.0.2] - Unreleased

This release publishes everything merged since 1.0.1, fixes outputs that were wrong, and states the real status of each feature. PyPI has served 1.0.1 since 2025-09-29; nothing listed under "Added" has been on PyPI before. The measurements behind each status are in section 1 of [docs/ROADMAP.md](docs/ROADMAP.md).

Status labels used below:

- **Stable**: behaves as documented and is covered by tests.
- **Experimental**: runs, but its output is unvalidated or known to be partly wrong. Do not rely on it. Items marked "warns" emit `ueq.ExperimentalWarning`.
- **Disabled**: raises `NotImplementedError`.

### Added

#### From PR #28 (issues #9, #10, #17)

- Issue #9, interval metrics:
  - `interval_score()` (the Winkler interval score), `interval_width()` and `miscoverage_rate()`. **Stable.**
  - `evaluate_uncertainty()`, which computes several metrics in one call. **Stable** for `coverage`, `sharpness`, `interval_width`, `interval_score` and `miscoverage`. Its `ece` and `mce` entries, computed by default, come from the interval `expected_calibration_error()` and `maximum_calibration_error()` that have shipped since 0.1.0. These bin intervals by their position in the array, not by any predicted quantity, so the value is not meaningful: exactly calibrated intervals score 0.41. Ignore these entries; they will be deprecated in 1.1.0.
- Issue #10, `check_calibration()`. **Experimental.** Its coverage check is correct, but it flags valid split conformal intervals as miscalibrated because their width is constant.
- Issues #10 and #17, plots:
  - `plot_intervals()`. **Stable.**
  - `plot_coverage_vs_confidence()`, which plots coverage by interval-width bin. **Stable.**
  - `plot_reliability_diagram()`. **Experimental.** It rescales each interval as if its width grew linearly with the nominal level, which does not hold in general. `plot_calibration_curve()` (from 0.1.0, completed in this PR) makes the same assumption.
  - The plots call `plt.show()` and return nothing. The older `ueq.utils.plotting.plot_intervals` is still exported as `ueq.utils.plot_intervals_old`.
- `docs/ENHANCED_FEATURES.md`, `examples/demo_enhanced_evaluation.py` and 23 tests.

#### From PR #30 (issues #5, #6, #8, #11, #12, #15, #18, #22, #27)

- Issue #5, online conformal prediction in `ueq.methods`:
  - `OnlineConformalUQ` with a rolling window. **Stable, as a heuristic**: there is no coverage guarantee under shift. After a noise shift that dropped static coverage to 0.416, a 500-point rolling window recovered to 0.869. Symmetric scores only, no prefit mode and no API for delayed labels. When feedback arrives in bursts (about 250 scores per failed engine on NASA C-MAPSS data), the default window of 500 holds only about two units, and interval widths grew from 67 to 78-88 cycles.
  - `AdaptiveConformalUQ`. **Experimental (warns).** Its recalibration does nothing: the quantile stays identical to the plain rolling window while `recalibration_count` increases.
- Issue #6, nonconformity scores for `ConformalUQ`:
  - `nonconformity="quantile"`, an asymmetric signed-residual score (not conformalized quantile regression). **Stable**: 0.902 coverage at a 0.90 target over 200 trials.
  - `nonconformity="normalized"`. **Disabled**: 71%, 19% and 4% coverage at a 0.90 target in three probes.
  - `nonconformity="margin"` for classification. **Disabled**: it returned every class for every input.
- Issue #8, class-conditional (Mondrian) calibration, `class_conditional=True`. **Disabled**: minority-class coverage was 22-61% at a 0.90 target. Use MAPIE or crepes for class-conditional sets.
- Issue #11, `DriftAwareRecalibrator`. **Experimental (warns).** Its drift score stayed at 0.0 under a 3-sigma shift while coverage fell to 0.47, and recalibration never changes the quantile.
- Issue #12, `UncertaintyInflator`. **Experimental (warns).** It widens intervals by a heuristic factor, with no coverage guarantee.
- Issue #15, synthetic benchmark generators `make_synthetic_regression()`, `make_heteroscedastic_data()`, `make_concept_drift_data()` and `make_covariate_shift_data()`. **Stable** with the fixes in this release. As merged, they reseeded the global NumPy random state, and `shift="covariate"` shifted X after y had been generated, which is a concept shift rather than a covariate shift.
- Issue #18, `plot_uncertainty_timeline()`. **Experimental.** It draws fixed 0.90 and 0.95 coverage lines whatever alpha was used, and assumes the uncertainty values are symmetric half-widths.
- Issue #22, contribution templates in `templates/` (documentation).
- Issue #27, `TESTING_GUIDE.md` and `COMMUNITY_GUIDELINES.md` (documentation). The test examples in `TESTING_GUIDE.md` use a single seed and a fixed band; the statistical review gate in `CONTRIBUTING.md` takes precedence.

#### From PR #31 (issue #9)

- `ueq.evaluate`, an alias for `evaluate_uncertainty()`. **Stable.** Note that `ueq.utils.evaluate` is a different, older function that fits a model and then evaluates it; 1.1.0 will rename one of the two and keep a deprecated alias.
- A `"calibration"` metric key in `evaluate_uncertainty()`, an alias for `"ece"`. It inherits the interval ECE problem described above, so its value is not meaningful.

#### From PR #32 (issue #10)

- New `ueq.diagnostics` module, with `examples/diagnostics_example.py` and tests:
  - `check_classification_calibration()` and `plot_calibration(task="classification")`, based on top-label ECE. **Stable.**
  - `check_regression_calibration()`. **Experimental.** The coverage figure is correct, but constant-width intervals, such as valid split conformal intervals, get a "CONSTANT INTERVALS" warning and `is_well_calibrated=False`.
  - `plot_calibration(task="regression")`. **Experimental.** It has the same constant-width false alarm, and its reliability panel rescales widths linearly with the nominal level.

### Fixed

- `UQ(classifier)` now detects classifiers with `sklearn.base.is_classifier` and uses conformal classification (LAC / `inverse_probability`); classifiers without `predict_proba` (e.g. default `SVC()`, `RidgeClassifier`) raise a clear `ValueError`.
- Conformal classification maps labels through the model's `classes_`, so string labels and labels like {-1, +1} or {1, 2, 3} work; prediction sets contain class labels, not column indices.
- Split conformal returns infinite bounds (regression) or the full label set (classification) when the calibration set is too small for the requested alpha, with a warning, instead of silently clamping to the largest score. Same for `OnlineConformalUQ` until its buffer is large enough.
- `ConformalUQ` raises on multi-output y, and accepts an `(n, 1)` column-vector target.
- `ConformalUQ(model, task_type="classification")` works again without naming a score: the default nonconformity score now follows the task (`residual` for regression, `inverse_probability` for classification). Between 1.0.1 and this release it raised `ValueError`.
- The conformal rank `ceil((1 - alpha)(n + 1))` is computed with a tolerance for floating-point error, so a calibration set of exactly the minimum size (for example n = 9 at alpha = 0.1) keeps a finite quantile.
- Synthetic generators use `np.random.default_rng` and no longer reseed the global NumPy RNG; `shift="covariate"` shifts X before y is generated. `make_synthetic_regression()` now returns the noise type in `meta["noise_type"]`; it returned the noise array.
- Removed unused torch imports from `ueq/core.py` and `ueq/methods/cross_ensemble.py`; `print()` and bare `except:` removed from library code.
- `plot_calibration_curve()` was incomplete in 1.0.1: `ueq/utils/visualization.py` stopped mid-function at line 75, and the curve shrank widths by `1 - q` instead of scaling them by `q` (PR #28; see the caveat under `plot_reliability_diagram()` above).

In 1.0.1, `UQ(classifier)` ran conformal prediction in regression mode and returned the predicted label plus or minus a number, and {-1, +1} labels gave trivial sets. That output had no valid use, so under the versioning policy below the classifier and label fixes are not treated as breaking changes.

### Changed

- `nonconformity="normalized"`, `nonconformity="margin"` and `class_conditional=True` raise `NotImplementedError` (never released on PyPI; they gave invalid coverage) and point to MAPIE / crepes.
- `FutureWarning` when `UQ(model)` auto-selects bootstrap for a regressor (confidence interval of the mean, not a prediction interval; default changes to split conformal in 1.1.0). `UserWarning` when bootstrap, deep-ensemble or Bayesian-linear intervals are requested explicitly. `UQ.predict` warns that MC dropout returns (mean, std), not intervals.
- New `ueq.ExperimentalWarning`, emitted by `AdaptiveConformalUQ`, `DriftAwareRecalibrator`, `UncertaintyInflator`, `UQMonitor`, `UQ.monitor` and `CrossFrameworkEnsembleUQ`.

### Tests and project

- New tests: classifier routing and labels, finite-sample edge cases, online conformal recovery after a shift (statistical), synthetic generators, and strict-xfail specs for the recalibration modules.
- An earlier version of this entry said 40 tests; the suite had 76 before the changes above. The final count will be stated at release.
- Continuous integration on GitHub Actions: pytest on Python 3.10 to 3.14, ruff checks for bare `except:` (E722) and `print()` (T201) in `ueq/`, and a smoke run of the examples.
- Releases are built and published to PyPI from a version tag by `.github/workflows/release.yml`, which fails if the tag does not match the package version.
- `.github/CODEOWNERS` and a statistical review gate in `CONTRIBUTING.md`: changes to `ueq/methods/`, the metrics or the diagnostics need a repeated-trial coverage test and maintainer sign-off.
- README: corrected the version header and the licence (Apache-2.0, as in `LICENSE`; the README said MIT), added a status note and a table of what is validated, and removed examples that did not run. The `setup.py` description and keywords were rewritten.
- Per-PR summary files and earlier release notes moved to `docs/releases/`.

### Known issues

- **Bootstrap intervals are not prediction intervals.** Bootstrap is still what `UQ(model)` selects for every scikit-learn regressor. It estimates uncertainty in the mean prediction, not in new outcomes. At a nominal 95%, its intervals covered 7.7% of outcomes with `LinearRegression` and 48.2% with `RandomForestRegressor` (a second probe gave 10.6% and 56%). On NASA C-MAPSS engine data at a nominal 90% they covered about 22-36%, depending on the setup. 1.0.2 emits a `FutureWarning`; the default switches to split conformal in 1.1.0. Pass `method="conformal"` for prediction intervals now.
- **Deep ensembles are epistemic only.** `DeepEnsembleUQ` intervals covered 7% of outcomes at a nominal 95%.
- **Bayesian linear coverage depends on the scale of y.** `BayesianLinearUQ` fixes the noise precision (`beta`), so its coverage ranged from 0.7% to 100% depending on the scale of y. Its `alpha` argument is the prior precision, not a miscoverage level.
- **MC dropout returns (mean, std), not intervals.** `UQ.predict` now warns.
- **Monitoring does not detect drift.** `UQMonitor` and `UQ.monitor` ignore the baseline given to the constructor, so the drift score is always 0.0, and they never see y. Experimental (warns).
- **`CrossFrameworkEnsembleUQ` drops members** that fail to fit or predict, for example a PyTorch model given scikit-learn-style arguments, and carries on with the rest, now with a warning. Experimental (warns).
- **Interval ECE and MCE are not meaningful**, including the `ece`, `mce` and `calibration` entries of `evaluate_uncertainty()`. They will be deprecated in 1.1.0.
- **Constant-width false alarm.** `check_calibration()`, `check_regression_calibration()` and `plot_calibration(task="regression")` report valid split conformal intervals as miscalibrated.
- **Inconsistent `alpha`.** Conformal methods take `alpha` at construction, bootstrap at predict time, so `UQ(regressor, alpha=0.1)` raises `TypeError`. One convention comes in 1.1.0.
- **pandas input.** Bootstrap indexes `X[idx]`, which fails on DataFrames; pass NumPy arrays.
- **Dependencies.** `torch` and `matplotlib` are hard dependencies, so `import ueq` needs PyTorch even for scikit-learn users. `setup.py` still lists Python 3.8 and 3.9, while CI tests 3.10 to 3.14. Both change in 1.1.0.

### Versioning policy

UEQ follows Semantic Versioning with these rules:

- Released APIs are deprecated with a warning for at least one minor release, and removed only in a major release.
- Options that were never released on PyPI can be disabled or removed at any time. This is why `normalized`, `margin` and `class_conditional=True` raise in 1.0.2 instead of going through a deprecation period.
- Fixes to output that is numerically wrong, such as classifier routing, are not treated as breaking changes.

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
