# UEQ 1.0.2

**UEQ** (Uncertainty Everywhere) is an open-source Python library for **uncertainty quantification (UQ)** in machine learning models. Its validated core is split conformal prediction for scikit-learn models: prediction intervals for regression and prediction sets for classification, plus metrics to check their coverage.

## Status

`UQ(regressor)` still auto-selects bootstrap, and **bootstrap intervals are not valid prediction intervals**: they describe uncertainty in the mean prediction, not in new outcomes. At a nominal 95%, they covered 7.7% of outcomes with `LinearRegression` and 48.2% with `RandomForestRegressor` in our tests. UEQ 1.0.2 emits a `FutureWarning` when this happens, and the default changes to split conformal in 1.1.0. Pass `method="conformal"` to get prediction intervals now, as in the quick start below.

### What is validated

| Component | Status |
|---|---|
| Split conformal regression (`method="conformal"`), residual and asymmetric (`nonconformity="quantile"`) scores | Validated: 0.906 and 0.902 coverage at a 0.90 target over 200 trials |
| Conformal classification sets (LAC, `nonconformity="inverse_probability"`), including `UQ(classifier)` | Validated: 0.899 coverage at a 0.90 target |
| `OnlineConformalUQ` with a rolling window | Works as a heuristic; no coverage guarantee under shift |
| Metrics: `coverage`, `interval_width`, `interval_score` | Correct |
| Bootstrap, deep ensembles, Bayesian linear regression | Epistemic only: not prediction intervals |
| Monitoring, recalibration and the cross-framework ensemble | Experimental: emit `ueq.ExperimentalWarning`; not validated |
| Interval ECE/MCE (`expected_calibration_error`, `maximum_calibration_error`) | Not meaningful; to be deprecated in 1.1.0 |
| `nonconformity="normalized"`, `nonconformity="margin"`, `class_conditional=True` | Disabled: raise `NotImplementedError`; use MAPIE or crepes |

Validity here means marginal coverage when calibration and test data are exchangeable. The full list of known issues is in the [CHANGELOG](https://github.com/kiplangatkorir/ueq/blob/main/CHANGELOG.md).

## Installation

```bash
pip install ueq
```

UEQ 1.0.x installs PyTorch and matplotlib as dependencies. They become optional in 1.1.0.

## Quick start

### Prediction intervals for a regressor

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from ueq import UQ, coverage, interval_width

X, y = make_regression(n_samples=3000, n_features=5, noise=10, random_state=0)
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

uq = UQ(LinearRegression(), method="conformal", alpha=0.1)  # 90% prediction intervals
uq.fit(X_train, y_train, X_calib, y_calib)
y_pred, intervals = uq.predict(X_test, return_interval=True)

print(f"coverage: {coverage(y_test, intervals):.3f}")  # close to 0.90
print(f"mean width: {interval_width(intervals):.1f}")
```

The calibration set must not be used for training. If it is too small for the requested `alpha` (fewer than about `1 / alpha` points), the intervals are infinite and UEQ warns.

### Prediction sets for a classifier

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ueq import UQ

X, y = load_iris(return_X_y=True)
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.6, random_state=0)
X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

uq = UQ(LogisticRegression(max_iter=1000), alpha=0.1)  # conformal classification
uq.fit(X_train, y_train, X_calib, y_calib)
sets = uq.predict(X_test)  # one set of class labels per row
```

The classifier needs `predict_proba`. Classifiers without it, such as the default `SVC()` or `RidgeClassifier`, raise a `ValueError`; use `SVC(probability=True)` or wrap the model in `CalibratedClassifierCV`.

## Automatic method selection

`UQ(model)` chooses a method from the model type:

* scikit-learn regressor: bootstrap (confidence interval of the mean; warns, and changes to split conformal in 1.1.0)
* scikit-learn classifier with `predict_proba`: conformal prediction sets
* PyTorch model: Monte Carlo dropout, which returns (mean, std), not intervals
* callable model constructor: deep ensemble (epistemic only)
* no model: Bayesian linear regression (coverage depends on the scale of y)

Pass `method=` to choose explicitly.

## Experimental modules

`UQMonitor`, `UQ.monitor`, `AdaptiveConformalUQ`, `DriftAwareRecalibrator`, `UncertaintyInflator` and `CrossFrameworkEnsembleUQ` emit `ueq.ExperimentalWarning`. Their output is not validated: for example, the monitor's drift score is always 0.0, and `AdaptiveConformalUQ` and `DriftAwareRecalibrator` never change the calibrated quantile. Do not rely on them.

## Roadmap

UEQ is being refocused on a narrower question: whether a deployed model's prediction intervals and sets still hold, per segment and over time, whichever library produced them. The plan, including what will be deprecated and when, is in [docs/ROADMAP.md](https://github.com/kiplangatkorir/ueq/blob/main/docs/ROADMAP.md). Changes in each release are listed in [CHANGELOG.md](https://github.com/kiplangatkorir/ueq/blob/main/CHANGELOG.md).

## Documentation

* API reference: `docs/API.md`
* Tutorial: `docs/TUTORIAL.md`
* Examples: `docs/EXAMPLES.md`
* Production guide (experimental APIs): `docs/PRODUCTION_GUIDE.md`

## Contributing

UEQ is actively seeking contributors.

There are no applications or interviews. If you are interested, start by picking an issue and contributing.

Please read `CONTRIBUTING.md` for guidelines. Changes to the statistical code need a repeated-trial coverage test and maintainer review.

## Community and Support

* GitHub Issues: bug reports and feature requests
* Discussions: design and research conversations
* Discord: planned once the contributor base grows

## License

Apache License 2.0. See `LICENSE`.
