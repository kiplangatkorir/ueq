# UEQ Documentation

Documentation for **UEQ (Uncertainty Everywhere)**, a Python library for uncertainty quantification of machine learning models.

Before relying on a method, check its status in the table below or in the [main README](../README.md). In 1.0.2, split conformal prediction is the validated core. Bootstrap, deep ensembles and Bayesian linear regression give model (epistemic) uncertainty only, and the monitoring, recalibration and cross-framework APIs are experimental.

## Documents

| Document | Description | Audience |
|----------|-------------|----------|
| **[API.md](API.md)** | API reference for classes, methods and parameters | Developers, API users |
| **[TUTORIAL.md](TUTORIAL.md)** | Step-by-step tutorials | Beginners |
| **[EXAMPLES.md](EXAMPLES.md)** | Walkthroughs of the scripts in `examples/` | All users |
| **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)** | Deployment, monitoring and scaling patterns (experimental APIs) | ML engineers |
| **[ROADMAP.md](ROADMAP.md)** | Proposed development roadmap (Sept 2026 to Sept 2027): current status, real-world positioning, phased plan, issue triage | Maintainers, contributors |
| **[USE_CASES.md](USE_CASES.md)** | Research behind the roadmap: eight real-world domains, verified datasets, competitor landscape | Maintainers, contributors |
| **[releases/](releases/)** | Archived release notes and per-PR summaries | Maintainers |

## Installation

```bash
pip install ueq
```

PyPI currently serves 1.0.1. To use the fixes described in the [CHANGELOG](../CHANGELOG.md) before 1.0.2 is published, install from source:

```bash
git clone https://github.com/kiplangatkorir/ueq.git
cd ueq
pip install -e .
```

## Quick start

Split conformal prediction gives intervals that cover new outcomes at the requested rate when calibration and test data are exchangeable. It needs a separate calibration set.

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from ueq import UQ, coverage

X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

uq = UQ(LinearRegression(), method="conformal", alpha=0.1)  # 90% intervals
uq.fit(X_train, y_train, X_calib, y_calib)
predictions, intervals = uq.predict(X_test, return_interval=True)

print(f"Coverage: {coverage(y_test, intervals):.3f} (target 0.90)")
```

For a classifier with `predict_proba`, `UQ(classifier)` returns prediction sets of class labels with the same `fit` signature.

## Methods and their status

| Method | Framework | What the output means | Status in 1.0.2 |
|--------|-----------|-----------------------|-----------------|
| **Split conformal** (`method="conformal"`) | scikit-learn | Prediction intervals or label sets with marginal coverage | Validated |
| **Online conformal** (`OnlineConformalUQ`) | scikit-learn | Conformal quantile over a rolling window | Works as a heuristic; no guarantee under shift |
| **Bootstrap** | scikit-learn | Interval for the mean prediction, not for new outcomes | Epistemic only; still the auto default for regressors, changing in 1.1.0 |
| **Deep ensemble** | PyTorch | Spread of the members' mean predictions | Epistemic only |
| **MC dropout** | PyTorch | Mean and standard deviation, not intervals | Epistemic only |
| **Bayesian linear** | NumPy | Gaussian interval with a fixed noise precision | Coverage depends on the scale of y |
| **Cross-framework ensemble** | Any | Aggregated member intervals | Experimental |

## Metrics

```python
from ueq import coverage, interval_width, interval_score

cov = coverage(y_test, intervals)                       # fraction of outcomes inside
width = interval_width(intervals)                       # mean width; lower is sharper
score = interval_score(y_test, intervals, alpha=0.1)    # Winkler score; lower is better
```

The interval `expected_calibration_error` and `maximum_calibration_error` bin intervals by their position in the array, so their values are not meaningful; they will be deprecated in 1.1.0. For classifiers, use `check_classification_calibration` from `ueq.diagnostics`.

## Research papers

- Conformal prediction: [Vovk, Gammerman and Shafer, 2005](https://link.springer.com/book/10.1007/978-3-319-04013-4)
- Bootstrap: Efron and Tibshirani, *An Introduction to the Bootstrap*, 1993
- MC dropout: [Gal and Ghahramani, 2016](https://arxiv.org/abs/1506.02142)
- Deep ensembles: [Lakshminarayanan et al., 2017](https://arxiv.org/abs/1612.01474)

## Contributing and support

See [CONTRIBUTING.md](../CONTRIBUTING.md), including the statistical review gate for changes to methods and metrics. Report bugs on [GitHub Issues](https://github.com/kiplangatkorir/ueq/issues) with your Python and UEQ versions (`ueq.__version__`), steps to reproduce, and any error messages.

## License

Apache License 2.0; see [LICENSE](../LICENSE).
