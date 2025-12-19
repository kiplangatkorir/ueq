# UEQ v1.0.1 — Phoenix Edition

**UEQ** is an open-source Python library for **uncertainty estimation and quantification (UQ)** in machine learning models. The Phoenix release marks a clear shift from a research-oriented prototype to a **production-aware, extensible UQ framework**.

UEQ provides a **unified interface** for uncertainty estimation across common ML frameworks, with sensible defaults that work out of the box and advanced features for deployment and monitoring.

## Motivation

Modern ML systems increasingly operate in high-stakes, non-stationary environments. Point predictions alone are insufficient.

UEQ is designed to:

* Quantify predictive uncertainty in a principled way
* Work seamlessly across sklearn and deep learning models
* Support monitoring, drift detection, and recalibration in production
* Bridge the gap between academic UQ methods and real-world ML systems

## Key Capabilities

### Automatic Method Selection

UEQ can infer an appropriate uncertainty estimation strategy directly from the model type.

```python
from ueq import UQ

uq = UQ(model)  # Auto-detects an appropriate UQ method
```

Current auto-detection behavior:

* sklearn regressors → bootstrap-based ensembles
* sklearn classifiers → conformal prediction
* PyTorch models → Monte Carlo dropout
* Callable constructors → deep ensembles
* No model provided → Bayesian linear regression

Manual configuration is still fully supported.

### Cross-Framework Ensembles

UEQ supports ensembles composed of models from different frameworks under a single interface.

```python
models = [sklearn_model, pytorch_model, xgboost_model]
uq = UQ(models)
```

This enables:

* Unified predictive intervals
* Consistent uncertainty aggregation
* Flexible ensemble weighting strategies

### Production-Oriented Features

UEQ includes utilities designed for deployment and monitoring:

```python
from ueq import UQMonitor

monitor = UQMonitor(baseline_data=X_train)
results = monitor.monitor(X_new, y_new)
```

Key production features:

* Drift detection on predictions and uncertainty
* Rolling and batch-based evaluation
* Memory-efficient large dataset inference
* Clear separation between modeling and monitoring logic

### Performance and Scalability

* Batch-based prediction for large datasets
* Parallel execution where applicable
* Designed to integrate with REST services and containerized workflows
* Monitoring hooks suitable for real-time systems

## Quick Start

### Installation

```bash
pip install ueq
```

### Basic Usage

```python
from ueq import UQ
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
model = LinearRegression()

uq = UQ(model)
uq.fit(X, y)

predictions, intervals = uq.predict(X[:10], return_interval=True)
```

## Monitoring Example

```python
from ueq import UQ, UQMonitor

uq = UQ(model)
uq.fit(X_train, y_train)

monitor = UQMonitor(baseline_data=X_train)
results = monitor.monitor(X_test, y_test)

if results["drift_score"] > 0.1:
    print("Distribution shift detected")
```

## Project Scope

UEQ currently focuses on:

* Bootstrap-based uncertainty
* Conformal prediction
* Monte Carlo dropout
* Deep ensembles
* Monitoring and drift detection utilities

The roadmap includes adaptive conformal methods, Bayesian approximations, evidential uncertainty, and time-series support.

## Project Status & Roadmap

### Project Status

UEQ is under **active development**.

* Core APIs are stabilizing in the v1.x series
* Research and experimental modules may evolve between minor releases
* Backward compatibility is preserved where feasible, but not guaranteed for experimental features

The project intentionally balances **research velocity** with **production reliability**.

### Roadmap

#### v1.0.2 (Next Release)

Focus: **Robustness, evaluation, and production readiness**

Planned work includes:

* Standardized UQ evaluation metrics (coverage, sharpness, calibration)
* Reliability and calibration diagnostics
* Drift-aware recalibration utilities
* Uncertainty inflation under detected distribution shift
* Improved visualization of uncertainty and drift

#### v1.1.x

Focus: **Adaptive and online uncertainty**

* Adaptive and online conformal prediction
* Rolling and streaming calibration
* Time-series uncertainty support
* Expanded benchmarking infrastructure

#### v1.2.x

Focus: **Bayesian and evidential methods**

* Evidential regression and classification
* Bayesian neural networks (approximate inference)
* Laplace approximations for pretrained models

#### v2.0

Focus: **Scalability and extensibility**

* Plugin-based architecture for UQ methods
* Distributed and cloud-native execution
* Structured output uncertainty (sequences, detection)

## Documentation

* API Reference: `docs/API.md`
* Tutorials: `docs/TUTORIAL.md`
* Production Guide: `docs/PRODUCTION_GUIDE.md`
* Examples: `docs/EXAMPLES.md`

## Contributing

UEQ is actively seeking contributors.

There are no applications or interviews. If you are interested, start by picking an issue and contributing.

Please read `CONTRIBUTING.md` for guidelines.

## Community and Support

* GitHub Issues: bug reports and feature requests
* Discussions: design and research conversations
* Discord: planned once the contributor base grows

## License

MIT License

UEQ aims to make uncertainty quantification **practical, rigorous, and deployable**. Contributions and critical feedback are welcome.
