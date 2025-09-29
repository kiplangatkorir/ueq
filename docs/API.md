# UEQ API Documentation

## Overview

Uncertainty Everywhere (UEQ) is a unified Python library for Uncertainty Quantification (UQ) that provides a single interface for multiple UQ methods across different machine learning frameworks.

## Core Classes

### UQ Class

The main interface for uncertainty quantification.

```python
from ueq import UQ

uq = UQ(model=None, method="auto", **kwargs)
```

#### Parameters

- **model** (object or callable, optional): 
  - For sklearn-style models: pass the model instance (Bootstrap, Conformal)
  - For deep learning ensembles: pass a model constructor (e.g., `lambda: Net()`)
  - For PyTorch models: pass the model instance (MC Dropout)
  - Not required for Bayesian Linear Regression
  - For cross-framework ensembles: pass a list of models

- **method** (str, default="auto"): 
  - `"auto"`: Automatically selects the best method based on model type
  - `"bootstrap"`: Bootstrap ensembles for sklearn models
  - `"conformal"`: Conformal prediction for distribution-free coverage
  - `"mc_dropout"`: Monte Carlo Dropout for PyTorch models
  - `"deep_ensemble"`: Deep ensembles for PyTorch models
  - `"bayesian_linear"`: Bayesian linear regression
  - `"cross_ensemble"`: Cross-framework ensembles (auto-detected)

- **\*\*kwargs**: Additional arguments passed to the chosen method

#### Methods

##### `fit(*args, **kwargs)`
Fit the underlying model(s).

**Parameters:**
- For sklearn models: `fit(X, y)`
- For PyTorch models: `fit(train_loader, criterion, optimizer, epochs)`
- For conformal prediction: `fit(X_train, y_train, X_calib, y_calib)`
- For cross-framework ensembles: `fit(X, y)`

**Returns:** `self`

##### `predict(X, return_interval=True, **kwargs)`
Predict with uncertainty estimates.

**Parameters:**
- **X** (array-like): Test inputs
- **return_interval** (bool, default=True): Whether to return prediction intervals
- **\*\*kwargs**: Additional arguments for specific methods

**Returns:**
- If `return_interval=True`: `(mean_pred, intervals)`
- If `return_interval=False`: `mean_pred`

##### `predict_large_dataset(X, batch_size=1000, return_interval=True)`
Memory-efficient prediction for large datasets.

**Parameters:**
- **X** (array-like): Input data
- **batch_size** (int, default=1000): Batch size for processing
- **return_interval** (bool, default=True): Whether to return uncertainty intervals

**Returns:** Same as `predict()`

##### `monitor(X, y=None, baseline_data=None, baseline_uncertainty=None)`
Monitor model performance and uncertainty drift.

**Parameters:**
- **X** (array-like): Input data
- **y** (array-like, optional): True target values
- **baseline_data** (array-like, optional): Baseline data for drift detection
- **baseline_uncertainty** (array-like, optional): Baseline uncertainty estimates

**Returns:**
```python
{
    'drift_score': float,
    'alerts': list,
    'current_stats': dict,
    'baseline_stats': dict,
    'history_size': int
}
```

##### `benchmark(X, y=None, methods=None)`
Benchmark model performance.

**Parameters:**
- **X** (array-like): Input data
- **y** (array-like, optional): True target values
- **methods** (dict, optional): Additional methods to benchmark

**Returns:**
```python
{
    'method_name': {
        'result': array,
        'timing': dict,
        'success': bool
    }
}
```

##### `get_info()`
Get information about the detected model type and selected method.

**Returns:**
```python
{
    'model_type': str,
    'method': str,
    'model_class': str,
    # Additional info for cross-framework ensembles
    'n_models': int,
    'model_classes': list,
    'ensemble_info': dict
}
```

##### `predict_dist(X, **kwargs)`
Return predictive distribution (if available).

**Parameters:**
- **X** (array-like): Input data
- **\*\*kwargs**: Additional arguments

**Returns:** Predictive distribution samples

##### `calibrate(*args, **kwargs)`
Optional calibration step (for conformal methods, etc.).

**Parameters:** Method-specific arguments

**Returns:** Calibrated model

## UQ Methods

### BootstrapUQ

Bootstrap-based uncertainty quantification for sklearn models.

```python
from ueq.methods.bootstrap import BootstrapUQ

bootstrap_uq = BootstrapUQ(model, n_models=100, random_state=None)
```

#### Parameters

- **model** (object): Any scikit-learn compatible estimator
- **n_models** (int, default=100): Number of bootstrap models to train
- **random_state** (int, optional): Random seed for reproducibility

#### Methods

- `fit(X, y)`: Fit bootstrap models on resampled datasets
- `predict(X, return_interval=True, alpha=0.05)`: Predict with uncertainty intervals
- `predict_dist(X)`: Return full predictive distribution

### ConformalUQ

Conformal prediction for distribution-free coverage guarantees.

```python
from ueq.methods.conformal import ConformalUQ

conformal_uq = ConformalUQ(model, alpha=0.05, task_type="regression")
```

#### Parameters

- **model** (object): Any scikit-learn compatible estimator
- **alpha** (float, default=0.05): Significance level (e.g., 0.05 = 95% confidence)
- **task_type** (str, default="regression"): Either "regression" or "classification"

#### Methods

- `fit(X_train, y_train, X_calib, y_calib)`: Fit model and calibrate
- `predict(X, return_interval=False)`: Predict with conformal intervals

### MCDropoutUQ

Monte Carlo Dropout for PyTorch models.

```python
from ueq.methods.mc_dropout import MCDropoutUQ

mc_dropout_uq = MCDropoutUQ(model, n_forward_passes=50, device="cpu")
```

#### Parameters

- **model** (torch.nn.Module): A PyTorch model with dropout layers
- **n_forward_passes** (int, default=50): Number of stochastic passes
- **device** (str, default="cpu"): Device to run on ("cpu" or "cuda")

#### Methods

- `fit(train_loader, criterion, optimizer, epochs=10)`: Train the PyTorch model
- `predict(X)`: Predict with MC Dropout uncertainty estimates

### DeepEnsembleUQ

Deep ensembles for PyTorch models.

```python
from ueq.methods.deep_ensemble import DeepEnsembleUQ

deep_ensemble_uq = DeepEnsembleUQ(model_fn, n_models=5, device="cpu")
```

#### Parameters

- **model_fn** (callable): Function that returns a fresh model instance
- **n_models** (int, default=5): Number of ensemble members
- **device** (str, default="cpu"): Device to train models on

#### Methods

- `fit(train_loader, criterion, optimizer_fn, epochs=10)`: Train all ensemble members
- `predict(X, return_interval=True, alpha=0.05)`: Predict with uncertainty estimates
- `predict_dist(X)`: Return full predictive distribution

### BayesianLinearUQ

Bayesian linear regression with closed-form uncertainty.

```python
from ueq.methods.bayesian_linear import BayesianLinearUQ

bayesian_uq = BayesianLinearUQ(alpha=1.0, beta=1.0)
```

#### Parameters

- **alpha** (float, default=1.0): Precision of the prior distribution
- **beta** (float, default=1.0): Precision of the noise distribution

#### Methods

- `fit(X, y)`: Fit Bayesian Linear Regression
- `predict(X, return_interval=True, alpha=0.05)`: Predict with uncertainty intervals
- `predict_dist(X, n_samples=100)`: Sample from the predictive distribution

### CrossFrameworkEnsembleUQ

Cross-framework ensemble for combining models from different frameworks.

```python
from ueq.methods.cross_ensemble import CrossFrameworkEnsembleUQ

cross_ensemble_uq = CrossFrameworkEnsembleUQ(models, weights=None, aggregation_method="mean")
```

#### Parameters

- **models** (list): List of models from different frameworks
- **weights** (list, optional): Weights for each model
- **aggregation_method** (str, default="mean"): "mean", "median", or "weighted_mean"

#### Methods

- `fit(*args, **kwargs)`: Fit all models in the ensemble
- `predict(X, return_interval=True, alpha=0.05)`: Predict with ensemble uncertainty
- `predict_dist(X)`: Return full predictive distribution
- `get_model_info()`: Get information about all models

## Utility Functions

### Metrics

```python
from ueq import coverage, sharpness, expected_calibration_error, maximum_calibration_error
```

#### `coverage(y_true, intervals)`
Compute coverage: fraction of true values inside prediction intervals.

**Parameters:**
- **y_true** (array-like): True target values
- **intervals** (list): Prediction intervals [(lower, upper), ...]

**Returns:** float - Fraction of points within intervals

#### `sharpness(intervals)`
Compute sharpness: average width of prediction intervals.

**Parameters:**
- **intervals** (list): Prediction intervals [(lower, upper), ...]

**Returns:** float - Mean interval width

#### `expected_calibration_error(y_true, intervals, n_bins=10)`
Compute Expected Calibration Error (ECE).

**Parameters:**
- **y_true** (array-like): True values
- **intervals** (list): Prediction intervals
- **n_bins** (int): Number of bins for calibration

**Returns:** float - Expected calibration error

#### `maximum_calibration_error(y_true, intervals, n_bins=10)`
Compute Maximum Calibration Error (MCE).

**Parameters:**
- **y_true** (array-like): True values
- **intervals** (list): Prediction intervals
- **n_bins** (int): Number of bins

**Returns:** float - Maximum calibration error

### Monitoring

```python
from ueq import UQMonitor, PerformanceMonitor, detect_uncertainty_drift
```

#### `UQMonitor`
Monitor uncertainty quantification models for drift and performance degradation.

**Parameters:**
- **baseline_data** (array-like, optional): Baseline data for drift detection
- **baseline_uncertainty** (array-like, optional): Baseline uncertainty estimates
- **window_size** (int, default=100): Size of sliding window
- **drift_threshold** (float, default=0.1): Threshold for drift detection

**Methods:**
- `update_baseline(data, uncertainty)`: Update baseline statistics
- `monitor(predictions, uncertainty)`: Monitor new predictions
- `get_summary()`: Get monitoring summary

#### `PerformanceMonitor`
Monitor model performance metrics in production.

**Parameters:**
- **window_size** (int, default=100): Size of sliding window

**Methods:**
- `log_performance(predictions, true_values, inference_time=None)`: Log performance
- `get_performance_summary()`: Get performance summary

#### `detect_uncertainty_drift(baseline_uncertainty, current_uncertainty, method='statistical', threshold=0.1)`
Detect uncertainty drift between baseline and current estimates.

**Parameters:**
- **baseline_uncertainty** (array-like): Baseline uncertainty estimates
- **current_uncertainty** (array-like): Current uncertainty estimates
- **method** (str, default='statistical'): Drift detection method
- **threshold** (float, default=0.1): Drift threshold

**Returns:** dict - Drift detection results

### Performance

```python
from ueq import BatchProcessor, PerformanceProfiler, optimize_batch_size, memory_efficient_predict
```

#### `BatchProcessor`
Efficient batch processing for large datasets.

**Parameters:**
- **batch_size** (int, default=1000): Size of batches
- **n_jobs** (int, default=1): Number of parallel jobs
- **backend** (str, default='threading'): 'threading' or 'multiprocessing'

**Methods:**
- `process_batches(data, process_func, **kwargs)`: Process data in batches
- `merge_results(results, merge_func=None)`: Merge batch results

#### `PerformanceProfiler`
Profile performance of uncertainty quantification methods.

**Parameters:**
- **warmup_runs** (int, default=3): Number of warmup runs

**Methods:**
- `profile(func, *args, **kwargs)`: Profile a function
- `benchmark_methods(methods, data, **kwargs)`: Benchmark multiple methods

#### `optimize_batch_size(model, data, min_batch=32, max_batch=2048, step=32, target_time=1.0)`
Find optimal batch size for a model.

**Parameters:**
- **model** (object): Model to optimize
- **data** (array-like): Sample data
- **min_batch** (int, default=32): Minimum batch size
- **max_batch** (int, default=2048): Maximum batch size
- **step** (int, default=32): Step size for search
- **target_time** (float, default=1.0): Target execution time

**Returns:** int - Optimal batch size

#### `memory_efficient_predict(model, data, batch_size=1000, return_uncertainty=True)`
Memory-efficient prediction for large datasets.

**Parameters:**
- **model** (object): Model with predict method
- **data** (array-like): Input data
- **batch_size** (int, default=1000): Batch size
- **return_uncertainty** (bool, default=True): Return uncertainty estimates

**Returns:** Predictions and optionally uncertainty estimates

### API Convenience

```python
from ueq.utils.api import evaluate
```

#### `evaluate(model=None, method="bayesian_linear", X_train=None, y_train=None, X_test=None, y_test=None, metrics=("coverage", "sharpness", "ece", "mce"), **kwargs)`
Train and evaluate a UQ method on given dataset.

**Parameters:**
- **model** (object, optional): Model to evaluate
- **method** (str, default="bayesian_linear"): UQ method
- **X_train, y_train** (array-like): Training data
- **X_test, y_test** (array-like): Test data
- **metrics** (tuple): Metrics to compute
- **\*\*kwargs**: Additional arguments for UQ method

**Returns:**
```python
{
    "mean": array,
    "intervals": list,
    "metrics": dict
}
```

### Visualization

```python
from ueq.utils.plotting import plot_intervals
```

#### `plot_intervals(X, y, mean, intervals, title="UQ Prediction Intervals")`
Plot predictions with uncertainty intervals.

**Parameters:**
- **X** (array-like): Input data (1D only)
- **y** (array-like): True values
- **mean** (array-like): Predicted means
- **intervals** (list): Prediction intervals
- **title** (str): Plot title

**Returns:** None (displays plot)
