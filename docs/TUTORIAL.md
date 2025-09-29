# UEQ Tutorial Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Auto-Detection System](#auto-detection-system)
4. [Cross-Framework Ensembles](#cross-framework-ensembles)
5. [Production Features](#production-features)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)

## Getting Started

### Installation

```bash
pip install ueq
```

**That's it!** UEQ is now available on PyPI. No need to clone or build from source.

### Development Installation

If you want to contribute or use the latest development version:

```bash
git clone https://github.com/kiplangatkorir/ueq.git
cd ueq
pip install -e .
```

### Quick Start

```python
from ueq import UQ
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

# Create model
model = LinearRegression()

# Wrap with UEQ (auto-detection)
uq = UQ(model)  # Automatically selects bootstrap method

# Fit and predict
uq.fit(X, y)
predictions, intervals = uq.predict(X[:10], return_interval=True)

print("Predictions:", predictions)
print("Uncertainty intervals:", intervals)
```

## Basic Usage

### 1. Scikit-learn Models

#### Bootstrap Ensembles

```python
from sklearn.ensemble import RandomForestRegressor
from ueq import UQ

# Create model
model = RandomForestRegressor(n_estimators=100)

# Wrap with UEQ
uq = UQ(model, method="bootstrap", n_models=50)

# Fit and predict
uq.fit(X_train, y_train)
predictions, intervals = uq.predict(X_test, return_interval=True)

# Evaluate uncertainty quality
from ueq import coverage, sharpness
cov = coverage(y_test, intervals)
sharp = sharpness(intervals)

print(f"Coverage: {cov:.3f}")
print(f"Sharpness: {sharp:.3f}")
```

#### Conformal Prediction

```python
from sklearn.linear_model import LinearRegression
from ueq import UQ

# Create model
model = LinearRegression()

# Wrap with UEQ
uq = UQ(model, method="conformal", alpha=0.1)  # 90% confidence

# Fit with calibration data
uq.fit(X_train, y_train, X_calib, y_calib)

# Predict with conformal intervals
predictions, intervals = uq.predict(X_test, return_interval=True)
```

### 2. PyTorch Models

#### MC Dropout

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ueq import UQ

# Define model with dropout
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Create model and data
model = Net()
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Wrap with UEQ
uq = UQ(model, method="mc_dropout", n_forward_passes=100)

# Train
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
uq.fit(loader, criterion, optimizer, epochs=10)

# Predict with uncertainty
X_test_tensor = torch.FloatTensor(X_test)
mean_pred, std_pred = uq.predict(X_test_tensor)
```

#### Deep Ensembles

```python
from ueq import UQ

# Define model constructor
def create_model():
    return Net()

# Wrap with UEQ
uq = UQ(create_model, method="deep_ensemble", n_models=5)

# Train ensemble
criterion = nn.MSELoss()
optimizer_fn = lambda params: optim.Adam(params, lr=0.01)
uq.fit(loader, criterion, optimizer_fn, epochs=10)

# Predict with ensemble uncertainty
mean_pred, intervals = uq.predict(X_test_tensor, return_interval=True)
```

### 3. Bayesian Linear Regression

```python
from ueq import UQ

# No model needed - Bayesian linear regression
uq = UQ(method="bayesian_linear", alpha=2.0, beta=25.0)

# Fit and predict
uq.fit(X_train, y_train)
predictions, intervals = uq.predict(X_test, return_interval=True)

# Get predictive distribution
samples = uq.predict_dist(X_test, n_samples=1000)
```

## Auto-Detection System

UEQ automatically detects your model type and selects the optimal UQ method:

```python
from ueq import UQ
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn

# Scikit-learn regressor → Bootstrap
sklearn_reg = LinearRegression()
uq1 = UQ(sklearn_reg)  # method="bootstrap"

# Scikit-learn classifier → Conformal
sklearn_clf = RandomForestClassifier()
uq2 = UQ(sklearn_clf)  # method="conformal"

# PyTorch model → MC Dropout
pytorch_model = nn.Linear(10, 1)
uq3 = UQ(pytorch_model)  # method="mc_dropout"

# Constructor function → Deep Ensemble
def create_model():
    return nn.Linear(10, 1)
uq4 = UQ(create_model)  # method="deep_ensemble"

# No model → Bayesian Linear
uq5 = UQ()  # method="bayesian_linear"

# Multiple models → Cross-framework ensemble
models = [sklearn_reg, pytorch_model]
uq6 = UQ(models)  # method="cross_ensemble"
```

### Get Model Information

```python
# Get information about detected model and method
info = uq.get_info()
print(f"Model type: {info['model_type']}")
print(f"Selected method: {info['method']}")
print(f"Model class: {info['model_class']}")
```

## Cross-Framework Ensembles

Combine models from different frameworks in unified uncertainty estimates:

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch.nn as nn
from ueq import UQ

# Create models from different frameworks
sklearn_model1 = LinearRegression()
sklearn_model2 = RandomForestRegressor(n_estimators=50)

class PyTorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

pytorch_model = PyTorchNet()

# Create cross-framework ensemble
models = [sklearn_model1, sklearn_model2, pytorch_model]
uq = UQ(models)  # Auto-detects cross-framework ensemble

# Fit ensemble
uq.fit(X_train, y_train)

# Predict with unified uncertainty
predictions, intervals = uq.predict(X_test, return_interval=True)

# Get ensemble information
ensemble_info = uq.uq_model.get_model_info()
print(f"Successfully fitted models: {ensemble_info['n_models']}")
print(f"Aggregation method: {ensemble_info['aggregation_method']}")
```

## Production Features

### Model Monitoring and Drift Detection

```python
from ueq import UQ, UQMonitor

# Create and train model
uq = UQ(model)
uq.fit(X_train, y_train)

# Get baseline uncertainty
baseline_pred, baseline_uncertainty = uq.predict(X_train[:100], return_interval=True)

# Create monitor
monitor = UQMonitor(
    baseline_data=X_train[:100],
    baseline_uncertainty=baseline_uncertainty,
    drift_threshold=0.1
)

# Monitor new data
new_pred, new_uncertainty = uq.predict(X_new, return_interval=True)
results = monitor.monitor(new_pred, new_uncertainty)

print(f"Drift score: {results['drift_score']:.3f}")
print(f"Alerts: {len(results['alerts'])}")

# Check for alerts
if results['alerts']:
    for alert in results['alerts']:
        print(f"⚠️ {alert['type']}: {alert['message']}")
else:
    print("✅ No alerts - model is healthy")
```

### Performance Optimization

```python
from ueq import UQ, BatchProcessor

# Create model
uq = UQ(model)

# Large dataset prediction
X_large = np.random.randn(10000, 10)

# Method 1: Built-in large dataset prediction
predictions = uq.predict_large_dataset(X_large, batch_size=1000)

# Method 2: Custom batch processing
batch_processor = BatchProcessor(batch_size=500, n_jobs=4)

def process_batch(batch):
    return uq.predict(batch, return_interval=False)

batch_results = batch_processor.process_batches(X_large, process_batch)
final_predictions = batch_processor.merge_results(batch_results)
```

### Production Service

```python
from ueq import UQ, UQMonitor, PerformanceMonitor

class ProductionUQService:
    def __init__(self, model, baseline_data=None, baseline_uncertainty=None):
        self.uq = UQ(model)
        self.monitor = UQMonitor(
            baseline_data=baseline_data,
            baseline_uncertainty=baseline_uncertainty
        )
        self.performance_monitor = PerformanceMonitor()
    
    def predict(self, X, return_uncertainty=True):
        """Production prediction with monitoring."""
        import time
        start_time = time.time()
        
        # Get predictions
        if return_uncertainty:
            predictions, uncertainty = self.uq.predict(X, return_interval=True)
        else:
            predictions = self.uq.predict(X, return_interval=False)
            uncertainty = None
        
        # Monitor
        monitoring_results = self.monitor.monitor(predictions, uncertainty)
        
        # Log performance
        inference_time = time.time() - start_time
        self.performance_monitor.log_performance(
            predictions, np.zeros(len(predictions)), inference_time
        )
        
        return {
            'predictions': predictions,
            'uncertainty': uncertainty,
            'monitoring': monitoring_results,
            'inference_time': inference_time
        }
    
    def get_health_status(self):
        """Get service health status."""
        monitor_summary = self.monitor.get_summary()
        perf_summary = self.performance_monitor.get_performance_summary()
        
        return {
            'monitoring': monitor_summary,
            'performance': perf_summary,
            'status': 'healthy' if monitor_summary.get('status') == 'healthy' else 'warning'
        }

# Use the service
service = ProductionUQService(model, baseline_data=X_train[:100])
service.uq.fit(X_train, y_train)

# Make predictions
results = service.predict(X_test[:10], return_uncertainty=True)
health = service.get_health_status()

print(f"Service status: {health['status']}")
print(f"Inference time: {results['inference_time']:.3f}s")
```

## Advanced Usage

### Custom UQ Methods

```python
from ueq.core import UQ
from ueq.methods.bootstrap import BootstrapUQ

class CustomUQ(BootstrapUQ):
    def __init__(self, model, n_models=100, custom_param=1.0):
        super().__init__(model, n_models)
        self.custom_param = custom_param
    
    def predict(self, X, return_interval=True, alpha=0.05):
        # Custom prediction logic
        preds, intervals = super().predict(X, return_interval, alpha)
        
        # Apply custom modifications
        if return_interval:
            # Modify intervals based on custom parameter
            modified_intervals = []
            for lower, upper in intervals:
                width = upper - lower
                new_lower = lower - width * self.custom_param * 0.1
                new_upper = upper + width * self.custom_param * 0.1
                modified_intervals.append((new_lower, new_upper))
            return preds, modified_intervals
        
        return preds

# Use custom method
custom_uq = CustomUQ(model, n_models=50, custom_param=2.0)
custom_uq.fit(X_train, y_train)
predictions, intervals = custom_uq.predict(X_test, return_interval=True)
```

### Performance Benchmarking

```python
from ueq import UQ, PerformanceProfiler

# Create models
uq1 = UQ(model1, method="bootstrap", n_models=20)
uq2 = UQ(model2, method="bootstrap", n_models=50)
uq3 = UQ(model3, method="bootstrap", n_models=100)

# Benchmark performance
profiler = PerformanceProfiler()

methods = {
    'bootstrap_20': lambda x: uq1.predict(x, return_interval=False),
    'bootstrap_50': lambda x: uq2.predict(x, return_interval=False),
    'bootstrap_100': lambda x: uq3.predict(x, return_interval=False)
}

results = profiler.benchmark_methods(methods, X_test)

for method_name, result in results.items():
    if result['success']:
        print(f"{method_name}: {result['timing']['execution_time']:.3f}s")
    else:
        print(f"{method_name}: Failed - {result['error']}")
```

### Uncertainty Calibration

```python
from ueq import UQ, expected_calibration_error, maximum_calibration_error

# Train model
uq = UQ(model)
uq.fit(X_train, y_train)

# Get predictions
predictions, intervals = uq.predict(X_test, return_interval=True)

# Evaluate calibration
ece = expected_calibration_error(y_test, intervals, n_bins=10)
mce = maximum_calibration_error(y_test, intervals, n_bins=10)

print(f"Expected Calibration Error: {ece:.3f}")
print(f"Maximum Calibration Error: {mce:.3f}")

# Calibrate if needed
if ece > 0.1:  # Threshold for good calibration
    print("Model needs calibration")
    # Implement calibration strategy
```

## Best Practices

### 1. Model Selection

```python
# Choose UQ method based on your needs:
# - Bootstrap: Good for most sklearn models, provides ensemble uncertainty
# - Conformal: Provides distribution-free coverage guarantees
# - MC Dropout: Good for PyTorch models with dropout
# - Deep Ensembles: Best uncertainty for PyTorch models
# - Bayesian Linear: Fast, closed-form uncertainty for linear problems
# - Cross-framework: Combine different model types

# Example: For production ML systems
if model_type == "sklearn":
    uq = UQ(model, method="bootstrap", n_models=100)
elif model_type == "pytorch":
    uq = UQ(model, method="deep_ensemble", n_models=5)
else:
    uq = UQ(model)  # Auto-detect
```

### 2. Data Splitting

```python
# For conformal prediction, use proper train/calibration/test splits
from sklearn.model_selection import train_test_split

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Use calibration set for conformal prediction
uq = UQ(model, method="conformal", alpha=0.1)
uq.fit(X_train, y_train, X_calib, y_calib)
```

### 3. Uncertainty Evaluation

```python
# Always evaluate uncertainty quality
from ueq import coverage, sharpness, expected_calibration_error

# Get predictions
predictions, intervals = uq.predict(X_test, return_interval=True)

# Compute metrics
cov = coverage(y_test, intervals)
sharp = sharpness(intervals)
ece = expected_calibration_error(y_test, intervals)

print(f"Coverage: {cov:.3f} (target: 0.95)")
print(f"Sharpness: {sharp:.3f} (lower is better)")
print(f"ECE: {ece:.3f} (lower is better)")

# Check if uncertainty is well-calibrated
if abs(cov - 0.95) > 0.05:
    print("⚠️ Uncertainty may not be well-calibrated")
```

### 4. Production Deployment

```python
# Use monitoring in production
monitor = UQMonitor(baseline_data=X_train, baseline_uncertainty=baseline_unc)

# Monitor continuously
for new_batch in data_stream:
    predictions, uncertainty = uq.predict(new_batch, return_interval=True)
    results = monitor.monitor(predictions, uncertainty)
    
    # Check for drift
    if results['drift_score'] > 0.1:
        print("⚠️ Drift detected - consider retraining")
        # Trigger retraining or alert
    
    # Log performance
    performance_monitor.log_performance(predictions, true_values, inference_time)
```

### 5. Memory Management

```python
# For large datasets, use batch processing
batch_size = 1000  # Adjust based on available memory

# Method 1: Built-in large dataset prediction
predictions = uq.predict_large_dataset(X_large, batch_size=batch_size)

# Method 2: Custom batching
for i in range(0, len(X_large), batch_size):
    batch = X_large[i:i+batch_size]
    batch_pred = uq.predict(batch, return_interval=False)
    # Process batch results
```

### 6. Error Handling

```python
try:
    # Fit model
    uq.fit(X_train, y_train)
    
    # Get predictions
    predictions, intervals = uq.predict(X_test, return_interval=True)
    
    # Evaluate
    cov = coverage(y_test, intervals)
    print(f"Coverage: {cov:.3f}")
    
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
```

This tutorial covers the essential aspects of using UEQ effectively. For more specific use cases, refer to the examples in the `examples/` directory.

## 🐛 Issues and Support

If you encounter any problems while following this tutorial:

1. **Check the [API documentation](API.md)** for detailed method signatures
2. **Search existing [issues](https://github.com/kiplangatkorir/ueq/issues)** to see if your problem is already reported
3. **Create a new issue** with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and UEQ version (`ueq.__version__`)
   - Error messages (if any)

**We're here to help!** 🚀
