# UEQ Examples Documentation

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Advanced Examples](#advanced-examples)
3. [Production Examples](#production-examples)
4. [Benchmark Examples](#benchmark-examples)
5. [Cross-Framework Examples](#cross-framework-examples)

## Basic Examples

### 1. Bootstrap UQ with Random Forest

**File:** `examples/bootstrap_demo.py`

```python
#!/usr/bin/env python3
"""
Bootstrap UQ Demo

Demonstrates bootstrap-based uncertainty quantification with a RandomForestRegressor.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from ueq import UQ

def main():
    # Generate synthetic data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Create and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    uq = UQ(model, method="bootstrap", n_models=50)
    uq.fit(X_train, y_train)
    
    # Get predictions with uncertainty
    predictions, intervals = uq.predict(X_test, return_interval=True)
    
    # Evaluate uncertainty quality
    from ueq import coverage, sharpness
    cov = coverage(y_test, intervals)
    sharp = sharpness(intervals)
    
    print(f"Coverage: {cov:.3f}")
    print(f"Sharpness: {sharp:.3f}")
    
    # Plot results
    from ueq.utils.plotting import plot_intervals
    plot_intervals(X_test[:, 0], y_test, predictions, intervals, 
                   title="Bootstrap UQ Results")

if __name__ == "__main__":
    main()
```

**Key Features:**
- Bootstrap ensembles for sklearn models
- Uncertainty interval evaluation
- Visualization of results

### 2. Conformal Prediction

**File:** `examples/conformal_demo.py`

```python
#!/usr/bin/env python3
"""
Conformal Prediction Demo

Demonstrates conformal prediction for distribution-free coverage guarantees.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from ueq import UQ

def main():
    # Generate data
    X, y = make_regression(n_samples=300, n_features=5, noise=15, random_state=42)
    
    # Split into train, calibration, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Create and train model
    model = LinearRegression()
    uq = UQ(model, method="conformal", alpha=0.1)  # 90% confidence
    
    # Fit with calibration data
    uq.fit(X_train, y_train, X_calib, y_calib)
    
    # Get predictions with conformal intervals
    predictions, intervals = uq.predict(X_test, return_interval=True)
    
    # Evaluate coverage
    from ueq import coverage
    cov = coverage(y_test, intervals)
    
    print(f"Target coverage: 0.90")
    print(f"Actual coverage: {cov:.3f}")
    
    # Plot results
    from ueq.utils.plotting import plot_intervals
    plot_intervals(X_test[:, 0], y_test, predictions, intervals,
                   title="Conformal Prediction Results")

if __name__ == "__main__":
    main()
```

**Key Features:**
- Distribution-free coverage guarantees
- Proper train/calibration/test splits
- Coverage validation

### 3. MC Dropout with PyTorch

**File:** `examples/mc_dropout_demo.py`

```python
#!/usr/bin/env python3
"""
MC Dropout Demo

Demonstrates Monte Carlo Dropout for PyTorch models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from ueq import UQ

class Net(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def main():
    # Generate data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create data loader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = Net(input_size=5, hidden_size=50, dropout_rate=0.2)
    
    # Wrap with UEQ
    uq = UQ(model, method="mc_dropout", n_forward_passes=100)
    
    # Train model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    uq.fit(loader, criterion, optimizer, epochs=20)
    
    # Get predictions with uncertainty
    mean_pred, std_pred = uq.predict(X_test_tensor)
    
    # Convert to numpy for evaluation
    mean_pred_np = mean_pred.detach().numpy().flatten()
    std_pred_np = std_pred.detach().numpy().flatten()
    
    # Create intervals
    intervals = [(m - 1.96*s, m + 1.96*s) for m, s in zip(mean_pred_np, std_pred_np)]
    
    # Evaluate
    from ueq import coverage, sharpness
    cov = coverage(y_test, intervals)
    sharp = sharpness(intervals)
    
    print(f"Coverage: {cov:.3f}")
    print(f"Sharpness: {sharp:.3f}")
    
    # Plot results
    from ueq.utils.plotting import plot_intervals
    plot_intervals(X_test[:, 0], y_test, mean_pred_np, intervals,
                   title="MC Dropout Results")

if __name__ == "__main__":
    main()
```

**Key Features:**
- PyTorch model with dropout
- Monte Carlo sampling for uncertainty
- Proper tensor handling

### 4. Deep Ensembles

**File:** `examples/deep_ensemble_demo.py`

```python
#!/usr/bin/env python3
"""
Deep Ensemble Demo

Demonstrates deep ensembles for PyTorch models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from ueq import UQ

class Net(nn.Module):
    def __init__(self, input_size=5, hidden_size=50):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def create_model():
    """Model constructor for deep ensemble."""
    return Net(input_size=5, hidden_size=50)

def main():
    # Generate data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create data loader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create deep ensemble
    uq = UQ(create_model, method="deep_ensemble", n_models=5)
    
    # Train ensemble
    criterion = nn.MSELoss()
    optimizer_fn = lambda params: optim.Adam(params, lr=0.01)
    uq.fit(loader, criterion, optimizer_fn, epochs=20)
    
    # Get predictions with uncertainty
    mean_pred, intervals = uq.predict(X_test_tensor, return_interval=True)
    
    # Convert to numpy for evaluation
    mean_pred_np = mean_pred.detach().numpy().flatten()
    
    # Evaluate
    from ueq import coverage, sharpness
    cov = coverage(y_test, intervals)
    sharp = sharpness(intervals)
    
    print(f"Coverage: {cov:.3f}")
    print(f"Sharpness: {sharp:.3f}")
    
    # Plot results
    from ueq.utils.plotting import plot_intervals
    plot_intervals(X_test[:, 0], y_test, mean_pred_np, intervals,
                   title="Deep Ensemble Results")

if __name__ == "__main__":
    main()
```

**Key Features:**
- Multiple independently trained models
- Ensemble uncertainty estimation
- Model constructor pattern

### 5. Bayesian Linear Regression

**File:** `examples/bayesian_linear_demo.py`

```python
#!/usr/bin/env python3
"""
Bayesian Linear Regression Demo

Demonstrates Bayesian linear regression with closed-form uncertainty.
"""

import numpy as np
from sklearn.datasets import make_regression
from ueq import UQ

def main():
    # Generate data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Create Bayesian linear regression
    uq = UQ(method="bayesian_linear", alpha=2.0, beta=25.0)
    
    # Fit model
    uq.fit(X_train, y_train)
    
    # Get predictions with uncertainty
    predictions, intervals = uq.predict(X_test, return_interval=True)
    
    # Get predictive distribution samples
    samples = uq.predict_dist(X_test, n_samples=1000)
    
    # Evaluate
    from ueq import coverage, sharpness
    cov = coverage(y_test, intervals)
    sharp = sharpness(intervals)
    
    print(f"Coverage: {cov:.3f}")
    print(f"Sharpness: {sharp:.3f}")
    print(f"Predictive samples shape: {samples.shape}")
    
    # Plot results
    from ueq.utils.plotting import plot_intervals
    plot_intervals(X_test[:, 0], y_test, predictions, intervals,
                   title="Bayesian Linear Regression Results")

if __name__ == "__main__":
    main()
```

**Key Features:**
- Closed-form Bayesian uncertainty
- Predictive distribution sampling
- No model required

## Advanced Examples

### 1. Auto-Detection System

**File:** `examples/auto_detection_demo.py`

```python
#!/usr/bin/env python3
"""
Auto-Detection Demo

Demonstrates UEQ's automatic model type detection and method selection.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
from ueq import UQ

def main():
    # Generate data
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    print("🔍 Auto-Detection Demo")
    print("=" * 30)
    
    # 1. Scikit-learn regressor → Bootstrap
    print("\n1. Scikit-learn Regressor")
    sklearn_reg = LinearRegression()
    uq1 = UQ(sklearn_reg)  # Auto-detects bootstrap
    
    info1 = uq1.get_info()
    print(f"   Model type: {info1['model_type']}")
    print(f"   Selected method: {info1['method']}")
    print(f"   Model class: {info1['model_class']}")
    
    # 2. Scikit-learn classifier → Conformal
    print("\n2. Scikit-learn Classifier")
    sklearn_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    uq2 = UQ(sklearn_clf)  # Auto-detects conformal
    
    info2 = uq2.get_info()
    print(f"   Model type: {info2['model_type']}")
    print(f"   Selected method: {info2['method']}")
    print(f"   Model class: {info2['model_class']}")
    
    # 3. PyTorch model → MC Dropout
    print("\n3. PyTorch Model")
    pytorch_model = nn.Linear(5, 1)
    uq3 = UQ(pytorch_model)  # Auto-detects MC dropout
    
    info3 = uq3.get_info()
    print(f"   Model type: {info3['model_type']}")
    print(f"   Selected method: {info3['method']}")
    print(f"   Model class: {info3['model_class']}")
    
    # 4. Constructor function → Deep Ensemble
    print("\n4. Constructor Function")
    def create_model():
        return nn.Linear(5, 1)
    
    uq4 = UQ(create_model)  # Auto-detects deep ensemble
    
    info4 = uq4.get_info()
    print(f"   Model type: {info4['model_type']}")
    print(f"   Selected method: {info4['method']}")
    print(f"   Model class: {info4['model_class']}")
    
    # 5. No model → Bayesian Linear
    print("\n5. No Model")
    uq5 = UQ()  # Auto-detects Bayesian linear
    
    info5 = uq5.get_info()
    print(f"   Model type: {info5['model_type']}")
    print(f"   Selected method: {info5['method']}")
    print(f"   Model class: {info5['model_class']}")
    
    print("\n✅ Auto-detection completed!")

if __name__ == "__main__":
    main()
```

**Key Features:**
- Automatic model type detection
- Optimal method selection
- Zero-configuration UQ

### 2. Cross-Framework Ensembles

**File:** `examples/cross_framework_demo.py`

```python
#!/usr/bin/env python3
"""
Cross-Framework Ensemble Demo

Demonstrates combining models from different frameworks in unified uncertainty estimates.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch.nn as nn
from ueq import UQ

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)
    
    def forward(self, x):
        return self.fc(x)

def main():
    # Generate data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    print("🔄 Cross-Framework Ensemble Demo")
    print("=" * 40)
    
    # Create models from different frameworks
    sklearn_model1 = LinearRegression()
    sklearn_model2 = RandomForestRegressor(n_estimators=50, random_state=42)
    pytorch_model = SimpleNet()
    
    # Create cross-framework ensemble
    models = [sklearn_model1, sklearn_model2, pytorch_model]
    uq = UQ(models)  # Auto-detects cross-framework ensemble
    
    # Get ensemble information
    info = uq.get_info()
    print(f"Model type: {info['model_type']}")
    print(f"Method: {info['method']}")
    print(f"Number of models: {info['n_models']}")
    print(f"Model classes: {info['model_classes']}")
    
    # Fit ensemble
    print("\nTraining ensemble...")
    uq.fit(X_train, y_train)
    
    # Get predictions with unified uncertainty
    predictions, intervals = uq.predict(X_test, return_interval=True)
    
    # Evaluate
    from ueq import coverage, sharpness
    cov = coverage(y_test, intervals)
    sharp = sharpness(intervals)
    
    print(f"\nResults:")
    print(f"Coverage: {cov:.3f}")
    print(f"Sharpness: {sharp:.3f}")
    
    # Get detailed ensemble info
    ensemble_info = uq.uq_model.get_model_info()
    print(f"\nEnsemble Details:")
    print(f"Successfully fitted models: {ensemble_info['n_models']}")
    print(f"Aggregation method: {ensemble_info['aggregation_method']}")
    print(f"Weights: {ensemble_info['weights']}")
    
    for i, model_info in enumerate(ensemble_info['models']):
        print(f"  Model {i+1}: {model_info['model_class']} ({model_info['model_type']})")
    
    print("\n✅ Cross-framework ensemble completed!")

if __name__ == "__main__":
    main()
```

**Key Features:**
- Multi-framework model combination
- Unified uncertainty estimation
- Automatic model wrapping

## Production Examples

### 1. Production Service

**File:** `examples/production_demo.py`

```python
#!/usr/bin/env python3
"""
Production Features Demo

Demonstrates production-ready features of UEQ v1.0.1:
- Model monitoring and drift detection
- Performance optimization for large datasets
- Batch processing and parallelization
- Production deployment patterns
"""

import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from ueq import UQ
from ueq.utils.monitoring import UQMonitor, PerformanceMonitor
from ueq.utils.performance import BatchProcessor, PerformanceProfiler

def main():
    print("🏭 Production Features Demo")
    print("=" * 50)
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, noise=5, random_state=42)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Create and train model
    print("\n🤖 Training UQ Model...")
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    uq = UQ(model, method="bootstrap", n_models=20)
    uq.fit(X_train, y_train)
    
    print("✅ Model trained successfully")
    
    # 1. Model Monitoring and Drift Detection
    print("\n📊 Model Monitoring and Drift Detection")
    print("-" * 40)
    
    # Get baseline uncertainty
    baseline_pred, baseline_uncertainty = uq.predict(X_train[:100], return_interval=True)
    
    # Create monitor
    monitor = UQMonitor(
        baseline_data=X_train[:100],
        baseline_uncertainty=baseline_uncertainty,
        drift_threshold=0.1
    )
    
    # Monitor new data
    new_pred, new_uncertainty = uq.predict(X_test[:50], return_interval=True)
    monitoring_results = monitor.monitor(new_pred, new_uncertainty)
    
    print(f"Drift Score: {monitoring_results['drift_score']:.3f}")
    print(f"Alerts: {len(monitoring_results['alerts'])}")
    
    if monitoring_results['alerts']:
        for alert in monitoring_results['alerts']:
            print(f"  - {alert['type']}: {alert['message']}")
    else:
        print("  ✅ No alerts - model is healthy")
    
    # 2. Performance Optimization
    print("\n⚡ Performance Optimization")
    print("-" * 30)
    
    # Benchmark performance
    profiler = PerformanceProfiler()
    
    def predict_func(data):
        return uq.predict(data, return_interval=False)
    
    def predict_with_uncertainty_func(data):
        return uq.predict(data, return_interval=True)
    
    methods = {
        'predict': predict_func,
        'predict_with_uncertainty': predict_with_uncertainty_func
    }
    
    benchmark_results = profiler.benchmark_methods(methods, X_test[:100])
    
    for method_name, results in benchmark_results.items():
        if results['success']:
            print(f"{method_name}: {results['timing']['execution_time']:.3f}s")
        else:
            print(f"{method_name}: Failed - {results['error']}")
    
    # 3. Batch Processing for Large Datasets
    print("\n📦 Batch Processing for Large Datasets")
    print("-" * 40)
    
    # Create large dataset
    X_large = np.random.randn(5000, 10)
    
    # Process in batches
    batch_processor = BatchProcessor(batch_size=500, n_jobs=1)
    
    def process_batch(batch):
        return uq.predict(batch, return_interval=False)
    
    start_time = time.time()
    batch_results = batch_processor.process_batches(X_large, process_batch)
    batch_time = time.time() - start_time
    
    # Merge results
    final_predictions = batch_processor.merge_results(batch_results)
    
    print(f"Processed {len(X_large)} samples in {batch_time:.3f}s")
    print(f"Batch results: {len(batch_results)} batches")
    print(f"Final predictions shape: {final_predictions.shape}")
    
    # 4. Production Deployment Pattern
    print("\n🚀 Production Deployment Pattern")
    print("-" * 35)
    
    class ProductionUQService:
        """Production UQ service with monitoring and optimization."""
        
        def __init__(self, model, baseline_data=None, baseline_uncertainty=None):
            self.uq = UQ(model)
            self.monitor = UQMonitor(
                baseline_data=baseline_data,
                baseline_uncertainty=baseline_uncertainty
            )
            self.performance_monitor = PerformanceMonitor()
            self.batch_processor = BatchProcessor(batch_size=1000)
            
        def predict(self, X, return_uncertainty=True):
            """Production prediction with monitoring."""
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
        
        def predict_large(self, X, batch_size=1000):
            """Large dataset prediction."""
            def process_batch(batch):
                return self.uq.predict(batch, return_interval=False)
            
            results = self.batch_processor.process_batches(X, process_batch)
            return self.batch_processor.merge_results(results)
        
        def get_health_status(self):
            """Get service health status."""
            monitor_summary = self.monitor.get_summary()
            perf_summary = self.performance_monitor.get_performance_summary()
            
            return {
                'monitoring': monitor_summary,
                'performance': perf_summary,
                'status': 'healthy' if monitor_summary.get('status') == 'healthy' else 'warning'
            }
    
    # Create production service
    service = ProductionUQService(
        model=RandomForestRegressor(n_estimators=20, random_state=42),
        baseline_data=X_train[:100],
        baseline_uncertainty=baseline_uncertainty
    )
    
    # Train the service
    service.uq.fit(X_train, y_train)
    
    # Test production predictions
    print("Testing production service...")
    results = service.predict(X_test[:10], return_uncertainty=True)
    
    print(f"Predictions shape: {results['predictions'].shape}")
    print(f"Inference time: {results['inference_time']:.3f}s")
    print(f"Drift score: {results['monitoring']['drift_score']:.3f}")
    
    # Get health status
    health = service.get_health_status()
    print(f"Service status: {health['status']}")
    
    # 5. Real-time Monitoring Dashboard
    print("\n📈 Real-time Monitoring Dashboard")
    print("-" * 35)
    
    # Simulate real-time monitoring
    print("Simulating real-time monitoring...")
    
    for i in range(5):
        # Simulate new data
        new_data = X_test[i*10:(i+1)*10]
        new_results = service.predict(new_data, return_uncertainty=True)
        
        drift_score = new_results['monitoring']['drift_score']
        alerts = new_results['monitoring']['alerts']
        
        status = "🟢" if drift_score < 0.1 else "🟡" if drift_score < 0.2 else "🔴"
        
        print(f"Batch {i+1}: {status} Drift: {drift_score:.3f}, Alerts: {len(alerts)}")
        
        if alerts:
            for alert in alerts:
                print(f"  ⚠️  {alert['message']}")
    
    print("\n🎉 Production features demo completed!")
    print("=" * 50)

def demo_advanced_monitoring():
    """Demonstrate advanced monitoring capabilities."""
    
    print("\n🔍 Advanced Monitoring Demo")
    print("=" * 30)
    
    # Create model with different uncertainty patterns
    model1 = LinearRegression()
    model2 = RandomForestRegressor(n_estimators=10, random_state=42)
    
    uq1 = UQ(model1, method="bootstrap", n_models=10)
    uq2 = UQ(model2, method="bootstrap", n_models=10)
    
    # Generate data
    X, y = make_regression(n_samples=200, n_features=5, noise=5, random_state=42)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Train models
    uq1.fit(X_train, y_train)
    uq2.fit(X_train, y_train)
    
    # Get baseline uncertainties
    _, baseline_unc1 = uq1.predict(X_train[:50], return_interval=True)
    _, baseline_unc2 = uq2.predict(X_train[:50], return_interval=True)
    
    # Create monitors
    monitor1 = UQMonitor(baseline_uncertainty=baseline_unc1, drift_threshold=0.15)
    monitor2 = UQMonitor(baseline_uncertainty=baseline_unc2, drift_threshold=0.15)
    
    # Monitor both models
    _, unc1 = uq1.predict(X_test[:30], return_interval=True)
    _, unc2 = uq2.predict(X_test[:30], return_interval=True)
    
    results1 = monitor1.monitor(np.zeros(30), unc1)
    results2 = monitor2.monitor(np.zeros(30), unc2)
    
    print(f"Model 1 (Linear): Drift = {results1['drift_score']:.3f}")
    print(f"Model 2 (Random Forest): Drift = {results2['drift_score']:.3f}")
    
    # Compare models
    if results1['drift_score'] < results2['drift_score']:
        print("✅ Model 1 shows more stable uncertainty")
    else:
        print("✅ Model 2 shows more stable uncertainty")

if __name__ == "__main__":
    main()
    demo_advanced_monitoring()
```

**Key Features:**
- Production service architecture
- Real-time monitoring
- Performance optimization
- Health status tracking

## Benchmark Examples

### 1. Climate Forecasting Benchmark

**File:** `examples/benchmarks/climate_forecasting.py`

```python
#!/usr/bin/env python3
"""
Climate Forecasting Benchmark

Benchmarks MC Dropout UQ on the California Housing dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ueq import UQ
from ueq.utils.metrics import coverage, sharpness, expected_calibration_error

class ClimateNet(nn.Module):
    def __init__(self, input_size=8, hidden_size=100, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

def main():
    print("🌡️ Climate Forecasting Benchmark")
    print("=" * 40)
    
    # Load California Housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Preprocess data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create data loader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Create model
    model = ClimateNet(input_size=8, hidden_size=100, dropout_rate=0.2)
    
    # Wrap with UEQ
    uq = UQ(model, method="mc_dropout", n_forward_passes=100)
    
    # Train model
    print("\nTraining model...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    uq.fit(loader, criterion, optimizer, epochs=50)
    
    # Get predictions
    print("\nGetting predictions...")
    mean_pred, std_pred = uq.predict(X_test_tensor)
    
    # Convert to numpy
    mean_pred_np = mean_pred.detach().numpy().flatten()
    std_pred_np = std_pred.detach().numpy().flatten()
    
    # Create intervals
    intervals = [(m - 1.96*s, m + 1.96*s) for m, s in zip(mean_pred_np, std_pred_np)]
    
    # Evaluate uncertainty quality
    print("\n📊 Uncertainty Quality Metrics")
    print("-" * 30)
    
    cov = coverage(y_test, intervals)
    sharp = sharpness(intervals)
    ece = expected_calibration_error(y_test, intervals, n_bins=10)
    
    print(f"Coverage: {cov:.3f} (target: 0.95)")
    print(f"Sharpness: {sharp:.3f}")
    print(f"Expected Calibration Error: {ece:.3f}")
    
    # Performance metrics
    mse = np.mean((mean_pred_np - y_test) ** 2)
    mae = np.mean(np.abs(mean_pred_np - y_test))
    
    print(f"\n📈 Performance Metrics")
    print("-" * 25)
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    
    # Plot results
    print("\n📊 Plotting results...")
    from ueq.utils.plotting import plot_intervals
    
    # Plot first 100 points for clarity
    plot_intervals(
        X_test[:100, 0], y_test[:100], mean_pred_np[:100], intervals[:100],
        title="Climate Forecasting: MC Dropout UQ Results"
    )
    
    print("\n✅ Benchmark completed!")

if __name__ == "__main__":
    main()
```

**Key Features:**
- Real-world dataset (California Housing)
- PyTorch model with dropout
- Comprehensive uncertainty evaluation
- Performance benchmarking

### 2. API Evaluation Benchmark

**File:** `examples/benchmarks/api_evaluation.py`

```python
#!/usr/bin/env python3
"""
API Evaluation Benchmark

Benchmarks the evaluate utility function with different UQ methods.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from ueq.utils.api import evaluate

def main():
    print("🔧 API Evaluation Benchmark")
    print("=" * 35)
    
    # Generate data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test different UQ methods
    methods = [
        ("bootstrap", RandomForestRegressor(n_estimators=50, random_state=42)),
        ("conformal", LinearRegression()),
        ("bayesian_linear", None)
    ]
    
    results = {}
    
    for method_name, model in methods:
        print(f"\n🧪 Testing {method_name}...")
        
        try:
            result = evaluate(
                model=model,
                method=method_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                metrics=("coverage", "sharpness", "ece", "mce")
            )
            
            results[method_name] = result
            
            print(f"  ✅ Success!")
            print(f"  Coverage: {result['metrics']['coverage']:.3f}")
            print(f"  Sharpness: {result['metrics']['sharpness']:.3f}")
            print(f"  ECE: {result['metrics']['ece']:.3f}")
            print(f"  MCE: {result['metrics']['mce']:.3f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[method_name] = None
    
    # Summary
    print(f"\n📊 Summary")
    print("-" * 15)
    
    for method_name, result in results.items():
        if result is not None:
            print(f"{method_name:15}: ✅ Coverage={result['metrics']['coverage']:.3f}, ECE={result['metrics']['ece']:.3f}")
        else:
            print(f"{method_name:15}: ❌ Failed")
    
    print("\n✅ API evaluation completed!")

if __name__ == "__main__":
    main()
```

**Key Features:**
- API utility testing
- Multiple UQ methods
- Error handling
- Performance comparison

## Cross-Framework Examples

### 1. Multi-Framework Ensemble

**File:** `examples/cross_framework_advanced.py`

```python
#!/usr/bin/env python3
"""
Advanced Cross-Framework Ensemble

Demonstrates advanced cross-framework ensemble techniques.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ueq import UQ

class SimpleNet(nn.Module):
    def __init__(self, input_size=5, hidden_size=50):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def create_pytorch_model():
    """Create a fresh PyTorch model."""
    return SimpleNet(input_size=5, hidden_size=50)

def main():
    print("🔄 Advanced Cross-Framework Ensemble")
    print("=" * 45)
    
    # Generate data
    X, y = make_regression(n_samples=300, n_features=5, noise=15, random_state=42)
    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create diverse models
    models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        RandomForestRegressor(n_estimators=50, random_state=42),
        GradientBoostingRegressor(n_estimators=50, random_state=42),
        create_pytorch_model
    ]
    
    print(f"\nModels: {len(models)}")
    for i, model in enumerate(models):
        if callable(model):
            print(f"  {i+1}. PyTorch Model (constructor)")
        else:
            print(f"  {i+1}. {type(model).__name__}")
    
    # Create cross-framework ensemble
    uq = UQ(models)
    
    # Get ensemble information
    info = uq.get_info()
    print(f"\nEnsemble Info:")
    print(f"  Model type: {info['model_type']}")
    print(f"  Method: {info['method']}")
    print(f"  Number of models: {info['n_models']}")
    
    # Fit ensemble
    print(f"\nTraining ensemble...")
    uq.fit(X_train, y_train)
    
    # Get predictions
    predictions, intervals = uq.predict(X_test, return_interval=True)
    
    # Evaluate
    from ueq import coverage, sharpness, expected_calibration_error
    cov = coverage(y_test, intervals)
    sharp = sharpness(intervals)
    ece = expected_calibration_error(y_test, intervals)
    
    print(f"\n📊 Results")
    print("-" * 15)
    print(f"Coverage: {cov:.3f}")
    print(f"Sharpness: {sharp:.3f}")
    print(f"ECE: {ece:.3f}")
    
    # Get detailed ensemble info
    ensemble_info = uq.uq_model.get_model_info()
    print(f"\n🔍 Ensemble Details")
    print("-" * 20)
    print(f"Successfully fitted: {ensemble_info['n_models']} models")
    print(f"Aggregation method: {ensemble_info['aggregation_method']}")
    print(f"Weights: {ensemble_info['weights']}")
    
    for i, model_info in enumerate(ensemble_info['models']):
        print(f"  Model {i+1}: {model_info['model_class']} ({model_info['model_type']})")
    
    # Compare with individual models
    print(f"\n🔬 Individual Model Comparison")
    print("-" * 35)
    
    individual_results = {}
    
    for i, model in enumerate(models):
        try:
            if callable(model):
                # PyTorch model
                individual_uq = UQ(model, method="deep_ensemble", n_models=3)
                individual_uq.fit(X_train, y_train)
                pred, unc = individual_uq.predict(X_test, return_interval=True)
            else:
                # Sklearn model
                individual_uq = UQ(model, method="bootstrap", n_models=20)
                individual_uq.fit(X_train, y_train)
                pred, unc = individual_uq.predict(X_test, return_interval=True)
            
            individual_cov = coverage(y_test, unc)
            individual_sharp = sharpness(unc)
            
            individual_results[i] = {
                'coverage': individual_cov,
                'sharpness': individual_sharp
            }
            
            print(f"  Model {i+1}: Coverage={individual_cov:.3f}, Sharpness={individual_sharp:.3f}")
            
        except Exception as e:
            print(f"  Model {i+1}: Failed - {e}")
            individual_results[i] = None
    
    # Ensemble vs individual comparison
    print(f"\n📈 Ensemble vs Individual Models")
    print("-" * 35)
    
    print(f"Ensemble: Coverage={cov:.3f}, Sharpness={sharp:.3f}")
    
    for i, result in individual_results.items():
        if result is not None:
            model_name = "PyTorch" if callable(models[i]) else type(models[i]).__name__
            print(f"{model_name:15}: Coverage={result['coverage']:.3f}, Sharpness={result['sharpness']:.3f}")
    
    # Determine best approach
    if cov > max([r['coverage'] for r in individual_results.values() if r is not None]):
        print(f"\n✅ Ensemble provides better coverage than individual models")
    else:
        print(f"\n⚠️  Individual models may provide better coverage")
    
    print(f"\n🎉 Advanced cross-framework ensemble completed!")

if __name__ == "__main__":
    main()
```

**Key Features:**
- Multiple model types
- Individual vs ensemble comparison
- Advanced ensemble analysis
- Performance evaluation

This comprehensive examples documentation covers all the major use cases and features of UEQ, from basic usage to advanced production scenarios. Each example is designed to be educational and immediately runnable.
