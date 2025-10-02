<div align="center">

🌀

# 🔥 Uncertainty Everywhere (UEQ) - Phoenix Edition 🔥 

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache-green.svg)]()
[![Status](https://img.shields.io/badge/status-Phoenix-red)]()
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()

**"Rising from Research to Production"**  
_Bootstrap. Conformal. Dropout. Ensembles. Bayes._  
_Uncertainty for every model, everywhere._

</div>

**A unified Python library for Uncertainty Quantification (UQ).**  
Easily wrap your machine learning models and get predictions **with confidence intervals, coverage guarantees, or Bayesian-style uncertainty** - all from one interface.

**🔥 UEQ v1.0.1 Phoenix - "Rising from Research to Production" 🔥**

**🚀 NEW in Phoenix: Production-Ready Features!**
- **Auto-detection**: Automatically detects model types and selects optimal UQ methods
- **Cross-framework ensembles**: Combine models from different frameworks (sklearn + PyTorch)
- **Model monitoring**: Real-time drift detection and performance monitoring
- **Performance optimization**: Batch processing and memory-efficient predictions
- **Zero-configuration**: `UQ(model)` just works!

##  Features

*  **One API** for many uncertainty methods  
*  Works with **scikit-learn models** (e.g., LinearRegression, RandomForest)  
*  Works with **PyTorch deep learning models**  
*  Plug-and-play methods:  
  * **Bootstrap** (frequentist ensembles)  
  * **Conformal Prediction** (distribution-free coverage)  
  * **MC Dropout** (Bayesian deep learning approximation)  
  * **Deep Ensembles** (Lakshminarayanan et al., 2017)  
  * **Bayesian Linear Regression** (closed-form Bayesian updates)  
* ✅ Extensible: add new UQ methods without changing user code  


##  Installation

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

##  Quick Start

### 🎯 Auto-Detection (NEW!)

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
from ueq import UQ

# Auto-detects sklearn regressor → uses bootstrap
sklearn_model = LinearRegression()
uq1 = UQ(sklearn_model)  # method="auto" by default

# Auto-detects sklearn classifier → uses conformal prediction  
clf_model = RandomForestClassifier()
uq2 = UQ(clf_model)

# Auto-detects PyTorch model → uses MC dropout
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

pytorch_model = Net()
uq3 = UQ(pytorch_model)

# Auto-detects cross-framework ensemble
models = [sklearn_model, pytorch_model]
uq4 = UQ(models)  # Creates cross-framework ensemble
```

### 1. Bootstrap (scikit-learn)

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from ueq import UQ

X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)

uq = UQ(LinearRegression(), method="bootstrap", n_models=20)
uq.fit(X, y)
preds, intervals = uq.predict(X[:5])

print("Predictions:", preds)
print("Intervals:", intervals)
```

### 2. Conformal Prediction

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

uq = UQ(LinearRegression(), method="conformal", alpha=0.1)
uq.fit(X_train, y_train, X_calib, y_calib)

preds, intervals = uq.predict(X_test[:5], return_interval=True)
print("Predictions:", preds)
print("Intervals:", intervals)
```

### 3. MC Dropout (PyTorch)

```python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(200, 10)
y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(200, 1)
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))

uq = UQ(lambda: Net(), method="mc_dropout", n_forward_passes=100)

criterion = nn.MSELoss()
optimizer = optim.Adam(uq.uq_model.model.parameters(), lr=0.01)
uq.fit(loader, criterion, optimizer, epochs=10)

mean, std = uq.predict(torch.randn(5, 10))
print("Mean predictions:", mean)
print("Uncertainty:", std)
```

### 4. Deep Ensembles

```python
from ueq import UQ
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TinyNet(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

X = torch.randn(100, 3)
y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(100, 1)
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

uq = UQ(lambda: TinyNet(input_dim=3), method="deep_ensemble", n_models=3)
criterion = nn.MSELoss()
optimizer_fn = lambda params: optim.Adam(params, lr=0.01)

uq.fit(loader, criterion, optimizer_fn, epochs=5)
mean, intervals = uq.predict(torch.randn(5, 3))

print("Mean predictions:", mean)
print("Intervals:", intervals)
```

### 5. Bayesian Linear Regression

```python
from sklearn.datasets import make_regression
from ueq import UQ

X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)

uq = UQ(method="bayesian_linear", alpha=2.0, beta=25.0)
uq.fit(X, y)

preds, intervals = uq.predict(X[:5])
print("Predictions:", preds)
print("Intervals:", intervals)
```

### 6. Cross-Framework Ensemble (NEW!)

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch.nn as nn
from ueq import UQ

# Create models from different frameworks
sklearn_model1 = LinearRegression()
sklearn_model2 = RandomForestRegressor()
pytorch_model = Net()  # Your PyTorch model

# Combine them in a cross-framework ensemble
models = [sklearn_model1, sklearn_model2, pytorch_model]
uq = UQ(models)  # Auto-detects cross-framework ensemble

# Fit and predict with unified uncertainty
uq.fit(X_train, y_train)
mean_pred, intervals = uq.predict(X_test, return_interval=True)

print("Cross-framework predictions:", mean_pred)
print("Unified uncertainty intervals:", intervals)
```

### 7. Production Features (NEW!)

```python
from ueq import UQ, UQMonitor, BatchProcessor

# Model monitoring and drift detection
uq = UQ(model)
monitor = UQMonitor(baseline_data=X_train, baseline_uncertainty=baseline_unc)

# Monitor new data
results = uq.monitor(X_new, y_new)
print(f"Drift score: {results['drift_score']:.3f}")
print(f"Alerts: {len(results['alerts'])}")

# Performance optimization for large datasets
batch_processor = BatchProcessor(batch_size=1000, n_jobs=4)
predictions = uq.predict_large_dataset(X_large, batch_size=1000)

# Production deployment
class ProductionUQService:
    def __init__(self, model):
        self.uq = UQ(model)
        self.monitor = UQMonitor()
    
    def predict(self, X):
        predictions, uncertainty = self.uq.predict(X, return_interval=True)
        monitoring = self.monitor.monitor(predictions, uncertainty)
        return predictions, uncertainty, monitoring
```

##  Roadmap

* [x] **Auto-detection system** - Automatically select optimal UQ methods
* [x] **Cross-framework ensembles** - Combine models from different frameworks
* [x] **Production features** - Model monitoring, drift detection, performance optimization
* [x] **Enhanced visualization** - Calibration plots and coverage curves
* [ ] Additional UQ methods (Quantile Regression, Gaussian Processes, Normalizing Flows)
* [ ] TensorFlow/Keras support
* [ ] XGBoost/LightGBM support
* [ ] Documentation website with tutorials
* [x] **Publish to PyPI** - Now available with `pip install ueq`! 🎉

##  Contributing

Pull requests and ideas are welcome!
Whether it's new methods, bug fixes, or docs improvements — let's make UQ accessible everywhere.

### 🐛 Issues and Support

If you encounter any problems or have questions:

1. **Check the [documentation](docs/)** for comprehensive guides and examples
2. **Search existing [issues](https://github.com/kiplangatkorir/ueq/issues)** to see if your problem is already reported
3. **Create a new issue** with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and UEQ version (`ueq.__version__`)
   - Error messages (if any)

**We're here to help!** 🚀

##  License

Licensed under the **Apache License 2.0**.
You may use, modify, and distribute this library in research and production under the terms of the license.
See the [LICENSE](LICENSE) file for details.
