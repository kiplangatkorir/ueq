<div align="center">

# 🌀 Uncertainty Everywhere (UEQ)

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache-green.svg)]()
[![Status](https://img.shields.io/badge/status-MVP-orange)]()
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()

**Bootstrap. Conformal. Dropout.**  
_Uncertainty for every model, everywhere._

</div>


**A unified Python library for Uncertainty Quantification (UQ).**
Easily wrap your machine learning models and get predictions **with confidence intervals, coverage guarantees, or Bayesian-style uncertainty** — all from one interface.

##  Features

* ✅ **One API** for many uncertainty methods
* ✅ Works with **scikit-learn models** (e.g., LinearRegression, RandomForest)
* ✅ Works with **PyTorch deep learning models**
* ✅ Plug-and-play methods:

  * **Bootstrap** (frequentist ensembles)
  * **Conformal Prediction** (distribution-free coverage)
  * **MC Dropout** (Bayesian deep learning approximation)
* ✅ Extensible: add new UQ methods without changing user code

##  Installation

```bash
git clone https://github.com/kiplangatkorir/ueq/.git
cd ueq
pip install -e .
```

##  Quick Start

### 1. Bootstrap (scikit-learn)

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from ueq import UQ

X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)

uq = UQ(LinearRegression(), method="bootstrap", n_samples=20)
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

preds, intervals = uq.predict(X_test[:5])
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

model = Net()
uq = UQ(model, method="mc_dropout", n_forward_passes=100)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
uq.fit(loader, criterion, optimizer, epochs=10)

mean, std = uq.predict(torch.randn(5, 10))
print("Mean predictions:", mean)
print("Uncertainty:", std)
```

## Roadmap

* [ ] Visualization utilities (calibration plots, coverage curves)
* [ ] Additional UQ methods (Quantile Regression, Bayesian Linear Models, Deep Ensembles)
* [ ] Documentation website with tutorials
* [ ] Publish to PyPI (`pip install ueq`)

## Contributing

Pull requests and ideas are welcome!
Whether it’s new methods, bug fixes, or docs improvements — let’s make UQ accessible everywhere.

## 📜 License
Licensed under the **Apache License 2.0**.  
You may use, modify, and distribute this library in research and production under the terms of the license.  
See the [LICENSE](LICENSE) file for details.

