import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from ueq import UQ


# -------------------------------
# Bootstrap test
# -------------------------------
def test_core_bootstrap():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    base_model = LinearRegression()
    uq = UQ(base_model, method="bootstrap", n_models=5)

    uq.fit(X, y)
    preds, intervals = uq.predict(X[:5])

    assert preds.shape[0] == 5
    assert len(intervals) == 5
    assert all(len(iv) == 2 for iv in intervals)


# -------------------------------
# Conformal test
# -------------------------------
def test_core_conformal():
    X, y = make_regression(n_samples=200, n_features=3, noise=5, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    base_model = LinearRegression()
    uq = UQ(base_model, method="conformal", alpha=0.1)

    uq.fit(X_train, y_train, X_calib, y_calib)
    preds, intervals = uq.predict(X_test[:5])

    assert preds.shape[0] == 5
    assert len(intervals) == 5
    assert all(len(iv) == 2 for iv in intervals)


# -------------------------------
# MC Dropout test
# -------------------------------
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 10)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def test_core_mc_dropout():
    X = torch.randn(50, 5)
    y = torch.sum(X, dim=1, keepdim=True)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)

    model = SmallNet()
    uq = UQ(model, method="mc_dropout", n_forward_passes=5)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    uq.fit(loader, criterion, optimizer, epochs=1)
    mean, std = uq.predict(torch.randn(5, 5))

    assert mean.shape == (5, 1)
    assert std.shape == (5, 1)
    assert (std >= 0).all()
