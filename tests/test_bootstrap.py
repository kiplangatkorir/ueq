import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from ueq.methods.bootstrap import BootstrapUQ


def test_bootstrap_predict_shapes():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    model = LinearRegression()
    uq = BootstrapUQ(model, n_models=5)

    uq.fit(X, y)
    preds, intervals = uq.predict(X[:10])

    assert preds.shape[0] == 10
    assert len(intervals) == 10
    assert all(len(iv) == 2 for iv in intervals)


def test_bootstrap_interval_contains_mean():
    X, y = make_regression(n_samples=100, n_features=2, noise=5, random_state=42)
    model = LinearRegression()
    uq = BootstrapUQ(model, n_models=5)
    uq.fit(X, y)
    preds, intervals = uq.predict(X[:5])

    for p, (low, high) in zip(preds, intervals):
        assert low <= p <= high
