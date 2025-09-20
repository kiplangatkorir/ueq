import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from ueq.methods.bootstrap import BootstrapUQ


def test_bootstrap_predict_shapes():
    # Changed n_models to n_samples in make_regression
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    model = LinearRegression()
    # Use n_models to match BootstrapUQ parameter
    uq = BootstrapUQ(model, n_models=5)
    uq.fit(X, y)
    
    X_test = np.random.randn(10, 5)
    pred, intervals = uq.predict(X_test)
    
    assert pred.shape == (10,)
    assert len(intervals) == 10
    assert all(len(interval) == 2 for interval in intervals)


def test_bootstrap_interval_contains_mean():
    X, y = make_regression(n_samples=100, n_features=2, noise=5, random_state=42)
    model = LinearRegression()
    # Use n_models to match BootstrapUQ parameter
    uq = BootstrapUQ(model, n_models=5)
    uq.fit(X, y)
    
    X_test = np.random.randn(10, 2)
    pred, intervals = uq.predict(X_test)
    intervals = np.array(intervals)
    
    assert np.all(intervals[:, 0] <= pred)
    assert np.all(intervals[:, 1] >= pred)
