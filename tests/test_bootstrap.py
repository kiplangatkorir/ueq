import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from ueq.methods.bootstrap import BootstrapUQ


def test_bootstrap_predict_shapes():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    model = LinearRegression()
    # Changed from n_models=5 to n_samples=5
    uq = BootstrapUQ(model, n_samples=5)
    uq.fit(X, y)
    
    X_test = np.random.randn(10, 5)
    pred, intervals = uq.predict(X_test)
    
    # Convert intervals from list of tuples to numpy array for shape checking
    intervals = np.array(intervals)
    assert pred.shape == (10,)
    assert intervals.shape == (10, 2)


def test_bootstrap_interval_contains_mean():
    X, y = make_regression(n_samples=100, n_features=2, noise=5, random_state=42)
    model = LinearRegression()
    # Changed from n_models=5 to n_samples=5
    uq = BootstrapUQ(model, n_samples=5)
    uq.fit(X, y)
    
    X_test = np.random.randn(10, 2)
    pred, intervals = uq.predict(X_test)
    intervals = np.array(intervals)
    
    assert np.all(intervals[:, 0] <= pred)
    assert np.all(intervals[:, 1] >= pred)
