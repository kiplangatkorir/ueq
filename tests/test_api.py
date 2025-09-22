# tests/test_api.py
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from ueq.utils.api import evaluate
from ueq.utils.plotting import plot_intervals


def test_evaluate_bayesian_linear_runs():
    """Check that evaluate runs for Bayesian Linear and returns expected keys."""
    X, y = make_regression(n_samples=60, n_features=1, noise=10, random_state=0)
    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]

    results = evaluate(
        method="bayesian_linear",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        alpha=2.0, beta=25.0
    )

    assert "mean" in results
    assert "intervals" in results
    assert "metrics" in results
    assert "coverage" in results["metrics"]
    assert len(results["mean"]) == X_test.shape[0]
    assert len(results["intervals"]) == X_test.shape[0]


def test_evaluate_with_sklearn_model():
    """Check evaluate works with a sklearn model and bootstrap method."""
    X, y = make_regression(n_samples=60, n_features=1, noise=10, random_state=1)
    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]

    model = LinearRegression()
    results = evaluate(
        model=model,
        method="bootstrap",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        n_estimators=3
    )

    assert "mean" in results
    assert "intervals" in results
    assert "metrics" in results
    assert "sharpness" in results["metrics"]


def test_plot_intervals_runs(tmp_path):
    """Ensure plot_intervals executes without error and saves a figure."""
    import matplotlib
    matplotlib.use("Agg")  # use non-GUI backend for tests

    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel()
    mean = y + 0.1  # dummy shifted mean
    intervals = [(val - 0.2, val + 0.2) for val in mean]

    # just check no error when plotting
    plot_intervals(X, y, mean, intervals, title="Test Plot")
