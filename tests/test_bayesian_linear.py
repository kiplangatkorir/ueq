import numpy as np
from sklearn.datasets import make_regression
from ueq.core import UQ


def test_bayesian_linear_fit_predict():
    # synthetic regression data
    X, y = make_regression(n_samples=80, n_features=3, noise=0.1, random_state=42)
    X_train, X_test = X[:60], X[60:]
    y_train, y_test = y[:60], y[60:]

    # create Bayesian Linear UQ model
    uq = UQ(method="bayesian_linear", alpha=2.0, beta=25.0)
    uq.fit(X_train, y_train)

    # predictions with intervals
    mean_pred, intervals = uq.predict(X_test, return_interval=True)

    # assertions
    assert mean_pred.shape[0] == X_test.shape[0]
    assert len(intervals) == X_test.shape[0]

    # intervals should be (lower, upper)
    lower, upper = intervals[0]
    assert lower < upper

    # also test predictive distribution
    dist = uq.predict_dist(X_test)
    assert dist.shape[0] == X_test.shape[0]
