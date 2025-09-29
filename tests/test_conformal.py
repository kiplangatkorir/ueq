import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from ueq.methods.conformal import ConformalUQ


def test_conformal_intervals_cover():
    X, y = make_regression(n_samples=200, n_features=3, noise=5, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = LinearRegression()
    uq = ConformalUQ(model, alpha=0.1)
    uq.fit(X_train, y_train, X_calib, y_calib)
    preds, intervals = uq.predict(X_test, return_interval=True)

    # Coverage check: ~90% should fall in interval
    covered = sum(low <= yt <= high for yt, (_, (low, high)) in zip(y_test, zip(preds, intervals)))
    coverage = covered / len(y_test)
    assert 0.7 <= coverage <= 1.0  # allow slack due to randomness
