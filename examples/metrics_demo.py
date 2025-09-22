import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from ueq import UQ
from ueq.utils.metrics import coverage, sharpness, expected_calibration_error, maximum_calibration_error
from ueq.utils.visualization import plot_predictions_with_intervals, plot_calibration_curve


def main():
    # synthetic regression dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
    X_train, X_test = X[:70], X[70:]
    y_train, y_test = y[:70], y[70:]

    # use Bayesian Linear Regression UQ
    uq = UQ(method="bayesian_linear", alpha=2.0, beta=25.0)
    uq.fit(X_train, y_train)

    # get predictions with intervals
    mean_pred, intervals = uq.predict(X_test, return_interval=True)

    # compute metrics
    cov = coverage(y_test, intervals)
    sharp = sharpness(intervals)
    ece = expected_calibration_error(y_test, intervals, n_bins=10)
    mce = maximum_calibration_error(y_test, intervals, n_bins=10)

    print("📊 UQ Metrics")
    print(f"Coverage: {cov:.3f}")
    print(f"Sharpness: {sharp:.3f}")
    print(f"ECE: {ece:.3f}")
    print(f"MCE: {mce:.3f}")

    # --- Visualization ---
    plot_predictions_with_intervals(X_test, y_test, mean_pred, intervals,
                                    title="Uncertainty Quantification with Bayesian Linear Regression")

    plot_calibration_curve(intervals, y_test, confidence=0.95,
                           title="Calibration Curve (Bayesian Linear Regression)")


if __name__ == "__main__":
    main()
