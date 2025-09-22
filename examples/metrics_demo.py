import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from ueq import UQ, coverage, sharpness, expected_calibration_error, maximum_calibration_error


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
    ece = expected_calibration_error(y_test, intervals, n_bins=5)
    mce = maximum_calibration_error(y_test, intervals, n_bins=5)

    print("📊 UQ Metrics")
    print(f"Coverage: {cov:.3f}")
    print(f"Sharpness: {sharp:.3f}")
    print(f"ECE: {ece:.3f}")
    print(f"MCE: {mce:.3f}")

    # visualization
    plt.figure(figsize=(8, 5))
    X_test_sorted_idx = np.argsort(X_test[:, 0])
    X_plot = X_test[X_test_sorted_idx]
    mean_plot = mean_pred[X_test_sorted_idx]
    lower = np.array([lo for lo, hi in intervals])[X_test_sorted_idx]
    upper = np.array([hi for lo, hi in intervals])[X_test_sorted_idx]

    plt.scatter(X_test, y_test, color="black", label="True")
    plt.plot(X_plot, mean_plot, color="blue", label="Pred mean")
    plt.fill_between(X_plot[:, 0], lower, upper, color="blue", alpha=0.2, label="Interval")
    plt.legend()
    plt.title("Uncertainty Quantification with Bayesian Linear Regression")
    plt.show()


if __name__ == "__main__":
    main()
