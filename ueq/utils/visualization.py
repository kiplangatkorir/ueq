import numpy as np
import matplotlib.pyplot as plt


def plot_predictions_with_intervals(X, y_true, mean_pred, intervals, title="UQ Prediction Intervals"):
    """
    Plot predictive mean and intervals against true values.
    """
    X = np.array(X)
    y_true = np.array(y_true)
    mean_pred = np.array(mean_pred)
    intervals = np.array(intervals)

    # Sort by first feature for clarity
    sort_idx = np.argsort(X[:, 0])
    X_sorted = X[sort_idx]
    y_sorted = y_true[sort_idx]
    mean_sorted = mean_pred[sort_idx]
    lower = intervals[:, 0][sort_idx]
    upper = intervals[:, 1][sort_idx]

    plt.figure(figsize=(8, 5))
    plt.scatter(X_sorted[:, 0], y_sorted, color="black", label="True")
    plt.plot(X_sorted[:, 0], mean_sorted, color="blue", label="Pred mean")
    plt.fill_between(X_sorted[:, 0], lower, upper, color="blue", alpha=0.2, label="Interval")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_calibration_curve(intervals, y_true, confidence=0.95, n_bins=10, title="Calibration Curve"):
    """
    Plot a calibration curve (reliability diagram) for prediction intervals.

    Parameters
    ----------
    intervals : list of tuple
        Prediction intervals [(lower, upper), ...].
    y_true : np.ndarray
        True values (n_samples,).
    confidence : float
        Expected confidence level (e.g., 0.95 for 95% intervals).
    n_bins : int
        Number of bins for empirical coverage.
    title : str
        Plot title.
    """
    y_true = np.array(y_true)
    intervals = np.array(intervals)
    lower, upper = intervals[:, 0], intervals[:, 1]

    # Empirical coverage
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = covered.mean()

    # Create bins of nominal coverage
    nominal_coverages = np.linspace(0, 1, n_bins)
    empirical_coverages = []

    for q in nominal_coverages:
        # shrink interval according to quantile q
        width = (upper - lower) * (1 - q)
        mid = (upper + lower) / 2
        shrunk_lower = mid - width / 2
        shrunk_upper = mid + width / 2
        covered_q = (y_true >= shrunk_lower) & (y_true <= shrunk_upper)
        empirical_coverages.append(covered_q.mean())

    # Plot reliability diagram
    plt.figure(figsize=(6, 6))
    plt.plot(nominal_coverages, empirical_coverages, marker="o", label="Empirical")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title
