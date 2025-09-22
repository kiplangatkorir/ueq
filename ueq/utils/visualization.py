import numpy as np
import matplotlib.pyplot as plt


def plot_predictions_with_intervals(X, y_true, mean_pred, intervals, title="UQ Prediction Intervals"):
    """
    Plot predictive mean and intervals against true values.

    Parameters
    ----------
    X : np.ndarray
        Test inputs (n_samples, n_features).
    y_true : np.ndarray
        True target values (n_samples,).
    mean_pred : np.ndarray
        Predicted mean values (n_samples,).
    intervals : list of tuple
        List of (lower, upper) intervals for each sample.
    title : str
        Plot title.
    """
    X = np.array(X)
    y_true = np.array(y_true)
    mean_pred = np.array(mean_pred)
    intervals = np.array(intervals)

    # Sort by X[:, 0] for plotting clarity if X has multiple dims
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
