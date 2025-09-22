import matplotlib.pyplot as plt
import numpy as np

def plot_intervals(
    X: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    intervals: list,
    title: str = "UQ Prediction Intervals"
) -> None:
    """Plot predictions with uncertainty intervals."""
    if X.shape[1] != 1:
        raise ValueError("Plotting only supports 1D inputs")
        
    X_sorted_idx = np.argsort(X[:, 0])
    X_plot = X[X_sorted_idx]
    mean_plot = mean[X_sorted_idx]
    lower = np.array([lo for lo, hi in intervals])[X_sorted_idx]
    upper = np.array([hi for lo, hi in intervals])[X_sorted_idx]

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="black", label="True")
    plt.plot(X_plot, mean_plot, color="blue", label="Pred mean")
    plt.fill_between(X_plot[:, 0], lower, upper, 
                    color="blue", alpha=0.2, label="Interval")
    plt.legend()
    plt.title(title)
    plt.show()