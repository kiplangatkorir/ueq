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
        # Scale interval width to nominal coverage q
        width = (upper - lower) * q
        mid = (upper + lower) / 2
        scaled_lower = mid - width / 2
        scaled_upper = mid + width / 2
        covered_q = (y_true >= scaled_lower) & (y_true <= scaled_upper)
        empirical_coverages.append(covered_q.mean())

    # Plot reliability diagram
    plt.figure(figsize=(6, 6))
    plt.plot(nominal_coverages, empirical_coverages, marker="o", label="Empirical")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_intervals(x, y_pred, lower, upper, y_true=None, 
                   title="Prediction Intervals", xlabel="X", ylabel="Y",
                   figsize=(10, 6), alpha=0.3, show_points=True):
    """
    Plot prediction intervals with optional true values.
    
    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Input values (for x-axis).
    y_pred : array-like, shape (n_samples,)
        Point predictions (mean).
    lower : array-like, shape (n_samples,)
        Lower bounds of prediction intervals.
    upper : array-like, shape (n_samples,)
        Upper bounds of prediction intervals.
    y_true : array-like, shape (n_samples,), optional
        True target values.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : tuple
        Figure size.
    alpha : float
        Transparency of interval bands.
    show_points : bool
        Whether to show prediction points.
    """
    x = np.asarray(x).flatten()
    y_pred = np.asarray(y_pred).flatten()
    lower = np.asarray(lower).flatten()
    upper = np.asarray(upper).flatten()
    
    # Sort by x for cleaner visualization
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    lower_sorted = lower[sort_idx]
    upper_sorted = upper[sort_idx]
    
    plt.figure(figsize=figsize)
    
    # Plot prediction intervals
    plt.fill_between(x_sorted, lower_sorted, upper_sorted, 
                     alpha=alpha, color='blue', label='Prediction interval')
    
    # Plot predictions
    if show_points:
        plt.plot(x_sorted, y_pred_sorted, 'b-', linewidth=2, label='Predictions')
    
    # Plot true values if provided
    if y_true is not None:
        y_true = np.asarray(y_true).flatten()
        y_true_sorted = y_true[sort_idx]
        plt.scatter(x_sorted, y_true_sorted, color='red', s=30, 
                   alpha=0.6, label='True values', zorder=5)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_reliability_diagram(y_true, y_pred, intervals, n_bins=10, 
                             confidence=0.95, title="Reliability Diagram",
                             figsize=(8, 8)):
    """
    Plot reliability diagram showing nominal vs empirical coverage.
    
    Also known as a calibration curve for prediction intervals.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True target values.
    y_pred : array-like, shape (n_samples,)
        Point predictions (not used for intervals, kept for API consistency).
    intervals : array-like, shape (n_samples, 2) or list of tuples
        Prediction intervals [(lower, upper), ...].
    n_bins : int
        Number of bins for empirical coverage calculation.
    confidence : float
        Expected nominal confidence level (e.g., 0.95).
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    """
    y_true = np.asarray(y_true)
    if isinstance(intervals, list):
        intervals = np.array(intervals)
    
    lower = intervals[:, 0]
    upper = intervals[:, 1]
    
    # Calculate coverage at different nominal levels
    nominal_levels = np.linspace(0, 1, n_bins)
    empirical_coverages = []
    
    for nom_level in nominal_levels:
        # Scale intervals to nominal level
        width = (upper - lower) * nom_level
        center = (upper + lower) / 2
        scaled_lower = center - width / 2
        scaled_upper = center + width / 2
        
        # Calculate empirical coverage
        covered = (y_true >= scaled_lower) & (y_true <= scaled_upper)
        empirical_coverages.append(covered.mean())
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Reliability diagram
    ax1.plot(nominal_levels, empirical_coverages, 'o-', 
            linewidth=2, markersize=8, label='Empirical coverage')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax1.fill_between([0, 1], [0, 1], alpha=0.1, color='green')
    ax1.set_xlabel('Nominal Coverage', fontsize=12)
    ax1.set_ylabel('Empirical Coverage', fontsize=12)
    ax1.set_title('Reliability Diagram', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Calibration error bars
    calibration_errors = np.array(empirical_coverages) - nominal_levels
    ax2.bar(range(len(calibration_errors)), calibration_errors, 
           color=['red' if e < 0 else 'green' for e in calibration_errors],
           alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Bin Index', fontsize=12)
    ax2.set_ylabel('Calibration Error\n(Empirical - Nominal)', fontsize=12)
    ax2.set_title('Calibration Error by Bin', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_coverage_vs_confidence(y_true, intervals, n_bins=20,
                                title="Coverage vs Interval Width",
                                figsize=(10, 6)):
    """
    Plot empirical coverage against interval width.
    
    Helps diagnose if wider intervals have better coverage (as expected).
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True target values.
    intervals : array-like, shape (n_samples, 2) or list of tuples
        Prediction intervals.
    n_bins : int
        Number of bins for grouping by interval width.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    """
    y_true = np.asarray(y_true)
    if isinstance(intervals, list):
        intervals = np.array(intervals)
    
    lower = intervals[:, 0]
    upper = intervals[:, 1]
    widths = upper - lower
    covered = (y_true >= lower) & (y_true <= upper)
    
    # Bin by interval width
    width_bins = np.percentile(widths, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(widths, width_bins[1:-1])
    
    bin_widths = []
    bin_coverages = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_widths.append(widths[mask].mean())
            bin_coverages.append(covered[mask].mean())
            bin_counts.append(mask.sum())
    
    # Create the plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(bin_widths, bin_coverages, s=[c*2 for c in bin_counts],
                         alpha=0.6, c=bin_coverages, cmap='RdYlGn', 
                         vmin=0, vmax=1, edgecolors='black')
    plt.colorbar(scatter, label='Coverage')
    plt.xlabel('Mean Interval Width', fontsize=12)
    plt.ylabel('Empirical Coverage', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
