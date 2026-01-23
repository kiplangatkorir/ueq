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


def plot_uncertainty_timeline(timestamps, uncertainties, y_true=None, y_pred=None,
                              drift_scores=None, title="Uncertainty Over Time",
                              figsize=(14, 8), window_size=None):
    """
    Plot uncertainty dynamics over time with optional drift signals.
    
    Visualizes how uncertainty evolves in production systems, helping detect
    temporal degradation or instability in confidence estimates.
    
    Parameters
    ----------
    timestamps : array-like, shape (n_samples,)
        Time indices or timestamps for each prediction.
    uncertainties : array-like, shape (n_samples,)
        Uncertainty measures (e.g., interval widths, standard deviations).
        For coverage calculation, these are assumed to be half-widths of
        symmetric intervals: [pred - unc, pred + unc].
    y_true : array-like, shape (n_samples,), optional
        True target values.
    y_pred : array-like, shape (n_samples,), optional
        Predicted values.
    drift_scores : array-like, shape (n_samples,), optional
        Drift detection scores to overlay on the plot.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    window_size : int, optional
        Rolling window size for smoothed uncertainty trends.
    
    Notes
    -----
    When calculating coverage, this function assumes symmetric prediction
    intervals of the form [y_pred[i] - uncertainties[i], y_pred[i] + uncertainties[i]].
    If your uncertainties are not half-widths, consider preprocessing them.
    """
    timestamps = np.asarray(timestamps)
    uncertainties = np.asarray(uncertainties)
    
    n_plots = 2 if drift_scores is not None else 1
    if y_true is not None and y_pred is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # Plot 1: Uncertainty over time
    axes[ax_idx].plot(timestamps, uncertainties, alpha=0.6, linewidth=1, label='Uncertainty')
    
    # Add rolling average if window_size specified
    if window_size is not None and window_size > 1:
        rolling_mean = np.convolve(uncertainties, np.ones(window_size)/window_size, mode='valid')
        rolling_timestamps = timestamps[window_size-1:]
        axes[ax_idx].plot(rolling_timestamps, rolling_mean, 'r-', linewidth=2, 
                         label=f'Rolling mean (window={window_size})')
    
    axes[ax_idx].axhline(y=np.mean(uncertainties), color='k', linestyle='--', 
                         linewidth=1, alpha=0.5, label='Mean uncertainty')
    axes[ax_idx].fill_between(timestamps, 
                              np.mean(uncertainties) - np.std(uncertainties),
                              np.mean(uncertainties) + np.std(uncertainties),
                              alpha=0.2, color='gray', label='±1 std')
    axes[ax_idx].set_ylabel('Uncertainty', fontsize=12)
    axes[ax_idx].set_title('Uncertainty Dynamics', fontsize=14)
    axes[ax_idx].legend(loc='best')
    axes[ax_idx].grid(True, alpha=0.3)
    ax_idx += 1
    
    # Plot 2: Coverage over time (if true values provided)
    if y_true is not None and y_pred is not None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Compute rolling coverage (assuming uncertainties are interval widths)
        if window_size is None:
            window_size = max(20, len(timestamps) // 20)
        
        rolling_coverage = []
        rolling_times = []
        
        for i in range(window_size, len(timestamps)):
            window_start = i - window_size
            # Assuming symmetric intervals: [pred - unc, pred + unc]
            window_covered = np.abs(y_true[window_start:i] - y_pred[window_start:i]) <= uncertainties[window_start:i]
            rolling_coverage.append(window_covered.mean())
            rolling_times.append(timestamps[i])
        
        axes[ax_idx].plot(rolling_times, rolling_coverage, 'b-', linewidth=2, label='Rolling coverage')
        axes[ax_idx].axhline(y=0.9, color='g', linestyle='--', linewidth=1.5, 
                            alpha=0.7, label='Target (90%)')
        axes[ax_idx].axhline(y=0.95, color='orange', linestyle='--', linewidth=1.5, 
                            alpha=0.7, label='Target (95%)')
        axes[ax_idx].set_ylabel('Coverage', fontsize=12)
        axes[ax_idx].set_title(f'Rolling Coverage (window={window_size})', fontsize=14)
        axes[ax_idx].legend(loc='best')
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_ylim([0, 1.05])
        ax_idx += 1
    
    # Plot 3: Drift scores overlay (if provided)
    if drift_scores is not None:
        drift_scores = np.asarray(drift_scores)
        axes[ax_idx].plot(timestamps, drift_scores, 'r-', linewidth=1.5, label='Drift score')
        axes[ax_idx].axhline(y=0.1, color='orange', linestyle='--', linewidth=1, 
                            alpha=0.7, label='Warning threshold')
        axes[ax_idx].axhline(y=0.2, color='red', linestyle='--', linewidth=1, 
                            alpha=0.7, label='Critical threshold')
        axes[ax_idx].set_ylabel('Drift Score', fontsize=12)
        axes[ax_idx].set_title('Distribution Drift Signals', fontsize=14)
        axes[ax_idx].legend(loc='best')
        axes[ax_idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time / Sample Index', fontsize=12)
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()
