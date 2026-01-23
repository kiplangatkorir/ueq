import numpy as np
from typing import Dict, List, Tuple, Union, Optional


def coverage(y_true: Union[np.ndarray, List], 
            intervals: Union[np.ndarray, List[Tuple]]) -> float:
    """
    Compute coverage: fraction of true values inside prediction intervals.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True target values.
    intervals : list of tuples
        Prediction intervals [(lower, upper), ...].

    Returns
    -------
    cov : float
        Fraction of points within intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.array([iv[0] for iv in intervals])
    upper = np.array([iv[1] for iv in intervals])

    return np.mean((y_true >= lower) & (y_true <= upper))


def sharpness(intervals: Union[np.ndarray, List[Tuple]]) -> float:
    """
    Compute sharpness: average width of prediction intervals.

    Parameters
    ----------
    intervals : list of tuples
        Prediction intervals [(lower, upper), ...].

    Returns
    -------
    sharp : float
        Mean interval width.
    """
    lower = np.array([iv[0] for iv in intervals])
    upper = np.array([iv[1] for iv in intervals])

    return np.mean(upper - lower)


def expected_calibration_error(y_true: Union[np.ndarray, List], 
                              intervals: Union[np.ndarray, List[Tuple]], 
                              n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) for prediction intervals.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True values.
    intervals : list of tuples
        Prediction intervals [(lower, upper), ...].
    n_bins : int
        Number of bins to compute calibration.

    Returns
    -------
    ece : float
        Expected calibration error.
    """
    y_true = np.asarray(y_true)
    n = len(y_true)

    # empirical coverage for each point
    covered = np.array([(l <= y <= u) for y, (l, u) in zip(y_true, intervals)])

    # bin edges for expected coverage
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(np.linspace(0, 1, n), bin_edges) - 1

    ece = 0.0
    for b in range(n_bins):
        in_bin = (bin_ids == b)
        if np.any(in_bin):
            acc = covered[in_bin].mean()
            conf = (bin_edges[b] + bin_edges[b + 1]) / 2
            ece += np.abs(acc - conf) * in_bin.mean()

    return ece


def maximum_calibration_error(y_true: Union[np.ndarray, List], 
                             intervals: Union[np.ndarray, List[Tuple]], 
                             n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    Parameters
    ----------
    y_true : array-like
        True values.
    intervals : list of tuples
        Prediction intervals.
    n_bins : int
        Number of bins.

    Returns
    -------
    mce : float
        Maximum calibration error across bins.
    """
    y_true = np.asarray(y_true)
    n = len(y_true)

    covered = np.array([(l <= y <= u) for y, (l, u) in zip(y_true, intervals)])
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(np.linspace(0, 1, n), bin_edges) - 1

    errors = []
    for b in range(n_bins):
        in_bin = (bin_ids == b)
        if np.any(in_bin):
            acc = covered[in_bin].mean()
            conf = (bin_edges[b] + bin_edges[b + 1]) / 2
            errors.append(np.abs(acc - conf))

    return max(errors) if errors else 0.0


def interval_width(intervals: Union[np.ndarray, List[Tuple]]) -> float:
    """
    Compute mean interval width (alias for sharpness for clarity).
    
    Parameters
    ----------
    intervals : list of tuples or array-like
        Prediction intervals [(lower, upper), ...].
    
    Returns
    -------
    width : float
        Mean interval width.
    """
    return sharpness(intervals)


def interval_score(y_true: Union[np.ndarray, List], 
                  intervals: Union[np.ndarray, List[Tuple]], 
                  alpha: float = 0.05) -> float:
    """
    Compute interval score (a proper scoring rule for prediction intervals).
    
    Lower is better. Penalizes both width and miscoverage.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True values.
    intervals : list of tuples or array-like
        Prediction intervals [(lower, upper), ...].
    alpha : float
        Miscoverage level (e.g., 0.05 for 95% intervals).
    
    Returns
    -------
    score : float
        Mean interval score.
    """
    y_true = np.asarray(y_true)
    lower = np.array([iv[0] for iv in intervals])
    upper = np.array([iv[1] for iv in intervals])
    
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    
    # Interval score formula
    scores = width + (2.0 / alpha) * ((lower - y_true) * below + 
                                       (y_true - upper) * above)
    
    return np.mean(scores)


def miscoverage_rate(y_true: Union[np.ndarray, List], 
                    intervals: Union[np.ndarray, List[Tuple]]) -> float:
    """
    Compute miscoverage rate (fraction of points outside intervals).
    
    Parameters
    ----------
    y_true : array-like
        True values.
    intervals : list of tuples or array-like
        Prediction intervals.
    
    Returns
    -------
    miscoverage : float
        Fraction of points outside intervals (1 - coverage).
    """
    return 1.0 - coverage(y_true, intervals)


def evaluate_uncertainty(y_true, y_pred, intervals, 
                        metrics: Optional[List[str]] = None,
                        alpha: float = 0.05,
                        n_bins: int = 10) -> Dict[str, float]:
    """
    Unified function to evaluate uncertainty quantification quality.
    
    Computes multiple standard UQ metrics in one call.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True target values.
    y_pred : array-like, shape (n_samples,)
        Point predictions (mean/median).
    intervals : list of tuples or array-like, shape (n_samples, 2)
        Prediction intervals [(lower, upper), ...].
    metrics : list of str, optional
        Metrics to compute. If None, computes all available metrics.
        Available: ['coverage', 'sharpness', 'interval_width', 'ece', 'calibration' (alias for ece), 
                   'mce', 'interval_score', 'miscoverage']
    alpha : float
        Significance level for interval score (default: 0.05 for 95% intervals).
    n_bins : int
        Number of bins for calibration error metrics.
    
    Returns
    -------
    results : dict
        Dictionary mapping metric names to computed values.
    """
    if metrics is None:
        metrics = ['coverage', 'sharpness', 'interval_width', 'ece', 'mce', 
                  'interval_score', 'miscoverage']
    
    available_metrics = {
        'coverage': lambda: coverage(y_true, intervals),
        'sharpness': lambda: sharpness(intervals),
        'interval_width': lambda: interval_width(intervals),
        'ece': lambda: expected_calibration_error(y_true, intervals, n_bins),
        'calibration': lambda: expected_calibration_error(y_true, intervals, n_bins),  # Alias for ece
        'mce': lambda: maximum_calibration_error(y_true, intervals, n_bins),
        'interval_score': lambda: interval_score(y_true, intervals, alpha),
        'miscoverage': lambda: miscoverage_rate(y_true, intervals),
    }
    
    results = {}
    for metric in metrics:
        if metric in available_metrics:
            results[metric] = available_metrics[metric]()
        else:
            raise ValueError(f"Unknown metric: {metric}. "
                           f"Available: {list(available_metrics.keys())}")
    
    return results


def check_calibration(y_true: Union[np.ndarray, List], 
                     intervals: Union[np.ndarray, List[Tuple]], 
                     confidence: float = 0.95, 
                     tolerance: float = 0.05) -> Dict[str, Union[float, List[str], bool]]:
    """
    Check if prediction intervals are well-calibrated.
    
    Returns diagnostic information and warnings.
    
    Parameters
    ----------
    y_true : array-like
        True values.
    intervals : list of tuples or array-like
        Prediction intervals.
    confidence : float
        Expected nominal coverage (e.g., 0.95).
    tolerance : float
        Acceptable deviation from nominal coverage.
    
    Returns
    -------
    diagnostics : dict
        Dictionary with calibration diagnostics and warnings.
    """
    empirical_coverage = coverage(y_true, intervals)
    mean_width = sharpness(intervals)
    miscov = miscoverage_rate(y_true, intervals)
    
    diagnostics = {
        'empirical_coverage': empirical_coverage,
        'nominal_coverage': confidence,
        'coverage_error': empirical_coverage - confidence,
        'mean_interval_width': mean_width,
        'miscoverage_rate': miscov,
        'warnings': []
    }
    
    # Check for calibration issues
    if abs(empirical_coverage - confidence) > tolerance:
        if empirical_coverage < confidence:
            diagnostics['warnings'].append(
                f"UNDERCOVERAGE: Empirical coverage ({empirical_coverage:.3f}) "
                f"is below nominal ({confidence:.3f}). Intervals may be too narrow."
            )
        else:
            diagnostics['warnings'].append(
                f"OVERCOVERAGE: Empirical coverage ({empirical_coverage:.3f}) "
                f"exceeds nominal ({confidence:.3f}). Intervals may be too conservative."
            )
    
    # Check for uncertainty collapse
    if mean_width < 1e-6:
        diagnostics['warnings'].append(
            f"UNCERTAINTY COLLAPSE: Mean interval width ({mean_width:.6f}) "
            "is extremely small. Model may be overconfident."
        )
    
    # Check if all intervals are identical (possible bug)
    lower = np.array([iv[0] for iv in intervals])
    upper = np.array([iv[1] for iv in intervals])
    if np.std(upper - lower) < 1e-6:
        diagnostics['warnings'].append(
            "CONSTANT INTERVALS: All intervals have similar width. "
            "Uncertainty may not be adaptive to input."
        )
    
    diagnostics['is_well_calibrated'] = len(diagnostics['warnings']) == 0
    
    return diagnostics
