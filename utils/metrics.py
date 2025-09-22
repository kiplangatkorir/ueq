import numpy as np


def coverage(y_true, intervals):
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


def sharpness(intervals):
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


def expected_calibration_error(y_true, intervals, n_bins=10):
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


def maximum_calibration_error(y_true, intervals, n_bins=10):
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
