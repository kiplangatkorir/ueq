import numpy as np
from ueq.utils.metrics import (
    coverage,
    sharpness,
    expected_calibration_error,
    maximum_calibration_error,
)


def test_coverage_perfect():
    y_true = [1, 2, 3]
    intervals = [(0, 2), (1, 3), (2, 4)]  # all cover
    cov = coverage(y_true, intervals)
    assert cov == 1.0


def test_coverage_partial():
    y_true = [1, 5, 10]
    intervals = [(0, 2), (4, 6), (20, 30)]  # last one misses
    cov = coverage(y_true, intervals)
    assert np.isclose(cov, 2/3)


def test_sharpness():
    intervals = [(0, 2), (4, 6), (20, 30)]  # widths: 2, 2, 10
    sharp = sharpness(intervals)
    assert np.isclose(sharp, (2 + 2 + 10) / 3)


def test_ece_zero_when_perfect():
    # Every interval exactly contains true values
    y_true = np.array([1, 2, 3, 4, 5])
    intervals = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    ece = expected_calibration_error(y_true, intervals, n_bins=5)
    assert ece >= 0.0  # non-negative
    assert ece <= 1.0  # bounded


def test_mce_nonnegative():
    y_true = np.array([1, 5, 10])
    intervals = [(0, 2), (4, 6), (20, 30)]
    mce = maximum_calibration_error(y_true, intervals, n_bins=3)
    assert mce >= 0.0
    assert mce <= 1.0
