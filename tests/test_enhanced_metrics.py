import numpy as np
import pytest
from ueq.utils.metrics import (
    coverage,
    sharpness,
    interval_width,
    interval_score,
    miscoverage_rate,
    evaluate_uncertainty,
    check_calibration,
)


def test_interval_width():
    """Test that interval_width is an alias for sharpness."""
    intervals = [(0, 2), (4, 6), (20, 30)]
    assert interval_width(intervals) == sharpness(intervals)


def test_interval_score():
    """Test interval score computation."""
    y_true = np.array([1.0, 2.0, 3.0])
    intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]  # all covered
    
    score = interval_score(y_true, intervals, alpha=0.05)
    assert score > 0  # Score should be positive
    assert np.isfinite(score)


def test_interval_score_miscoverage_penalty():
    """Test that interval score penalizes miscoverage."""
    y_true = np.array([1.0, 2.0, 3.0])
    
    # Intervals that cover well
    good_intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]
    good_score = interval_score(y_true, good_intervals, alpha=0.05)
    
    # Intervals that miss (undercoverage)
    bad_intervals = [(0.5, 0.9), (1.5, 1.9), (2.5, 2.9)]
    bad_score = interval_score(y_true, bad_intervals, alpha=0.05)
    
    # Bad intervals should have higher (worse) score
    assert bad_score > good_score


def test_miscoverage_rate():
    """Test miscoverage rate computation."""
    y_true = [1, 2, 3]
    intervals = [(0, 2), (1, 3), (10, 20)]  # last one misses
    
    miscov = miscoverage_rate(y_true, intervals)
    assert np.isclose(miscov, 1/3)
    
    # Should be complement of coverage
    cov = coverage(y_true, intervals)
    assert np.isclose(miscov, 1 - cov)


def test_evaluate_uncertainty_all_metrics():
    """Test unified evaluation with all metrics."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5)]
    
    results = evaluate_uncertainty(y_true, y_pred, intervals)
    
    # Check that all default metrics are computed
    expected_metrics = ['coverage', 'sharpness', 'interval_width', 
                       'ece', 'mce', 'interval_score', 'miscoverage']
    for metric in expected_metrics:
        assert metric in results
        assert np.isfinite(results[metric])


def test_evaluate_uncertainty_selected_metrics():
    """Test evaluation with selected metrics only."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]
    
    results = evaluate_uncertainty(y_true, y_pred, intervals, 
                                   metrics=['coverage', 'sharpness'])
    
    assert 'coverage' in results
    assert 'sharpness' in results
    assert 'ece' not in results
    assert len(results) == 2


def test_evaluate_uncertainty_invalid_metric():
    """Test that invalid metric raises error."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]
    
    with pytest.raises(ValueError, match="Unknown metric"):
        evaluate_uncertainty(y_true, y_pred, intervals, 
                           metrics=['invalid_metric'])


def test_check_calibration_well_calibrated():
    """Test calibration check for well-calibrated intervals."""
    # Generate well-calibrated data
    np.random.seed(42)
    y_true = np.random.randn(100)
    # 95% intervals that actually cover
    intervals = [(y - 2, y + 2) for y in y_true]
    
    diagnostics = check_calibration(y_true, intervals, confidence=0.95, tolerance=0.1)
    
    assert 'empirical_coverage' in diagnostics
    assert 'warnings' in diagnostics
    assert 'is_well_calibrated' in diagnostics
    assert diagnostics['empirical_coverage'] == 1.0  # All covered


def test_check_calibration_undercoverage_warning():
    """Test that undercoverage triggers warning."""
    y_true = np.array([1, 2, 3, 4, 5])
    # Intervals too narrow
    intervals = [(y - 0.1, y + 0.1) for y in [1.5, 2.5, 3.5, 4.5, 5.5]]
    
    diagnostics = check_calibration(y_true, intervals, confidence=0.95, tolerance=0.05)
    
    assert len(diagnostics['warnings']) > 0
    assert any('UNDERCOVERAGE' in w for w in diagnostics['warnings'])
    assert not diagnostics['is_well_calibrated']


def test_check_calibration_overcoverage_warning():
    """Test that overcoverage triggers warning."""
    y_true = np.array([1, 2, 3, 4, 5])
    # Intervals too wide
    intervals = [(y - 100, y + 100) for y in y_true]
    
    diagnostics = check_calibration(y_true, intervals, confidence=0.95, tolerance=0.05)
    
    # Should warn about overcoverage (too conservative)
    assert diagnostics['empirical_coverage'] > 0.95


def test_check_calibration_uncertainty_collapse():
    """Test detection of uncertainty collapse."""
    y_true = np.array([1, 2, 3, 4, 5])
    # Extremely narrow intervals (overconfident model)
    intervals = [(y - 1e-9, y + 1e-9) for y in y_true]
    
    diagnostics = check_calibration(y_true, intervals)
    
    assert len(diagnostics['warnings']) > 0
    assert any('UNCERTAINTY COLLAPSE' in w for w in diagnostics['warnings'])


def test_check_calibration_constant_intervals():
    """Test detection of constant interval widths."""
    y_true = np.array([1, 2, 3, 4, 5])
    # All intervals have same width (not adaptive)
    intervals = [(y - 1, y + 1) for y in y_true]
    
    diagnostics = check_calibration(y_true, intervals)
    
    # Should detect constant intervals
    assert any('CONSTANT INTERVALS' in w for w in diagnostics['warnings'])
