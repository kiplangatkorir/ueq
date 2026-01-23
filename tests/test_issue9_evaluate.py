"""
Tests for Issue #9: Standardized UQ Evaluation Metrics

Validates that the evaluate() function works across different UQ methods
and provides a unified interface for evaluating uncertainty quantification.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

import ueq


def test_evaluate_function_accessible():
    """Test that ueq.evaluate is accessible from the main module."""
    assert hasattr(ueq, 'evaluate')
    assert callable(ueq.evaluate)


def test_evaluate_with_calibration_metric():
    """Test that 'calibration' metric works as specified in Issue #9."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5)]
    
    results = ueq.evaluate(y_true, y_pred, intervals, metrics=['coverage', 'sharpness', 'calibration'])
    
    assert 'coverage' in results
    assert 'sharpness' in results
    assert 'calibration' in results
    assert all(isinstance(v, (int, float, np.number)) for v in results.values())


def test_evaluate_all_requested_metrics():
    """Test all metrics requested in Issue #9."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5)]
    
    # Metrics mentioned in Issue #9
    requested_metrics = ['coverage', 'sharpness', 'calibration']
    results = ueq.evaluate(y_true, y_pred, intervals, metrics=requested_metrics)
    
    # Verify all requested metrics are in the results
    for metric in requested_metrics:
        assert metric in results, f"Metric {metric} not found in results"
        assert np.isfinite(results[metric]), f"Metric {metric} is not finite"


def test_evaluate_with_bootstrap_uq():
    """Test evaluate() works with Bootstrap UQ method."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Use bootstrap method with correct parameter name
    model = LinearRegression()
    uq = ueq.UQ(model, method='bootstrap', n_models=10)
    uq.fit(X_train, y_train)
    
    y_pred, intervals = uq.predict(X_test, return_interval=True)
    
    # Evaluate using the new API
    results = ueq.evaluate(y_test, y_pred, intervals, 
                          metrics=['coverage', 'sharpness', 'calibration'])
    
    assert 'coverage' in results
    assert 0 <= results['coverage'] <= 1
    assert results['sharpness'] > 0
    assert 0 <= results['calibration'] <= 1


def test_evaluate_with_conformal_prediction():
    """Test evaluate() works with Conformal Prediction."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    # Split for training, calibration, and testing
    X_train, X_calib, X_test = X[:40], X[40:70], X[70:]
    y_train, y_calib, y_test = y[:40], y[40:70], y[70:]
    
    # Use conformal prediction with proper calibration split
    model = LinearRegression()
    uq = ueq.UQ(model, method='conformal')
    uq.fit(X_train, y_train, X_calib, y_calib)
    
    y_pred, intervals = uq.predict(X_test, return_interval=True)
    
    # Evaluate using the new API
    results = ueq.evaluate(y_test, y_pred, intervals,
                          metrics=['coverage', 'sharpness', 'calibration'])
    
    assert 'coverage' in results
    assert 0 <= results['coverage'] <= 1
    assert results['sharpness'] > 0
    assert 0 <= results['calibration'] <= 1


def test_evaluate_consistent_output_format():
    """Test that evaluate() returns consistent dict output."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]
    
    results = ueq.evaluate(y_true, y_pred, intervals, metrics=['coverage'])
    
    # Should return a dict
    assert isinstance(results, dict)
    
    # Should have exactly the metrics requested
    assert len(results) == 1
    assert 'coverage' in results


def test_evaluate_backward_compatibility():
    """Test that evaluate() is backward compatible with evaluate_uncertainty."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5)]
    
    # Both should return the same results
    results1 = ueq.evaluate(y_true, y_pred, intervals, metrics=['coverage', 'sharpness'])
    results2 = ueq.evaluate_uncertainty(y_true, y_pred, intervals, metrics=['coverage', 'sharpness'])
    
    assert results1.keys() == results2.keys()
    for key in results1:
        assert np.isclose(results1[key], results2[key])


def test_evaluate_interval_width_efficiency():
    """Test interval_width metric (interval width efficiency from Issue #9)."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    
    # Narrow intervals
    narrow_intervals = [(0.9, 1.1), (1.9, 2.1), (2.9, 3.1)]
    # Wide intervals
    wide_intervals = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]
    
    narrow_results = ueq.evaluate(y_true, y_pred, narrow_intervals, 
                                   metrics=['interval_width'])
    wide_results = ueq.evaluate(y_true, y_pred, wide_intervals,
                                metrics=['interval_width'])
    
    # Wide intervals should have larger interval_width
    assert wide_results['interval_width'] > narrow_results['interval_width']


def test_evaluate_with_synthetic_data():
    """Test evaluate() with synthetic data (known noise) as mentioned in acceptance criteria."""
    np.random.seed(42)
    
    # Generate synthetic data with known noise level
    n_samples = 100
    noise_level = 1.0
    
    X = np.random.randn(n_samples, 1)
    true_y = 2 * X[:, 0] + 1
    y = true_y + np.random.randn(n_samples) * noise_level
    
    # Create intervals based on known noise
    # For 95% coverage with normal noise, we need ~1.96 * noise_level
    interval_width = 1.96 * noise_level
    intervals = [(yi - interval_width, yi + interval_width) for yi in y]
    
    results = ueq.evaluate(y, y, intervals, metrics=['coverage', 'sharpness'])
    
    # With synthetic data and known noise, coverage should be close to 0.95
    # But since we're using the observed y (not true_y), coverage will be 1.0
    assert results['coverage'] == 1.0
    
    # Sharpness should be approximately 2 * interval_width
    expected_sharpness = 2 * interval_width
    assert np.isclose(results['sharpness'], expected_sharpness, rtol=0.01)


def test_calibration_ece_equivalence():
    """Test that 'calibration' and 'ece' return the same values."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5)]
    
    results_calibration = ueq.evaluate(y_true, y_pred, intervals, metrics=['calibration'])
    results_ece = ueq.evaluate(y_true, y_pred, intervals, metrics=['ece'])
    
    # calibration should be an alias for ece
    assert np.isclose(results_calibration['calibration'], results_ece['ece'])
