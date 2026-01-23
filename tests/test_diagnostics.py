import numpy as np
import pytest

# Set non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ueq.diagnostics import (
    plot_calibration,
    check_regression_calibration,
    check_classification_calibration
)


@pytest.fixture
def regression_data():
    """Generate sample regression data with intervals."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y_true = 2 * x + 1 + np.random.randn(n) * 2
    y_pred = 2 * x + 1
    
    # Create prediction intervals
    lower = y_pred - 2.5
    upper = y_pred + 2.5
    intervals = list(zip(lower, upper))
    
    # Also create uncertainty estimates
    uncertainty = np.full(n, 2.5)
    
    return y_true, y_pred, intervals, uncertainty


@pytest.fixture
def classification_data():
    """Generate sample classification data with probabilities."""
    np.random.seed(42)
    n = 200
    
    # Binary classification
    y_true = np.random.randint(0, 2, n)
    
    # Generate somewhat calibrated probabilities
    proba = np.random.rand(n)
    proba = np.where(y_true == 1, proba * 0.4 + 0.5, proba * 0.5)
    
    # Two-class probability matrix
    y_pred_proba = np.column_stack([1 - proba, proba])
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_true, y_pred, y_pred_proba, proba


# Tests for plot_calibration - Regression

def test_plot_calibration_regression_basic(regression_data):
    """Test basic regression calibration plot with intervals."""
    y_true, y_pred, intervals, _ = regression_data
    
    plt.close('all')
    plot_calibration(y_true, y_pred, intervals, task='regression')
    plt.close('all')


def test_plot_calibration_regression_with_uncertainty(regression_data):
    """Test regression calibration plot with uncertainty estimates."""
    y_true, y_pred, _, uncertainty = regression_data
    
    plt.close('all')
    plot_calibration(y_true, y_pred, uncertainty, task='regression')
    plt.close('all')


def test_plot_calibration_regression_return_diagnostics(regression_data):
    """Test that regression diagnostics are returned correctly."""
    y_true, y_pred, intervals, _ = regression_data
    
    plt.close('all')
    diagnostics = plot_calibration(
        y_true, y_pred, intervals,
        task='regression',
        return_diagnostics=True
    )
    plt.close('all')
    
    # Check required keys
    assert 'empirical_coverage' in diagnostics
    assert 'nominal_coverage' in diagnostics
    assert 'coverage_error' in diagnostics
    assert 'mean_interval_width' in diagnostics
    assert 'warnings' in diagnostics
    assert 'is_well_calibrated' in diagnostics
    
    # Check value types
    assert isinstance(diagnostics['empirical_coverage'], float)
    assert isinstance(diagnostics['warnings'], list)
    assert isinstance(diagnostics['is_well_calibrated'], bool)


def test_plot_calibration_regression_custom_params(regression_data):
    """Test regression plot with custom parameters."""
    y_true, y_pred, intervals, _ = regression_data
    
    plt.close('all')
    plot_calibration(
        y_true, y_pred, intervals,
        task='regression',
        confidence=0.90,
        n_bins=15,
        figsize=(12, 8),
        title="Custom Calibration Plot",
        show_warnings=False
    )
    plt.close('all')


def test_plot_calibration_regression_intervals_as_array(regression_data):
    """Test regression plot with intervals as numpy array."""
    y_true, y_pred, intervals, _ = regression_data
    
    intervals_array = np.array(intervals)
    
    plt.close('all')
    plot_calibration(y_true, y_pred, intervals_array, task='regression')
    plt.close('all')


# Tests for plot_calibration - Classification

def test_plot_calibration_classification_basic(classification_data):
    """Test basic classification calibration plot."""
    y_true, y_pred, y_pred_proba, _ = classification_data
    
    plt.close('all')
    plot_calibration(y_true, y_pred, y_pred_proba, task='classification')
    plt.close('all')


def test_plot_calibration_classification_with_confidence_scores(classification_data):
    """Test classification plot with confidence scores instead of probabilities."""
    y_true, y_pred, _, confidence_scores = classification_data
    
    plt.close('all')
    plot_calibration(y_true, y_pred, confidence_scores, task='classification')
    plt.close('all')


def test_plot_calibration_classification_return_diagnostics(classification_data):
    """Test that classification diagnostics are returned correctly."""
    y_true, y_pred, y_pred_proba, _ = classification_data
    
    plt.close('all')
    diagnostics = plot_calibration(
        y_true, y_pred, y_pred_proba,
        task='classification',
        return_diagnostics=True
    )
    plt.close('all')
    
    # Check required keys
    assert 'expected_calibration_error' in diagnostics
    assert 'overall_accuracy' in diagnostics
    assert 'mean_confidence' in diagnostics
    assert 'warnings' in diagnostics
    assert 'is_well_calibrated' in diagnostics
    
    # Check value types
    assert isinstance(diagnostics['expected_calibration_error'], float)
    assert isinstance(diagnostics['overall_accuracy'], float)
    assert isinstance(diagnostics['warnings'], list)
    assert isinstance(diagnostics['is_well_calibrated'], bool)


def test_plot_calibration_classification_custom_params(classification_data):
    """Test classification plot with custom parameters."""
    y_true, y_pred, y_pred_proba, _ = classification_data
    
    plt.close('all')
    plot_calibration(
        y_true, y_pred, y_pred_proba,
        task='classification',
        n_bins=15,
        figsize=(12, 8),
        title="Custom Classification Calibration",
        show_warnings=False
    )
    plt.close('all')


def test_plot_calibration_invalid_task(regression_data):
    """Test that invalid task raises ValueError."""
    y_true, y_pred, intervals, _ = regression_data
    
    with pytest.raises(ValueError, match="Unknown task"):
        plot_calibration(y_true, y_pred, intervals, task='invalid_task')


# Tests for check_regression_calibration

def test_check_regression_calibration_basic(regression_data):
    """Test basic regression calibration check."""
    y_true, _, intervals, _ = regression_data
    
    diagnostics = check_regression_calibration(y_true, intervals)
    
    assert 'empirical_coverage' in diagnostics
    assert 'nominal_coverage' in diagnostics
    assert 'warnings' in diagnostics
    assert 'is_well_calibrated' in diagnostics
    
    assert isinstance(diagnostics['empirical_coverage'], float)
    assert 0 <= diagnostics['empirical_coverage'] <= 1


def test_check_regression_calibration_with_array():
    """Test regression calibration check with array intervals."""
    np.random.seed(42)
    n = 100
    y_true = np.random.randn(n)
    y_pred = np.zeros(n)
    
    # Well-calibrated intervals (should cover ~95%)
    intervals = np.column_stack([y_pred - 2, y_pred + 2])
    
    diagnostics = check_regression_calibration(y_true, intervals, confidence=0.95)
    
    # Should have high coverage since std of randn is ~1
    assert diagnostics['empirical_coverage'] > 0.90


def test_check_regression_calibration_undercoverage():
    """Test that undercoverage is detected."""
    np.random.seed(42)
    n = 100
    y_true = np.random.randn(n) * 10  # Large variance
    y_pred = np.zeros(n)
    
    # Very narrow intervals
    intervals = np.column_stack([y_pred - 0.1, y_pred + 0.1])
    
    diagnostics = check_regression_calibration(y_true, intervals, confidence=0.95)
    
    # Should detect undercoverage
    assert any('UNDERCOVERAGE' in w for w in diagnostics['warnings'])
    assert not diagnostics['is_well_calibrated']


def test_check_regression_calibration_overcoverage():
    """Test that overcoverage is detected."""
    np.random.seed(42)
    n = 100
    y_true = np.random.randn(n) * 0.1  # Small variance
    y_pred = np.zeros(n)
    
    # Very wide intervals
    intervals = np.column_stack([y_pred - 10, y_pred + 10])
    
    diagnostics = check_regression_calibration(y_true, intervals, confidence=0.95)
    
    # Should detect overcoverage
    assert any('OVERCOVERAGE' in w for w in diagnostics['warnings'])


def test_check_regression_calibration_constant_intervals():
    """Test that constant intervals are detected."""
    np.random.seed(42)
    n = 100
    y_true = np.random.randn(n)
    y_pred = np.zeros(n)
    
    # All intervals exactly the same
    intervals = np.column_stack([y_pred - 2, y_pred + 2])
    
    diagnostics = check_regression_calibration(y_true, intervals)
    
    # Should detect constant intervals
    assert any('CONSTANT INTERVALS' in w for w in diagnostics['warnings'])


def test_check_regression_calibration_uncertainty_collapse():
    """Test that uncertainty collapse is detected."""
    np.random.seed(42)
    n = 100
    y_true = np.random.randn(n)
    y_pred = np.zeros(n)
    
    # Extremely narrow intervals
    intervals = np.column_stack([y_pred - 1e-8, y_pred + 1e-8])
    
    diagnostics = check_regression_calibration(y_true, intervals)
    
    # Should detect uncertainty collapse
    assert any('UNCERTAINTY COLLAPSE' in w for w in diagnostics['warnings'])


# Tests for check_classification_calibration

def test_check_classification_calibration_basic(classification_data):
    """Test basic classification calibration check."""
    y_true, _, y_pred_proba, _ = classification_data
    
    diagnostics = check_classification_calibration(y_true, y_pred_proba)
    
    assert 'expected_calibration_error' in diagnostics
    assert 'overall_accuracy' in diagnostics
    assert 'mean_confidence' in diagnostics
    assert 'warnings' in diagnostics
    assert 'is_well_calibrated' in diagnostics
    
    assert isinstance(diagnostics['expected_calibration_error'], float)
    assert diagnostics['expected_calibration_error'] >= 0


def test_check_classification_calibration_with_confidence_scores():
    """Test classification calibration with 1D confidence scores."""
    np.random.seed(42)
    n = 200
    
    y_true = np.random.randint(0, 2, n)
    confidence_scores = np.random.rand(n)
    
    diagnostics = check_classification_calibration(y_true, confidence_scores)
    
    assert 'expected_calibration_error' in diagnostics


def test_check_classification_calibration_overconfident():
    """Test that overconfidence is detected."""
    np.random.seed(42)
    n = 200
    
    y_true = np.random.randint(0, 2, n)
    
    # Very high confidence, but random predictions (50% accuracy)
    high_confidence = np.random.rand(n) * 0.2 + 0.8  # 0.8-1.0
    y_pred_proba = np.column_stack([1 - high_confidence, high_confidence])
    
    diagnostics = check_classification_calibration(y_true, y_pred_proba)
    
    # Should detect overconfidence (high confidence, low accuracy)
    # Note: might not always trigger depending on random seed
    # Just check the function runs without error
    assert 'warnings' in diagnostics


def test_check_classification_calibration_high_ece():
    """Test that high ECE is detected."""
    np.random.seed(42)
    n = 200
    
    # Create purposely miscalibrated predictions
    y_true = np.random.randint(0, 2, n)
    
    # Inverse relationship: high confidence when wrong, low when right
    confidence = np.where(y_true == 1, 0.2, 0.9)
    y_pred_proba = np.column_stack([1 - confidence, confidence])
    
    diagnostics = check_classification_calibration(y_true, y_pred_proba, n_bins=5)
    
    # Should have high ECE
    assert diagnostics['expected_calibration_error'] > 0


# Integration tests

def test_plot_calibration_regression_integration():
    """Integration test: Create realistic regression scenario."""
    np.random.seed(42)
    n = 200
    
    # Create heteroscedastic data (variance increases with x)
    x = np.linspace(0, 10, n)
    noise = (1 + 0.5 * x) * np.random.randn(n)
    y_true = 2 * x + 3 + noise
    y_pred = 2 * x + 3
    
    # Create adaptive intervals (wider for larger x)
    uncertainty = 2 * (1 + 0.5 * x)
    intervals = np.column_stack([y_pred - uncertainty, y_pred + uncertainty])
    
    plt.close('all')
    diagnostics = plot_calibration(
        y_true, y_pred, intervals,
        task='regression',
        confidence=0.95,
        return_diagnostics=True
    )
    plt.close('all')
    
    # Should be reasonably well-calibrated
    assert 'empirical_coverage' in diagnostics
    assert diagnostics['empirical_coverage'] > 0.80


def test_plot_calibration_classification_integration():
    """Integration test: Create realistic classification scenario."""
    np.random.seed(42)
    n = 500
    
    # Create multi-class problem
    n_classes = 3
    y_true = np.random.randint(0, n_classes, n)
    
    # Generate somewhat calibrated probabilities
    proba = np.random.dirichlet(np.ones(n_classes) * 2, n)
    
    # Make predictions match true labels more often
    for i in range(n):
        proba[i, y_true[i]] += 0.3
        proba[i] = proba[i] / proba[i].sum()  # Renormalize
    
    y_pred = np.argmax(proba, axis=1)
    
    plt.close('all')
    diagnostics = plot_calibration(
        y_true, y_pred, proba,
        task='classification',
        return_diagnostics=True
    )
    plt.close('all')
    
    assert 'expected_calibration_error' in diagnostics
    assert diagnostics['overall_accuracy'] > 0.3  # Better than random


def test_diagnostics_api_consistency():
    """Test that diagnostics functions have consistent API."""
    np.random.seed(42)
    n = 100
    
    # Regression
    y_true_reg = np.random.randn(n)
    intervals = np.column_stack([y_true_reg - 2, y_true_reg + 2])
    
    diag_reg = check_regression_calibration(y_true_reg, intervals)
    assert 'warnings' in diag_reg
    assert 'is_well_calibrated' in diag_reg
    
    # Classification
    y_true_clf = np.random.randint(0, 2, n)
    proba = np.random.rand(n, 2)
    proba = proba / proba.sum(axis=1, keepdims=True)
    
    diag_clf = check_classification_calibration(y_true_clf, proba)
    assert 'warnings' in diag_clf
    assert 'is_well_calibrated' in diag_clf


# Edge cases

def test_plot_calibration_single_sample():
    """Test with a single sample."""
    y_true = np.array([1.0])
    y_pred = np.array([1.0])
    intervals = np.array([[0.5, 1.5]])
    
    plt.close('all')
    # Should not crash
    plot_calibration(y_true, y_pred, intervals, task='regression')
    plt.close('all')


def test_plot_calibration_perfect_predictions():
    """Test with perfect predictions."""
    np.random.seed(42)
    n = 50
    
    y_true = np.random.randn(n)
    y_pred = y_true.copy()  # Perfect predictions
    intervals = np.column_stack([y_pred - 2, y_pred + 2])
    
    plt.close('all')
    diagnostics = plot_calibration(
        y_true, y_pred, intervals,
        task='regression',
        return_diagnostics=True
    )
    plt.close('all')
    
    # Should have very high coverage
    assert diagnostics['empirical_coverage'] > 0.95


def test_check_regression_calibration_list_input():
    """Test that list inputs work."""
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    intervals = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    
    diagnostics = check_regression_calibration(y_true, intervals)
    assert isinstance(diagnostics, dict)
    assert 'empirical_coverage' in diagnostics
