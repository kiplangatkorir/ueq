import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from ueq.utils.visualization import (
    plot_intervals,
    plot_reliability_diagram,
    plot_coverage_vs_confidence,
    plot_predictions_with_intervals,
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing visualizations."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y_true = 2 * x + 1 + np.random.randn(50) * 2
    y_pred = 2 * x + 1
    lower = y_pred - 2
    upper = y_pred + 2
    intervals = list(zip(lower, upper))
    
    return x, y_true, y_pred, lower, upper, intervals


def test_plot_intervals_basic(sample_data):
    """Test basic plot_intervals functionality."""
    x, y_true, y_pred, lower, upper, intervals = sample_data
    
    # Should not raise any errors
    plt.close('all')
    plot_intervals(x, y_pred, lower, upper)
    plt.close('all')


def test_plot_intervals_with_true_values(sample_data):
    """Test plot_intervals with true values."""
    x, y_true, y_pred, lower, upper, intervals = sample_data
    
    plt.close('all')
    plot_intervals(x, y_pred, lower, upper, y_true=y_true)
    plt.close('all')


def test_plot_intervals_custom_params(sample_data):
    """Test plot_intervals with custom parameters."""
    x, y_true, y_pred, lower, upper, intervals = sample_data
    
    plt.close('all')
    plot_intervals(
        x, y_pred, lower, upper, y_true=y_true,
        title="Custom Title",
        xlabel="Custom X",
        ylabel="Custom Y",
        figsize=(12, 8),
        alpha=0.5,
        show_points=False
    )
    plt.close('all')


def test_plot_reliability_diagram(sample_data):
    """Test reliability diagram plotting."""
    x, y_true, y_pred, lower, upper, intervals = sample_data
    
    plt.close('all')
    plot_reliability_diagram(y_true, y_pred, intervals)
    plt.close('all')


def test_plot_reliability_diagram_custom_params(sample_data):
    """Test reliability diagram with custom parameters."""
    x, y_true, y_pred, lower, upper, intervals = sample_data
    
    plt.close('all')
    plot_reliability_diagram(
        y_true, y_pred, intervals,
        n_bins=15,
        confidence=0.90,
        title="Custom Reliability",
        figsize=(10, 10)
    )
    plt.close('all')


def test_plot_coverage_vs_confidence(sample_data):
    """Test coverage vs confidence plotting."""
    x, y_true, y_pred, lower, upper, intervals = sample_data
    
    plt.close('all')
    plot_coverage_vs_confidence(y_true, intervals)
    plt.close('all')


def test_plot_coverage_vs_confidence_custom_bins(sample_data):
    """Test coverage vs confidence with custom bins."""
    x, y_true, y_pred, lower, upper, intervals = sample_data
    
    plt.close('all')
    plot_coverage_vs_confidence(
        y_true, intervals,
        n_bins=30,
        title="Custom Coverage Plot"
    )
    plt.close('all')


def test_plot_predictions_with_intervals(sample_data):
    """Test legacy predictions plotting function."""
    x, y_true, y_pred, lower, upper, intervals = sample_data
    
    # This function expects X to be 2D or specific format
    X = x.reshape(-1, 1)
    
    plt.close('all')
    plot_predictions_with_intervals(X, y_true, y_pred, intervals)
    plt.close('all')


def test_plots_handle_array_conversion():
    """Test that plots handle different input types (lists, arrays)."""
    x = [1, 2, 3, 4, 5]
    y_pred = [2, 4, 6, 8, 10]
    lower = [1, 3, 5, 7, 9]
    upper = [3, 5, 7, 9, 11]
    y_true = [2.1, 3.9, 6.2, 7.8, 10.1]
    
    plt.close('all')
    plot_intervals(x, y_pred, lower, upper, y_true=y_true)
    plt.close('all')


def test_reliability_diagram_with_list_intervals():
    """Test reliability diagram accepts list of tuples."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5)]
    
    plt.close('all')
    plot_reliability_diagram(y_true, y_pred, intervals)
    plt.close('all')


def test_plots_do_not_crash_with_edge_cases():
    """Test that plots handle edge cases gracefully."""
    # Small dataset
    x = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    lower = np.array([0, 1, 2])
    upper = np.array([2, 3, 4])
    y_true = np.array([1.5, 2.5, 3.5])
    intervals = list(zip(lower, upper))
    
    plt.close('all')
    
    # All plots should handle small datasets
    plot_intervals(x, y_pred, lower, upper, y_true=y_true)
    plot_reliability_diagram(y_true, y_pred, intervals, n_bins=2)
    plot_coverage_vs_confidence(y_true, intervals, n_bins=2)
    
    plt.close('all')
