"""Utility functions for uncertainty quantification."""

from .api import evaluate
from .plotting import plot_intervals as plot_intervals_old
from .visualization import (
    plot_intervals,
    plot_reliability_diagram, 
    plot_coverage_vs_confidence,
    plot_predictions_with_intervals,
    plot_calibration_curve
)
from .metrics import (
    coverage,
    sharpness, 
    interval_width,
    expected_calibration_error,
    maximum_calibration_error,
    interval_score,
    miscoverage_rate,
    evaluate_uncertainty,
    check_calibration
)
from .monitoring import UQMonitor, PerformanceMonitor, detect_uncertainty_drift
from .performance import BatchProcessor, PerformanceProfiler, optimize_batch_size, memory_efficient_predict

__all__ = [
    "evaluate", 
    "plot_intervals",
    "plot_intervals_old",
    "plot_reliability_diagram",
    "plot_coverage_vs_confidence",
    "plot_predictions_with_intervals",
    "plot_calibration_curve",
    "coverage",
    "sharpness",
    "interval_width",
    "expected_calibration_error",
    "maximum_calibration_error",
    "interval_score",
    "miscoverage_rate",
    "evaluate_uncertainty",
    "check_calibration",
    "UQMonitor",
    "PerformanceMonitor", 
    "detect_uncertainty_drift",
    "BatchProcessor",
    "PerformanceProfiler",
    "optimize_batch_size",
    "memory_efficient_predict"
]