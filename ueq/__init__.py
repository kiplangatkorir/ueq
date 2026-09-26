from .core import UQ
from ._warnings import ExperimentalWarning
from .utils.metrics import (
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

# Alias for more intuitive API as specified in Issue #9
evaluate = evaluate_uncertainty
from .utils.visualization import (
    plot_intervals,
    plot_reliability_diagram,
    plot_coverage_vs_confidence,
    plot_predictions_with_intervals,
    plot_calibration_curve,
    plot_uncertainty_timeline
)
from .utils.monitoring import UQMonitor, PerformanceMonitor, detect_uncertainty_drift
from .utils.performance import BatchProcessor, PerformanceProfiler, optimize_batch_size, memory_efficient_predict
from .utils.recalibration import DriftAwareRecalibrator, UncertaintyInflator
from .benchmarks import (
    make_synthetic_regression,
    make_heteroscedastic_data,
    make_concept_drift_data,
    make_covariate_shift_data
)
from .diagnostics import (
    plot_calibration,
    check_regression_calibration,
    check_classification_calibration
)

__version__ = "1.0.2"
__all__ = [
    "UQ",
    "ExperimentalWarning",
    "coverage",
    "sharpness",
    "interval_width",
    "expected_calibration_error",
    "maximum_calibration_error",
    "interval_score",
    "miscoverage_rate",
    "evaluate_uncertainty",
    "evaluate",  # Alias for evaluate_uncertainty (Issue #9)
    "check_calibration",
    "plot_intervals",
    "plot_reliability_diagram",
    "plot_coverage_vs_confidence",
    "plot_predictions_with_intervals",
    "plot_calibration_curve",
    "plot_uncertainty_timeline",
    "UQMonitor",
    "PerformanceMonitor",
    "detect_uncertainty_drift",
    "BatchProcessor",
    "PerformanceProfiler",
    "optimize_batch_size",
    "memory_efficient_predict",
    "DriftAwareRecalibrator",
    "UncertaintyInflator",
    "make_synthetic_regression",
    "make_heteroscedastic_data",
    "make_concept_drift_data",
    "make_covariate_shift_data",
    "plot_calibration",
    "check_regression_calibration",
    "check_classification_calibration"
]