from .core import UQ
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
from .benchmarks import (
    make_synthetic_regression,
    make_heteroscedastic_data,
    make_concept_drift_data,
    make_covariate_shift_data
)

__version__ = "1.0.2"
__all__ = [
    "UQ",
    "coverage",
    "sharpness",
    "interval_width",
    "expected_calibration_error",
    "maximum_calibration_error",
    "interval_score",
    "miscoverage_rate",
    "evaluate_uncertainty",
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
    "make_synthetic_regression",
    "make_heteroscedastic_data",
    "make_concept_drift_data",
    "make_covariate_shift_data"
]