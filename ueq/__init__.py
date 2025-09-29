from .core import UQ
from .utils.metrics import (
    coverage,
    sharpness,
    expected_calibration_error,
    maximum_calibration_error
)
from .utils.monitoring import UQMonitor, PerformanceMonitor, detect_uncertainty_drift
from .utils.performance import BatchProcessor, PerformanceProfiler, optimize_batch_size, memory_efficient_predict

__version__ = "1.0.1"
__all__ = [
    "UQ",
    "coverage",
    "sharpness",
    "expected_calibration_error",
    "maximum_calibration_error",
    "UQMonitor",
    "PerformanceMonitor",
    "detect_uncertainty_drift",
    "BatchProcessor",
    "PerformanceProfiler",
    "optimize_batch_size",
    "memory_efficient_predict"
]