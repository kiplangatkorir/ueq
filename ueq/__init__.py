from .core import UQ
from .utils.metrics import (
    coverage,
    sharpness,
    expected_calibration_error,
    maximum_calibration_error
)

__version__ = "1.0.1"
__all__ = [
    "UQ",
    "coverage",
    "sharpness",
    "expected_calibration_error",
    "maximum_calibration_error"
]