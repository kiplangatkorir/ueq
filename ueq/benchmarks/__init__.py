"""
Benchmark datasets and generators for UQ evaluation.
"""

from .synthetic import (
    make_synthetic_regression,
    make_heteroscedastic_data,
    make_concept_drift_data,
    make_covariate_shift_data
)

__all__ = [
    'make_synthetic_regression',
    'make_heteroscedastic_data',
    'make_concept_drift_data',
    'make_covariate_shift_data'
]
