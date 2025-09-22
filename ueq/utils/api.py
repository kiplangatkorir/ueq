import numpy as np
from ..core import UQ
from .metrics import coverage, sharpness, expected_calibration_error, maximum_calibration_error

def evaluate(
    model=None, 
    method="bayesian_linear", 
    X_train=None, 
    y_train=None, 
    X_test=None, 
    y_test=None, 
    metrics=("coverage", "sharpness", "ece", "mce"), 
    **kwargs
):
    """Train + evaluate a UQ method on given dataset."""
    # Initialize and fit
    uq = UQ(model, method=method, **kwargs)
    uq.fit(X_train, y_train)

    # Get predictions
    mean_pred, intervals = uq.predict(X_test, return_interval=True)

    results = {
        "mean": mean_pred,
        "intervals": intervals,
        "metrics": {}
    }

    # Compute requested metrics
    if y_test is not None:
        metric_funcs = {
            "coverage": lambda y, i: coverage(y, i),
            "sharpness": lambda y, i: sharpness(i),  # Fix: only pass intervals
            "ece": lambda y, i: expected_calibration_error(y, i),
            "mce": lambda y, i: maximum_calibration_error(y, i)
        }
        
        for metric in metrics:
            if metric in metric_funcs:
                results["metrics"][metric] = metric_funcs[metric](y_test, intervals)

    return results