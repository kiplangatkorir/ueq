# Contributing a New UQ Metric

This template helps you add new evaluation metrics for UQ.

## Quick Start

```python
def your_metric(y_true, intervals):
    """
    Compute your metric.
    
    Parameters
    ----------
    y_true : array-like
        True values
    intervals : list of tuples
        Prediction intervals
        
    Returns
    -------
    score : float
        Metric value (explain if higher/lower is better)
    """
    y_true = np.asarray(y_true)
    lower = np.array([iv[0] for iv in intervals])
    upper = np.array([iv[1] for iv in intervals])
    
    # Compute metric
    scores = compute_scores(y_true, lower, upper)
    return np.mean(scores)
```

See full template for complete documentation guidelines.
