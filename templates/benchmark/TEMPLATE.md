# Contributing a New Benchmark Dataset

This template helps you add benchmark datasets for evaluating UQ methods.

## Quick Start

```python
def make_your_benchmark(n_samples=1000, seed=None):
    """
    Generate benchmark with known uncertainty.
    
    Returns
    -------
    X : ndarray
        Features
    y : ndarray  
        Noisy targets
    meta : dict
        Must include 'y_true' and 'noise_std'
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features)
    y_true = true_function(X)
    noise_std = compute_noise_std(X)
    y = y_true + np.random.randn(n_samples) * noise_std
    
    return X, y, {'y_true': y_true, 'noise_std': noise_std}
```

See full template for complete guidelines.
