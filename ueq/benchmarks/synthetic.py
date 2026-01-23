"""
Synthetic benchmark datasets with known uncertainty structure.

These datasets enable controlled evaluation of UQ methods with ground-truth
uncertainty and known noise distributions.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal


def make_synthetic_regression(
    n_samples: int = 1000,
    n_features: int = 10,
    noise: Literal["homoscedastic", "heteroscedastic"] = "homoscedastic",
    noise_level: float = 1.0,
    shift: Optional[Literal["covariate", "label_noise", "concept_drift"]] = None,
    shift_strength: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic regression data with controllable uncertainty.
    
    Creates regression datasets with known noise distributions and optional
    distribution shifts for benchmarking UQ methods.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate.
    n_features : int, default=10
        Number of input features.
    noise : {"homoscedastic", "heteroscedastic"}, default="homoscedastic"
        Type of noise:
        - "homoscedastic": Constant variance across inputs
        - "heteroscedastic": Input-dependent variance
    noise_level : float, default=1.0
        Base noise level (standard deviation).
    shift : {None, "covariate", "label_noise", "concept_drift"}, optional
        Type of distribution shift to introduce:
        - None: No shift
        - "covariate": Shift in input distribution
        - "label_noise": Increased label noise in second half
        - "concept_drift": Gradual change in true function
    shift_strength : float, default=0.5
        Strength of distribution shift (if applicable).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Input features.
    y : ndarray, shape (n_samples,)
        Target values (with noise).
    meta : dict
        Metadata including:
        - 'y_true': Noise-free targets
        - 'noise_std': True noise standard deviation per sample
        - 'shift_point': Index where shift occurs (if applicable)
    
    Examples
    --------
    >>> X, y, meta = make_synthetic_regression(
    ...     n_samples=500,
    ...     noise="heteroscedastic",
    ...     shift="concept_drift"
    ... )
    >>> print(f"True variance range: {meta['noise_std'].min():.2f} to {meta['noise_std'].max():.2f}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate input features
    X = np.random.randn(n_samples, n_features)
    
    # True function (linear with some nonlinearity)
    weights = np.random.randn(n_features)
    y_true = X @ weights + 0.5 * np.sin(X[:, 0]) * X[:, 1]
    
    # Apply concept drift if requested
    shift_point = None
    if shift == "concept_drift":
        shift_point = n_samples // 2
        drift_effect = shift_strength * np.linspace(0, 1, n_samples)
        y_true += drift_effect * X[:, 0]
    
    # Generate noise
    if noise == "homoscedastic":
        noise_std = np.ones(n_samples) * noise_level
    elif noise == "heteroscedastic":
        # Noise depends on first feature (absolute value to ensure positive)
        noise_std = noise_level * (1.0 + np.abs(X[:, 0]))
    else:
        raise ValueError(f"Unknown noise type: {noise}")
    
    # Apply label noise shift if requested
    if shift == "label_noise":
        shift_point = n_samples // 2
        noise_std[shift_point:] *= (1.0 + shift_strength)
    
    # Apply covariate shift if requested
    if shift == "covariate":
        shift_point = n_samples // 2
        # Shift the distribution of the first feature
        X[shift_point:, 0] += shift_strength
    
    # Generate noisy targets
    noise = np.random.randn(n_samples) * noise_std
    y = y_true + noise
    
    meta = {
        'y_true': y_true,
        'noise_std': noise_std,
        'shift_point': shift_point,
        'noise_type': noise,
        'shift_type': shift
    }
    
    return X, y, meta


def make_heteroscedastic_data(
    n_samples: int = 1000,
    n_features: int = 5,
    variance_function: Literal["linear", "quadratic", "exponential"] = "linear",
    min_variance: float = 0.1,
    max_variance: float = 5.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate regression data with heteroscedastic (input-dependent) noise.
    
    Useful for testing UQ methods' ability to capture varying uncertainty
    across the input space.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples.
    n_features : int, default=5
        Number of features.
    variance_function : {"linear", "quadratic", "exponential"}, default="linear"
        How variance changes with inputs:
        - "linear": Variance scales linearly with first feature
        - "quadratic": Quadratic dependence
        - "exponential": Exponential dependence (strong heteroscedasticity)
    min_variance : float, default=0.1
        Minimum noise variance.
    max_variance : float, default=5.0
        Maximum noise variance.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Input features.
    y : ndarray, shape (n_samples,)
        Noisy targets.
    meta : dict
        Metadata with 'y_true' and 'noise_std'.
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_features)
    y_true = X @ weights
    
    # Normalize first feature to [0, 1] for variance computation
    x_norm = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min() + 1e-8)
    
    if variance_function == "linear":
        variance = min_variance + (max_variance - min_variance) * x_norm
    elif variance_function == "quadratic":
        variance = min_variance + (max_variance - min_variance) * (x_norm ** 2)
    elif variance_function == "exponential":
        variance = min_variance * np.exp(np.log(max_variance / min_variance) * x_norm)
    else:
        raise ValueError(f"Unknown variance_function: {variance_function}")
    
    noise_std = np.sqrt(variance)
    noise = np.random.randn(n_samples) * noise_std
    y = y_true + noise
    
    meta = {
        'y_true': y_true,
        'noise_std': noise_std,
        'variance_function': variance_function
    }
    
    return X, y, meta


def make_concept_drift_data(
    n_samples: int = 1000,
    n_features: int = 5,
    drift_type: Literal["gradual", "abrupt", "periodic"] = "gradual",
    drift_strength: float = 2.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate time-indexed data with concept drift.
    
    Simulates evolving relationships between features and target,
    useful for testing online/adaptive UQ methods.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of time-ordered samples.
    n_features : int, default=5
        Number of features.
    drift_type : {"gradual", "abrupt", "periodic"}, default="gradual"
        How the concept changes:
        - "gradual": Linear drift in weights over time
        - "abrupt": Sudden change at midpoint
        - "periodic": Oscillating concept
    drift_strength : float, default=2.0
        Magnitude of drift effect.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Input features (stationary).
    y : ndarray, shape (n_samples,)
        Targets (with drift in relationship).
    meta : dict
        Metadata with drift information.
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features)
    
    # Initial weights
    w_init = np.random.randn(n_features)
    
    # Time index
    t = np.linspace(0, 1, n_samples)
    
    if drift_type == "gradual":
        # Linearly evolving weights
        w_drift = np.random.randn(n_features) * drift_strength
        y_true = np.array([X[i] @ (w_init + t[i] * w_drift) for i in range(n_samples)])
    
    elif drift_type == "abrupt":
        # Sudden change at midpoint
        y_true = np.zeros(n_samples)
        mid = n_samples // 2
        w_new = w_init + np.random.randn(n_features) * drift_strength
        y_true[:mid] = X[:mid] @ w_init
        y_true[mid:] = X[mid:] @ w_new
    
    elif drift_type == "periodic":
        # Oscillating concept
        period = 4  # Number of periods over the dataset
        drift_signal = np.sin(2 * np.pi * period * t) * drift_strength
        w_drift = np.random.randn(n_features)
        y_true = np.array([X[i] @ (w_init + drift_signal[i] * w_drift) for i in range(n_samples)])
    
    else:
        raise ValueError(f"Unknown drift_type: {drift_type}")
    
    # Add noise
    noise = np.random.randn(n_samples) * 0.5
    y = y_true + noise
    
    meta = {
        'y_true': y_true,
        'drift_type': drift_type,
        'drift_strength': drift_strength,
        'time_index': t
    }
    
    return X, y, meta


def make_covariate_shift_data(
    n_train: int = 500,
    n_test: int = 200,
    n_features: int = 5,
    shift_strength: float = 2.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate train and test sets with covariate shift.
    
    Test distribution differs from training distribution, but the
    relationship Y|X remains the same. Tests OoD uncertainty behavior.
    
    Parameters
    ----------
    n_train : int, default=500
        Number of training samples.
    n_test : int, default=200
        Number of test samples.
    n_features : int, default=5
        Number of features.
    shift_strength : float, default=2.0
        Magnitude of distribution shift.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    X_train : ndarray, shape (n_train, n_features)
        Training features.
    y_train : ndarray, shape (n_train,)
        Training targets.
    X_test : ndarray, shape (n_test, n_features)
        Test features (shifted distribution).
    y_test : ndarray, shape (n_test,)
        Test targets.
    meta : dict
        Metadata about the shift.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Training data from N(0, 1)
    X_train = np.random.randn(n_train, n_features)
    
    # Test data with shifted mean
    X_test = np.random.randn(n_test, n_features)
    X_test[:, 0] += shift_strength  # Shift first feature
    
    # Same underlying function
    weights = np.random.randn(n_features)
    y_train_true = X_train @ weights
    y_test_true = X_test @ weights
    
    # Same noise level
    noise_std = 0.5
    y_train = y_train_true + np.random.randn(n_train) * noise_std
    y_test = y_test_true + np.random.randn(n_test) * noise_std
    
    meta = {
        'shift_strength': shift_strength,
        'y_train_true': y_train_true,
        'y_test_true': y_test_true,
        'shifted_feature': 0
    }
    
    return X_train, y_train, X_test, y_test, meta
