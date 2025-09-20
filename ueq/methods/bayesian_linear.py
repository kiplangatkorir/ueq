import numpy as np
from typing import Tuple, List, Optional
from scipy.stats import norm

class BayesianLinearUQ:
    """Bayesian Linear Regression with uncertainty quantification."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize Bayesian Linear Regression.
        
        Args:
            alpha: Precision of the prior distribution
            beta: Precision of the noise distribution
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearUQ":
        """Fit Bayesian Linear Regression."""
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Posterior distribution parameters
        S_N_inv = self.alpha * np.eye(X.shape[1]) + self.beta * X.T @ X
        S_N = np.linalg.inv(S_N_inv)
        m_N = self.beta * S_N @ X.T @ y
        
        self.mean = m_N
        self.cov = S_N
        self.is_fitted = True
        return self

    def predict(
        self, 
        X: np.ndarray, 
        return_interval: bool = True,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Predict with uncertainty intervals."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
            
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Mean prediction
        mean_pred = X @ self.mean
        
        if return_interval:
            # Predictive variance
            var = 1/self.beta + np.sum(X @ self.cov * X, axis=1)
            std = np.sqrt(var)
            
            # Confidence intervals using normal distribution
            z = norm.ppf(1 - alpha/2)  # Changed from multivariate_normal to norm
            intervals = list(zip(
                mean_pred - z * std,
                mean_pred + z * std
            ))
            return mean_pred, intervals
            
        return mean_pred

    def predict_dist(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Sample from the predictive distribution."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
            
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Sample from parameter posterior
        w_samples = np.random.multivariate_normal(
            self.mean.ravel(), 
            self.cov, 
            size=n_samples
        )
        
        # Get predictions for each sampled parameter
        predictions = X @ w_samples.T
        return predictions
