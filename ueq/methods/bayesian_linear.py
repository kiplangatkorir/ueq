import numpy as np


class BayesianLinearUQ:
    """
    Bayesian Linear Regression for Uncertainty Quantification.
    
    Uses a Gaussian prior on weights and conjugate Gaussian likelihood, 
    giving a closed-form posterior and predictive distribution.

    Model:
        y = Xw + ε
        w ~ N(0, α⁻¹ I)
        ε ~ N(0, β⁻¹)

    Parameters
    ----------
    alpha : float, default=1.0
        Precision of the prior (1 / variance).
    beta : float, default=1.0
        Precision of the noise (1 / variance).
    """

    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.m_N = None   # posterior mean
        self.S_N = None   # posterior covariance
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit Bayesian linear regression.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        n, d = X.shape

        # Prior precision matrix
        A = self.alpha * np.eye(d) + self.beta * (X.T @ X)

        # Posterior covariance
        self.S_N = np.linalg.inv(A)

        # Posterior mean
        self.m_N = self.beta * self.S_N @ X.T @ y

        self.is_fitted = True
        return self

    def predict(self, X, return_interval=True, alpha=0.05):
        """
        Predict with posterior predictive uncertainty.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        return_interval : bool, default=True
            Whether to return prediction intervals.
        alpha : float, default=0.05
            Confidence level (0.05 = 95% CI).

        Returns
        -------
        mean : ndarray
            Predictive mean.
        intervals : list of tuples
            Confidence intervals for each prediction (if return_interval=True).
        """
        if not self.is_fitted:
            raise RuntimeError("BayesianLinearUQ is not fitted yet.")

        X = np.asarray(X)
        mean = X @ self.m_N  # (n_samples, 1)
        mean = mean.squeeze()

        # Predictive variance = (1/beta) + X S_N X^T
        var = (1 / self.beta) + np.sum(X @ self.S_N * X, axis=1)
        std = np.sqrt(var)

        if return_interval:
            z = 1.96 if alpha == 0.05 else abs(np.percentile(np.random.randn(100000), 100 * (1 - alpha/2)))
            lower = mean - z * std
            upper = mean + z * std
            return mean, list(zip(lower, upper))

        return mean

    def predict_dist(self, X, n_samples=100):
        """
        Sample from the posterior predictive distribution.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        n_samples : int, default=100
            Number of posterior predictive samples.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_points)
            Predictive draws.
        """
        if not self.is_fitted:
            raise RuntimeError("BayesianLinearUQ is not fitted yet.")

        X = np.asarray(X)
        mean = X @ self.m_N
        cov = (1 / self.beta) * np.eye(X.shape[0]) + X @ self.S_N @ X.T

        # Draw predictive samples
        return np.random.multivariate_normal(mean.squeeze(), cov, size=n_samples)
