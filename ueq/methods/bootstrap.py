import numpy as np
from sklearn.base import clone


class BootstrapUQ:
    """
    Bootstrap-based Uncertainty Quantification.

    Trains multiple models on bootstrap resamples of the training data.
    Predictions are aggregated to estimate mean and confidence intervals.

    Parameters
    ----------
    model : object
        Any scikit-learn compatible estimator.
    n_models : int, default=100
        Number of bootstrap models to train.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, model, n_models=100, random_state=None):
        self.base_model = model
        self.n_models = n_models
        self.random_state = random_state
        self.models = []
        self.is_fitted = False

    def fit(self, X, y):
        """Fit bootstrap models on resampled datasets."""
        rng = np.random.default_rng(self.random_state)
        n = len(X)

        self.models = []
        for _ in range(self.n_models):
            idx = rng.choice(n, size=n, replace=True)
            m = clone(self.base_model)
            m.fit(X[idx], y[idx])
            self.models.append(m)

        self.is_fitted = True
        return self

    def predict(self, X, return_interval=True, alpha=0.05):
        """
        Predict with uncertainty intervals.

        Parameters
        ----------
        X : array-like
            Test inputs.
        return_interval : bool, default=True
            Whether to return prediction intervals.
        alpha : float, default=0.05
            Significance level for confidence intervals (e.g., 0.05 = 95% CI).

        Returns
        -------
        mean_pred : np.ndarray
            Mean predictions.
        intervals : list of tuples
            (lower, upper) confidence intervals for each prediction.
            Returned only if return_interval=True.
        """
        if not self.is_fitted:
            raise RuntimeError("BootstrapUQ model is not fitted yet.")

        preds = np.array([m.predict(X) for m in self.models])
        mean_pred = preds.mean(axis=0)

        if return_interval:
            lower = np.percentile(preds, 100 * alpha / 2, axis=0)
            upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
            return mean_pred, list(zip(lower, upper))

        return mean_pred

    def predict_dist(self, X):
        """
        Return the full predictive distribution (all bootstrap predictions).
        """
        if not self.is_fitted:
            raise RuntimeError("BootstrapUQ model is not fitted yet.")

        preds = np.array([m.predict(X) for m in self.models])
        return preds
