import numpy as np


class ConformalUQ:
    """
    Conformal Prediction for Uncertainty Quantification.

    Provides distribution-free prediction intervals with guaranteed coverage.

    Parameters
    ----------
    model : object
        Any scikit-learn compatible estimator.
    alpha : float, default=0.05
        Significance level (e.g., 0.05 = 95% confidence intervals).
    """

    def __init__(self, model, alpha=0.05):
        self.base_model = model
        self.alpha = alpha
        self.q = None
        self.is_fitted = False

    def fit(self, X_train, y_train, X_calib, y_calib):
        """
        Fit the model on training data and calibrate on calibration data.

        Parameters
        ----------
        X_train, y_train : array-like
            Training data.
        X_calib, y_calib : array-like
            Calibration data.
        """
        self.base_model.fit(X_train, y_train)

        # Predictions on calibration set
        preds = self.base_model.predict(X_calib)

        # Nonconformity scores
        scores = np.abs(y_calib - preds)

        # Quantile for intervals
        n = len(scores)
        k = int(np.ceil((1 - self.alpha) * (n + 1)))  # conformal quantile
        self.q = np.sort(scores)[min(k, n) - 1]

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Predict with conformal prediction intervals.

        Parameters
        ----------
        X : array-like
            Test inputs.

        Returns
        -------
        preds : np.ndarray
            Point predictions.
        intervals : list of tuples
            (lower, upper) prediction intervals.
        """
        if not self.is_fitted:
            raise RuntimeError("ConformalUQ model is not fitted yet.")

        preds = self.base_model.predict(X)
        intervals = [(p - self.q, p + self.q) for p in preds]

        return preds, intervals
