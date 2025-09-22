import numpy as np

class ConformalUQ:
    """
    Conformal Prediction for Uncertainty Quantification.

    Supports both regression (intervals) and classification (prediction sets).

    Parameters
    ----------
    model : object
        Any scikit-learn compatible estimator.
    alpha : float, default=0.05
        Significance level (e.g., 0.05 = 95% confidence level).
    task_type : str, default="regression"
        Either "regression" or "classification".
    """

    def __init__(self, model, alpha=0.05, task_type="regression"):
        self.base_model = model
        self.alpha = alpha
        self.task_type = task_type
        self.q = None
        self.is_fitted = False

    def fit(self, X_train, y_train, X_calib, y_calib):
        """
        Fit model on training data and calibrate using calibration set.
        """
        self.base_model.fit(X_train, y_train)

        if self.task_type == "regression":
            preds = self.base_model.predict(X_calib)
            scores = np.abs(y_calib - preds)  # residuals
            n = len(scores)
            k = int(np.ceil((1 - self.alpha) * (n + 1)))
            self.q = np.sort(scores)[min(k, n) - 1]

        elif self.task_type == "classification":
            probas = self.base_model.predict_proba(X_calib)
            true_class_probs = probas[np.arange(len(y_calib)), y_calib]
            scores = 1 - true_class_probs
            n = len(scores)
            k = int(np.ceil((1 - self.alpha) * (n + 1)))
            self.q = np.sort(scores)[min(k, n) - 1]

        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        self.is_fitted = True
        return self

    def predict(self, X, return_interval=False):
        """
        Predict with conformal intervals (regression) or prediction sets (classification).
        """
        if not self.is_fitted:
            raise RuntimeError("ConformalUQ model is not fitted yet.")

        if self.task_type == "regression":
            preds = self.base_model.predict(X)
            intervals = [(p - self.q, p + self.q) for p in preds]
            return (preds, intervals) if return_interval else preds

        elif self.task_type == "classification":
            probas = self.base_model.predict_proba(X)
            pred_sets = [set(np.where(p >= 1 - self.q)[0]) for p in probas]

            if return_interval:
                labels = [list(s)[0] if len(s) == 1 else -1 for s in pred_sets]
                return labels, pred_sets
            else:
                return pred_sets
