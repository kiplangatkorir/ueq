import numpy as np

from ._conformal_utils import (
    encode_labels,
    lower_quantile,
    model_classes,
    point_labels,
    prediction_sets,
    upper_quantile,
)

_REGRESSION_SCORES = ("residual", "quantile")
_CLASSIFICATION_SCORES = ("inverse_probability",)
_DISABLED_SCORES = ("normalized", "margin")


class ConformalUQ:
    """
    Split conformal prediction for uncertainty quantification.

    Supports both regression (intervals) and classification (prediction sets).
    Coverage is guaranteed marginally when calibration and test data are
    exchangeable.

    Parameters
    ----------
    model : object
        Any scikit-learn compatible estimator. Classification requires
        ``predict_proba``.
    alpha : float, default=0.05
        Miscoverage level (e.g., 0.05 = 95% coverage).
    task_type : str, default="regression"
        Either "regression" or "classification".
    nonconformity : str, optional
        Nonconformity score. For regression: "residual" (default, symmetric
        intervals) or "quantile" (asymmetric intervals from signed residuals).
        For classification: "inverse_probability" (default, LAC sets).
    class_conditional : bool, default=False
        Not supported in 1.0.2; use MAPIE or crepes for class-conditional
        (Mondrian) prediction sets.
    """

    def __init__(self, model, alpha=0.05, task_type="regression", nonconformity=None,
                 class_conditional=False):
        self.base_model = model
        self.alpha = alpha
        self.task_type = task_type
        if nonconformity is None:
            nonconformity = "inverse_probability" if task_type == "classification" else "residual"
        self.nonconformity = nonconformity
        self.class_conditional = class_conditional
        self.q = None
        self.q_lower = None  # For quantile-based methods
        self.classes_ = None
        self.is_fitted = False
        self._validate_options()

    def _validate_options(self):
        """Validate task type and nonconformity score selection."""
        if self.nonconformity in _DISABLED_SCORES:
            raise NotImplementedError(
                f"nonconformity='{self.nonconformity}' is disabled: it did not "
                "give valid coverage. For normalized or conditional scores, use "
                "MAPIE or crepes."
            )
        if self.class_conditional:
            raise NotImplementedError(
                "class_conditional=True is disabled: it did not give valid "
                "per-class coverage. Use MAPIE or crepes for class-conditional "
                "(Mondrian) prediction sets."
            )
        if self.task_type == "regression":
            if self.nonconformity not in _REGRESSION_SCORES:
                raise ValueError(f"For regression, nonconformity must be one of {list(_REGRESSION_SCORES)}")
        elif self.task_type == "classification":
            if self.nonconformity not in _CLASSIFICATION_SCORES:
                raise ValueError(f"For classification, nonconformity must be one of {list(_CLASSIFICATION_SCORES)}")
            if not hasattr(self.base_model, "predict_proba"):
                raise ValueError(
                    f"{type(self.base_model).__name__} has no predict_proba, which "
                    "conformal classification needs. Use a probabilistic classifier, "
                    "e.g. SVC(probability=True) or CalibratedClassifierCV."
                )
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def fit(self, X_train, y_train, X_calib, y_calib):
        """
        Fit model on training data and calibrate using calibration set.
        """
        self.base_model.fit(X_train, y_train)
        y_calib = np.asarray(y_calib)

        if self.task_type == "regression":
            preds = np.asarray(self.base_model.predict(X_calib))
            y_calib, preds = _as_single_output(y_calib, preds)
            residuals = y_calib - preds

            if self.nonconformity == "quantile":
                # Signed residuals: one-sided quantiles at alpha/2 on each side
                self.q = upper_quantile(residuals, self.alpha / 2)
                self.q_lower = lower_quantile(residuals, self.alpha / 2)
            else:
                self.q = upper_quantile(np.abs(residuals), self.alpha)

        else:
            probas = self.base_model.predict_proba(X_calib)
            self.classes_ = model_classes(self.base_model, probas.shape[1])
            idx = encode_labels(y_calib, self.classes_)
            scores = 1 - probas[np.arange(len(idx)), idx]
            self.q = upper_quantile(scores, self.alpha)

        self.is_fitted = True
        return self

    def predict(self, X, return_interval=False):
        """
        Predict with conformal intervals (regression) or prediction sets (classification).

        For classification, prediction sets contain class labels. With
        ``return_interval=True`` it returns ``(labels, sets)``, where a label is
        the single member of a singleton set and -1 otherwise.
        """
        if not self.is_fitted:
            raise RuntimeError("ConformalUQ model is not fitted yet.")

        if self.task_type == "regression":
            preds = np.asarray(self.base_model.predict(X))
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.ravel()

            if self.nonconformity == "quantile":
                # Asymmetric intervals from quantile-based scores
                intervals = [(p + self.q_lower, p + self.q) for p in preds]
            else:
                # Symmetric intervals
                intervals = [(p - self.q, p + self.q) for p in preds]

            return (preds, intervals) if return_interval else preds

        probas = self.base_model.predict_proba(X)
        pred_sets = prediction_sets(probas, self.q, self.classes_)

        if return_interval:
            return point_labels(pred_sets), pred_sets
        return pred_sets


def _as_single_output(y, preds):
    """Flatten (n, 1) targets; reject multi-output regression."""
    if y.ndim > 1:
        if y.shape[1] != 1:
            raise ValueError(
                "ConformalUQ supports a single regression target; "
                f"got y with shape {y.shape}."
            )
        y = y.ravel()
    return y, preds.reshape(len(y))
