import numpy as np

class ConformalUQ:
    """
    Conformal Prediction for Uncertainty Quantification.

    Supports both regression (intervals) and classification (prediction sets)
    with multiple nonconformity score options.

    Parameters
    ----------
    model : object
        Any scikit-learn compatible estimator.
    alpha : float, default=0.05
        Significance level (e.g., 0.05 = 95% confidence level).
    task_type : str, default="regression"
        Either "regression" or "classification".
    nonconformity : str, default="residual"
        Nonconformity score type. For regression: "residual", "quantile", "normalized".
        For classification: "margin", "inverse_probability".
    class_conditional : bool, default=False
        If True, performs class-conditional calibration for classification tasks.
        Calibrates separately per class to ensure valid coverage within each class.
    """

    def __init__(self, model, alpha=0.05, task_type="regression", nonconformity="residual", 
                 class_conditional=False):
        self.base_model = model
        self.alpha = alpha
        self.task_type = task_type
        self.nonconformity = nonconformity
        self.class_conditional = class_conditional
        self.q = None
        self.q_lower = None  # For quantile-based methods
        self.q_per_class = {}  # For class-conditional calibration
        self.is_fitted = False
        self._validate_nonconformity()
    
    def _validate_nonconformity(self):
        """Validate nonconformity score selection."""
        regression_scores = ["residual", "quantile", "normalized"]
        classification_scores = ["margin", "inverse_probability"]
        
        if self.task_type == "regression" and self.nonconformity not in regression_scores:
            raise ValueError(f"For regression, nonconformity must be one of {regression_scores}")
        elif self.task_type == "classification" and self.nonconformity not in classification_scores:
            raise ValueError(f"For classification, nonconformity must be one of {classification_scores}")

    def fit(self, X_train, y_train, X_calib, y_calib):
        """
        Fit model on training data and calibrate using calibration set.
        """
        self.base_model.fit(X_train, y_train)

        if self.task_type == "regression":
            preds = self.base_model.predict(X_calib)
            scores = self._compute_regression_scores(y_calib, preds, X_calib)
            n = len(scores)
            k = int(np.ceil((1 - self.alpha) * (n + 1)))
            
            if self.nonconformity == "quantile":
                # For quantile, compute both upper and lower quantiles
                k_upper = int(np.ceil((1 - self.alpha/2) * (n + 1)))
                k_lower = int(np.floor((self.alpha/2) * (n + 1)))
                residuals = y_calib - preds
                sorted_residuals = np.sort(residuals)
                self.q = sorted_residuals[min(k_upper - 1, n - 1)]  # upper quantile (fix indexing)
                self.q_lower = sorted_residuals[max(k_lower - 1, 0)]  # lower quantile (fix indexing)
            else:
                self.q = np.sort(scores)[min(k - 1, n - 1)]  # fix indexing

        elif self.task_type == "classification":
            probas = self.base_model.predict_proba(X_calib)
            
            if self.class_conditional:
                # Calibrate separately for each class
                unique_classes = np.unique(y_calib)
                min_samples = 10  # Minimum samples per class for calibration
                
                for cls in unique_classes:
                    cls_mask = (y_calib == cls)
                    n_cls = cls_mask.sum()
                    
                    if n_cls < min_samples:
                        # Fallback to global calibration for rare classes
                        scores_all = self._compute_classification_scores(y_calib, probas)
                        k_all = int(np.ceil((1 - self.alpha) * (len(scores_all) + 1)))
                        self.q_per_class[cls] = np.sort(scores_all)[min(k_all, len(scores_all)) - 1]
                    else:
                        # Class-specific calibration
                        scores_cls = self._compute_classification_scores(
                            y_calib[cls_mask], probas[cls_mask]
                        )
                        k_cls = int(np.ceil((1 - self.alpha) * (n_cls + 1)))
                        self.q_per_class[cls] = np.sort(scores_cls)[min(k_cls, n_cls) - 1]
            else:
                # Global calibration
                scores = self._compute_classification_scores(y_calib, probas)
                n = len(scores)
                k = int(np.ceil((1 - self.alpha) * (n + 1)))
                self.q = np.sort(scores)[min(k, n) - 1]

        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        self.is_fitted = True
        return self
    
    def _compute_regression_scores(self, y_true, y_pred, X=None):
        """Compute nonconformity scores for regression."""
        if self.nonconformity == "residual":
            return np.abs(y_true - y_pred)
        elif self.nonconformity == "quantile":
            # For quantile-based, return signed residuals
            return y_true - y_pred
        elif self.nonconformity == "normalized":
            # Normalized residuals (assumes model can provide std estimates)
            residuals = np.abs(y_true - y_pred)
            # Simple normalization by mean absolute deviation
            mad = np.median(residuals)
            if mad == 0:
                mad = 1.0  # Avoid division by zero
            return residuals / mad
        else:
            raise ValueError(f"Unknown regression nonconformity score: {self.nonconformity}")
    
    def _compute_classification_scores(self, y_true, probas):
        """Compute nonconformity scores for classification."""
        if self.nonconformity == "inverse_probability":
            # Standard conformal: 1 - P(true class)
            true_class_probs = probas[np.arange(len(y_true)), y_true]
            return 1 - true_class_probs
        elif self.nonconformity == "margin":
            # Margin-based: difference between top two probabilities
            sorted_probs = np.sort(probas, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            # For true class not being top, adjust margin
            true_class_probs = probas[np.arange(len(y_true)), y_true]
            top_probs = sorted_probs[:, -1]
            scores = np.where(true_class_probs == top_probs, 
                            1 - margins,  # True class is top
                            1 + margins)  # True class is not top
            return scores
        else:
            raise ValueError(f"Unknown classification nonconformity score: {self.nonconformity}")

    def predict(self, X, return_interval=False):
        """
        Predict with conformal intervals (regression) or prediction sets (classification).
        """
        if not self.is_fitted:
            raise RuntimeError("ConformalUQ model is not fitted yet.")

        if self.task_type == "regression":
            preds = self.base_model.predict(X)
            
            if self.nonconformity == "quantile":
                # Asymmetric intervals from quantile-based scores
                intervals = [(p + self.q_lower, p + self.q) for p in preds]
            else:
                # Symmetric intervals
                intervals = [(p - self.q, p + self.q) for p in preds]
            
            return (preds, intervals) if return_interval else preds

        elif self.task_type == "classification":
            probas = self.base_model.predict_proba(X)
            
            if self.class_conditional:
                # Use class-specific thresholds
                pred_sets = []
                for p in probas:
                    # For each sample, check which classes to include
                    # Use the max probability class to determine which threshold to use
                    pred_class = np.argmax(p)
                    threshold = self.q_per_class.get(pred_class)
                    
                    # If no class-specific threshold, fall back to global mean
                    if threshold is None:
                        threshold = np.mean(list(self.q_per_class.values()))
                    
                    pred_sets.append(set(np.where(p >= 1 - threshold)[0]))
            else:
                # Global threshold
                if self.nonconformity == "inverse_probability":
                    pred_sets = [set(np.where(p >= 1 - self.q)[0]) for p in probas]
                elif self.nonconformity == "margin":
                    # For margin-based, include classes with high enough probability
                    pred_sets = [set(np.where(p >= 1 - self.q)[0]) for p in probas]
                else:
                    pred_sets = [set(np.where(p >= 1 - self.q)[0]) for p in probas]

            if return_interval:
                labels = [list(s)[0] if len(s) == 1 else -1 for s in pred_sets]
                return labels, pred_sets
            else:
                return pred_sets
