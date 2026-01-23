"""
Online and Adaptive Conformal Prediction.

Implements streaming conformal prediction methods that maintain coverage
under non-stationary data and distribution drift.
"""

import numpy as np
from typing import Optional
from collections import deque


class OnlineConformalUQ:
    """
    Online / Adaptive Conformal Prediction for streaming data.
    
    Maintains valid coverage under non-stationary distributions using
    rolling calibration windows.
    
    Parameters
    ----------
    model : object
        Base prediction model (sklearn-compatible).
    alpha : float, default=0.05
        Significance level for prediction intervals.
    calibration_window : int, default=500
        Size of rolling calibration window.
    mode : {"rolling", "expanding"}, default="rolling"
        How to update calibration set:
        - "rolling": Fixed-size sliding window
        - "expanding": Growing window (all historical data)
    task_type : str, default="regression"
        Either "regression" or "classification".
    """
    
    def __init__(self, model, alpha=0.05, calibration_window=500, 
                 mode="rolling", task_type="regression"):
        self.base_model = model
        self.alpha = alpha
        self.calibration_window = calibration_window
        self.mode = mode
        self.task_type = task_type
        
        # Rolling calibration buffer
        if mode == "rolling":
            self.calib_scores = deque(maxlen=calibration_window)
        else:
            self.calib_scores = []
        
        self.q = None
        self.is_fitted = False
        self.n_updates = 0
    
    def fit(self, X_train, y_train, X_calib, y_calib):
        """
        Initial fit on training data and calibration set.
        """
        self.base_model.fit(X_train, y_train)
        
        # Compute initial calibration scores
        if self.task_type == "regression":
            preds = self.base_model.predict(X_calib)
            scores = np.abs(y_calib - preds)
        elif self.task_type == "classification":
            probas = self.base_model.predict_proba(X_calib)
            true_class_probs = probas[np.arange(len(y_calib)), y_calib]
            scores = 1 - true_class_probs
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
        
        # Initialize calibration buffer
        for score in scores:
            self.calib_scores.append(score)
        
        self._update_quantile()
        self.is_fitted = True
        return self
    
    def update(self, X_new, y_new):
        """
        Update calibration with new observed data (online learning).
        
        Parameters
        ----------
        X_new : array-like, shape (n_samples, n_features)
            New input samples.
        y_new : array-like, shape (n_samples,)
            True labels for the new samples.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before online updates")
        
        # Get predictions on new data
        if self.task_type == "regression":
            preds = self.base_model.predict(X_new)
            new_scores = np.abs(y_new - preds)
        elif self.task_type == "classification":
            probas = self.base_model.predict_proba(X_new)
            true_class_probs = probas[np.arange(len(y_new)), y_new]
            new_scores = 1 - true_class_probs
        
        # Add new scores to calibration buffer
        for score in new_scores:
            self.calib_scores.append(score)
        
        # Update quantile
        self._update_quantile()
        self.n_updates += 1
    
    def _update_quantile(self):
        """Recompute conformal quantile from current calibration buffer."""
        if len(self.calib_scores) == 0:
            self.q = 0
            return
        
        scores_array = np.array(self.calib_scores)
        n = len(scores_array)
        k = int(np.ceil((1 - self.alpha) * (n + 1)))
        # Fix: use min(k - 1, n - 1) for proper indexing
        self.q = np.sort(scores_array)[min(k - 1, n - 1)]
    
    def predict(self, X, return_interval=False):
        """
        Predict with current conformal intervals.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")
        
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
    
    def get_calibration_size(self):
        """Return current calibration set size."""
        return len(self.calib_scores)


class AdaptiveConformalUQ(OnlineConformalUQ):
    """
    Adaptive Conformal Prediction with drift-aware recalibration.
    
    Automatically triggers recalibration when drift is detected based on
    coverage violations.
    
    Parameters
    ----------
    model : object
        Base prediction model.
    alpha : float, default=0.05
        Target miscoverage rate.
    calibration_window : int, default=500
        Rolling window size.
    drift_threshold : float, default=0.1
        Maximum acceptable deviation from target coverage before recalibration.
    check_interval : int, default=50
        How often to check coverage and trigger recalibration.
    """
    
    def __init__(self, model, alpha=0.05, calibration_window=500,
                 drift_threshold=0.1, check_interval=50, task_type="regression"):
        super().__init__(model, alpha, calibration_window, "rolling", task_type)
        self.drift_threshold = drift_threshold
        self.check_interval = check_interval
        self.recent_coverage = deque(maxlen=check_interval)
        self.recalibration_count = 0
    
    def update(self, X_new, y_new):
        """
        Update with drift-aware recalibration.
        """
        # Get predictions before update
        if self.task_type == "regression":
            preds = self.base_model.predict(X_new)
            covered = np.abs(y_new - preds) <= self.q
        elif self.task_type == "classification":
            probas = self.base_model.predict_proba(X_new)
            pred_sets = [set(np.where(p >= 1 - self.q)[0]) for p in probas]
            covered = np.array([y in ps for y, ps in zip(y_new, pred_sets)])
        
        # Track recent coverage
        for c in covered:
            self.recent_coverage.append(c)
        
        # Standard online update
        super().update(X_new, y_new)
        
        # Check if recalibration needed
        if len(self.recent_coverage) == self.check_interval:
            empirical_coverage = np.mean(self.recent_coverage)
            target_coverage = 1 - self.alpha
            
            if abs(empirical_coverage - target_coverage) > self.drift_threshold:
                # Trigger recalibration
                self._recalibrate()
                self.recalibration_count += 1
    
    def _recalibrate(self):
        """Recompute quantile (already done in parent update)."""
        self._update_quantile()
