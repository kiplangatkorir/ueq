import numpy as np
import torch
from typing import List, Union, Tuple, Any
from ..core import UQ


class CrossFrameworkEnsembleUQ:
    """
    Cross-framework ensemble for uncertainty quantification.
    
    Combines models from different frameworks (sklearn, PyTorch, etc.)
    to create a unified uncertainty estimate.
    
    Parameters
    ----------
    models : list
        List of models from different frameworks.
    weights : list, optional
        Weights for each model in the ensemble. If None, equal weights are used.
    aggregation_method : str, default="mean"
        Method for aggregating predictions: "mean", "median", or "weighted_mean".
    """
    
    def __init__(self, models: List[Any], weights: List[float] = None, 
                 aggregation_method: str = "mean"):
        self.models = models
        self.n_models = len(models)
        self.weights = weights or [1.0 / self.n_models] * self.n_models
        self.aggregation_method = aggregation_method
        self.uq_wrappers = []
        self.is_fitted = False
        
        # Validate inputs
        if len(self.weights) != self.n_models:
            raise ValueError("Number of weights must match number of models")
        
        if not np.isclose(sum(self.weights), 1.0):
            # Normalize weights
            self.weights = [w / sum(self.weights) for w in self.weights]
    
    def _create_uq_wrapper(self, model):
        """Create appropriate UQ wrapper for each model."""
        return UQ(model)
    
    def fit(self, *args, **kwargs):
        """
        Fit all models in the ensemble.
        
        Parameters
        ----------
        *args : 
            Training data arguments (X, y, etc.)
        **kwargs :
            Additional arguments passed to each model's fit method
        """
        self.uq_wrappers = []
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.n_models}: {type(model).__name__}")
            
            # Create UQ wrapper for this model
            uq_wrapper = self._create_uq_wrapper(model)
            
            # Fit the model
            try:
                uq_wrapper.fit(*args, **kwargs)
                self.uq_wrappers.append(uq_wrapper)
            except Exception as e:
                print(f"Warning: Failed to fit model {i+1}: {e}")
                # Continue with other models
        
        if not self.uq_wrappers:
            raise RuntimeError("No models were successfully fitted")
        
        self.is_fitted = True
        return self
    
    def predict(self, X, return_interval=True, alpha=0.05):
        """
        Predict with uncertainty estimates from cross-framework ensemble.
        
        Parameters
        ----------
        X : array-like
            Test inputs
        return_interval : bool, default=True
            Whether to return prediction intervals
        alpha : float, default=0.05
            Significance level for confidence intervals
            
        Returns
        -------
        mean_pred : np.ndarray
            Ensemble mean predictions
        intervals : list of tuples
            Prediction intervals if return_interval=True
        """
        if not self.is_fitted:
            raise RuntimeError("CrossFrameworkEnsembleUQ is not fitted yet")
        
        # Get predictions from all models
        all_predictions = []
        all_intervals = []
        
        for i, uq_wrapper in enumerate(self.uq_wrappers):
            try:
                if hasattr(uq_wrapper.uq_model, 'predict'):
                    pred_result = uq_wrapper.predict(X, return_interval=return_interval)
                    
                    if isinstance(pred_result, tuple):
                        pred, intervals = pred_result
                        all_predictions.append(pred)
                        all_intervals.append(intervals)
                    else:
                        all_predictions.append(pred_result)
                        
            except Exception as e:
                print(f"Warning: Failed to predict with model {i+1}: {e}")
                continue
        
        if not all_predictions:
            raise RuntimeError("No models produced valid predictions")
        
        # Aggregate predictions
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples)
        
        if self.aggregation_method == "mean":
            mean_pred = np.mean(all_predictions, axis=0)
        elif self.aggregation_method == "median":
            mean_pred = np.median(all_predictions, axis=0)
        elif self.aggregation_method == "weighted_mean":
            weights_array = np.array(self.weights[:len(all_predictions)])
            mean_pred = np.average(all_predictions, axis=0, weights=weights_array)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        if not return_interval:
            return mean_pred
        
        # Aggregate intervals
        if all_intervals:
            # Convert intervals to arrays for easier manipulation
            interval_arrays = []
            for intervals in all_intervals:
                if isinstance(intervals, list):
                    intervals = np.array(intervals)
                interval_arrays.append(intervals)
            
            interval_arrays = np.array(interval_arrays)  # Shape: (n_models, n_samples, 2)
            
            # Aggregate intervals using the same method
            if self.aggregation_method == "mean":
                agg_intervals = np.mean(interval_arrays, axis=0)
            elif self.aggregation_method == "median":
                agg_intervals = np.median(interval_arrays, axis=0)
            elif self.aggregation_method == "weighted_mean":
                weights_array = np.array(self.weights[:len(interval_arrays)])
                agg_intervals = np.average(interval_arrays, axis=0, weights=weights_array)
            
            # Convert back to list of tuples
            intervals = [(agg_intervals[i, 0], agg_intervals[i, 1]) 
                        for i in range(agg_intervals.shape[0])]
        else:
            # Fallback: create intervals based on prediction variance
            pred_std = np.std(all_predictions, axis=0)
            z_score = 1.96  # 95% confidence
            intervals = [(mean_pred[i] - z_score * pred_std[i], 
                         mean_pred[i] + z_score * pred_std[i]) 
                        for i in range(len(mean_pred))]
        
        return mean_pred, intervals
    
    def predict_dist(self, X):
        """
        Return full predictive distribution from all models.
        
        Returns
        -------
        predictions : np.ndarray
            Shape: (n_models, n_samples)
        """
        if not self.is_fitted:
            raise RuntimeError("CrossFrameworkEnsembleUQ is not fitted yet")
        
        all_predictions = []
        
        for uq_wrapper in self.uq_wrappers:
            try:
                if hasattr(uq_wrapper.uq_model, 'predict_dist'):
                    pred_dist = uq_wrapper.predict_dist(X)
                    all_predictions.append(pred_dist)
                else:
                    # Fallback to regular predict
                    pred = uq_wrapper.predict(X, return_interval=False)
                    all_predictions.append(pred)
            except Exception as e:
                print(f"Warning: Failed to get predictive distribution: {e}")
                continue
        
        if not all_predictions:
            raise RuntimeError("No models produced valid predictions")
        
        return np.array(all_predictions)
    
    def get_model_info(self):
        """Get information about all models in the ensemble."""
        info = {
            "n_models": len(self.uq_wrappers),
            "models": [],
            "weights": self.weights,
            "aggregation_method": self.aggregation_method
        }
        
        for i, uq_wrapper in enumerate(self.uq_wrappers):
            model_info = uq_wrapper.get_info()
            model_info["weight"] = self.weights[i]
            info["models"].append(model_info)
        
        return info
