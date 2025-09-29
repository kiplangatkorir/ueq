from .methods.bootstrap import BootstrapUQ
from .methods.conformal import ConformalUQ
from .methods.mc_dropout import MCDropoutUQ
from .methods.deep_ensemble import DeepEnsembleUQ

import numpy as np
import torch
import torch.nn as nn

class UQ:
    """
    Unified interface for Uncertainty Quantification (UQ) with auto-detection.
    
    Parameters
    ----------
    model : object or callable, optional
        - For sklearn-style models: pass the model instance (Bootstrap, Conformal).                                                                             
        - For deep learning ensembles: pass a model constructor (e.g., lambda: Net()).                                                                          
        - For PyTorch models: pass the model instance (MC Dropout)
        - Not required for Bayesian Linear Regression.
    method : str, default="auto"
        Uncertainty method. Options: 
        ["auto", "bootstrap", "conformal", "mc_dropout", "deep_ensemble", "bayesian_linear"].
        If "auto", the best method is automatically selected based on model type.                                                                           
    **kwargs :
        Additional arguments passed to the chosen method.
    """

    def __init__(self, model=None, method="auto", **kwargs):
        self.model = model
        self.models = None  # For multi-model ensembles
        
        # Check if model is a list (cross-framework ensemble)
        if isinstance(model, (list, tuple)):
            self.models = model
            self.model_type = "cross_framework_ensemble"
            self.method = "cross_ensemble"
        else:
            self.model_type = self._detect_model_type(model)
            self.method = self._auto_select_method(self.model_type, method.lower())
        
        self.uq_model = self._init_method(**kwargs)

    def _detect_model_type(self, model):
        """
        Auto-detect model framework and type.
        
        Returns
        -------
        str
            Model type: "none", "sklearn_classifier", "sklearn_regressor", 
            "pytorch", "constructor", "unknown"
        """
        if model is None:
            return "none"
        
        # Check for scikit-learn style models
        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            if hasattr(model, 'predict_proba'):
                return "sklearn_classifier"
            else:
                return "sklearn_regressor"
        
        # Check for PyTorch models
        if hasattr(model, 'forward') and hasattr(model, 'parameters'):
            return "pytorch"
        
        # Check for constructor functions
        if callable(model):
            return "constructor"
        
        return "unknown"

    def _auto_select_method(self, model_type, method):
        """
        Auto-select best UQ method based on model type.
        
        Parameters
        ----------
        model_type : str
            Detected model type
        method : str
            User-specified method (or "auto")
            
        Returns
        -------
        str
            Selected UQ method
        """
        if method != "auto":
            return method
        
        # Auto-selection mapping
        method_map = {
            "sklearn_regressor": "bootstrap",
            "sklearn_classifier": "conformal", 
            "pytorch": "mc_dropout",
            "constructor": "deep_ensemble",
            "none": "bayesian_linear"
        }
        
        selected_method = method_map.get(model_type, "bootstrap")
        
        if model_type == "unknown":
            raise ValueError(
                f"Unknown model type: {type(self.model)}. "
                "Please specify method explicitly or use a supported model type."
            )
        
        return selected_method

    def _init_method(self, **kwargs):
        if self.method == "bootstrap":
            if self.model is None:
                raise ValueError("Bootstrap requires a model instance.")        
            return BootstrapUQ(self.model, **kwargs)

        elif self.method == "conformal":
            if self.model is None:
                raise ValueError("Conformal requires a model instance.")        
            return ConformalUQ(self.model, **kwargs)

        elif self.method == "mc_dropout":
            if self.model is None:
                raise ValueError("MC Dropout requires a model instance or constructor.")
            
            # Handle both model instances and constructors
            if self.model_type == "pytorch":
                # Direct PyTorch model instance
                return MCDropoutUQ(self.model, **kwargs)
            elif self.model_type == "constructor":
                # Model constructor function
                return MCDropoutUQ(self.model(), **kwargs)
            else:
                raise ValueError(
                    f"MC Dropout requires a PyTorch model or constructor, "
                    f"got {self.model_type}"
                )

        elif self.method == "deep_ensemble":
            if self.model is None:
                raise ValueError("Deep Ensemble requires a model constructor (e.g., lambda: Net()).")                                                           
            return DeepEnsembleUQ(self.model, **kwargs)

        elif self.method == "bayesian_linear":
            from .methods.bayesian_linear import BayesianLinearUQ
            return BayesianLinearUQ(**kwargs)

        elif self.method == "cross_ensemble":
            from .methods.cross_ensemble import CrossFrameworkEnsembleUQ
            return CrossFrameworkEnsembleUQ(self.models, **kwargs)

        else:
            raise ValueError(f"Unknown UQ method: {self.method}")

    def fit(self, *args, **kwargs):
        """Fit the underlying model(s)."""
        return self.uq_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Predict with uncertainty estimates.
        
        Returns
        -------
        mean : np.ndarray
            Predicted mean values.
        intervals : np.ndarray, shape (n_samples, 2), optional
            Lower and upper bounds of prediction intervals 
            (only if return_interval=True).
        """
        preds = self.uq_model.predict(*args, **kwargs)

        if isinstance(preds, tuple) and len(preds) == 2:
            mean, intervals = preds
            intervals = np.array(intervals)  # ensure array, not list of tuples
            return mean, intervals

        return preds
    def predict_dist(self, *args, **kwargs):
        """Return predictive distribution (if available)."""
        if hasattr(self.uq_model, "predict_dist"):
            return self.uq_model.predict_dist(*args, **kwargs)
        else:
            raise NotImplementedError(f"{self.method} does not support predict_dist().")

    def calibrate(self, *args, **kwargs):
        """Optional calibration step (for conformal methods, etc.)."""
        if hasattr(self.uq_model, "calibrate"):
            return self.uq_model.calibrate(*args, **kwargs)
        else:
            raise NotImplementedError(f"{self.method} does not support calibrate().")

    def get_info(self):
        """
        Get information about the detected model type and selected method.
        
        Returns
        -------
        dict
            Dictionary with model_type, method, and model_class information
        """
        if self.model_type == "cross_framework_ensemble":
            return {
                "model_type": self.model_type,
                "method": self.method,
                "n_models": len(self.models),
                "model_classes": [type(m).__name__ for m in self.models],
                "ensemble_info": self.uq_model.get_model_info() if hasattr(self.uq_model, 'get_model_info') else None
            }
        else:
            return {
                "model_type": self.model_type,
                "method": self.method,
                "model_class": type(self.model).__name__ if self.model is not None else None
            }

    def predict_large_dataset(self, X, batch_size=1000, return_interval=True):
        """
        Memory-efficient prediction for large datasets.
        
        Parameters
        ----------
        X : array-like
            Input data
        batch_size : int, default=1000
            Batch size for processing
        return_interval : bool, default=True
            Whether to return uncertainty intervals
            
        Returns
        -------
        tuple or array
            Predictions and optionally uncertainty estimates
        """
        from .utils.performance import memory_efficient_predict
        return memory_efficient_predict(
            self, X, batch_size=batch_size, 
            return_uncertainty=return_interval
        )

    def monitor(self, X, y=None, baseline_data=None, baseline_uncertainty=None):
        """
        Monitor model performance and uncertainty drift.
        
        Parameters
        ----------
        X : array-like
            Input data
        y : array-like, optional
            True target values
        baseline_data : array-like, optional
            Baseline data for drift detection
        baseline_uncertainty : array-like, optional
            Baseline uncertainty estimates
            
        Returns
        -------
        dict
            Monitoring results
        """
        from .utils.monitoring import UQMonitor
        
        # Get predictions
        try:
            predictions, uncertainty = self.predict(X, return_interval=True)
        except:
            predictions = self.predict(X, return_interval=False)
            uncertainty = None
        
        # Create monitor
        monitor = UQMonitor(
            baseline_data=baseline_data,
            baseline_uncertainty=baseline_uncertainty
        )
        
        # Monitor
        results = monitor.monitor(predictions, uncertainty)
        
        return results

    def benchmark(self, X, y=None, methods=None):
        """
        Benchmark model performance.
        
        Parameters
        ----------
        X : array-like
            Input data
        y : array-like, optional
            True target values
        methods : dict, optional
            Additional methods to benchmark
            
        Returns
        -------
        dict
            Benchmark results
        """
        from .utils.performance import PerformanceProfiler
        
        profiler = PerformanceProfiler()
        
        # Default methods
        if methods is None:
            methods = {
                'predict': lambda x: self.predict(x, return_interval=False),
                'predict_with_uncertainty': lambda x: self.predict(x, return_interval=True)
            }
        
        return profiler.benchmark_methods(methods, X)
