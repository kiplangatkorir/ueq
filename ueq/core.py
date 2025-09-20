from .methods.bootstrap import BootstrapUQ
from .methods.conformal import ConformalUQ
from .methods.mc_dropout import MCDropoutUQ
from .methods.deep_ensemble import DeepEnsembleUQ


class UQ:
    """
    Unified interface for Uncertainty Quantification (UQ).
    
    Parameters
    ----------
    model : object or callable, optional
        - For sklearn-style models: pass the model instance (Bootstrap, Conformal).
        - For deep learning ensembles: pass a model constructor (e.g., lambda: Net()).
        - Not required for Bayesian Linear Regression.
    method : str
        Uncertainty method. Options: 
        ["bootstrap", "conformal", "mc_dropout", "deep_ensemble", "bayesian_linear"].
    **kwargs :
        Additional arguments passed to the chosen method.
    """

    def __init__(self, model=None, method="bootstrap", **kwargs):
        self.model = model
        self.method = method.lower()
        self.uq_model = self._init_method(**kwargs)

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
                raise ValueError("MC Dropout requires a model constructor (e.g., lambda: Net()).")
            return MCDropoutUQ(self.model, **kwargs)

        elif self.method == "deep_ensemble":
            if self.model is None:
                raise ValueError("Deep Ensemble requires a model constructor (e.g., lambda: Net()).")
            return DeepEnsembleUQ(self.model, **kwargs)

        elif self.method == "bayesian_linear":
            from .methods.bayesian_linear import BayesianLinearUQ
            return BayesianLinearUQ(**kwargs)

        else:
            raise ValueError(f"Unknown UQ method: {self.method}")

    def fit(self, *args, **kwargs):
        """Fit the underlying model(s)."""
        return self.uq_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Predict with uncertainty estimates."""
        return self.uq_model.predict(*args, **kwargs)

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
