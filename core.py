from .methods.bootstrap import BootstrapUQ
from .methods.conformal import ConformalUQ
from .methods.mc_dropout import MCDropoutUQ


class UQ:
    """
    Unified interface for Uncertainty Quantification (UQ).
    
    Parameters
    ----------
    model : object
        Base model (e.g., scikit-learn estimator, PyTorch model).
    method : str
        Uncertainty method. Options: ["bootstrap", "conformal", "mc_dropout"].
    **kwargs :
        Additional arguments passed to the chosen method.
    """

    def __init__(self, model, method="bootstrap", **kwargs):
        self.model = model
        self.method = method.lower()
        self.uq_model = self._init_method(**kwargs)

    def _init_method(self, **kwargs):
        if self.method == "bootstrap":
            return BootstrapUQ(self.model, **kwargs)
        elif self.method == "conformal":
            return ConformalUQ(self.model, **kwargs)
        elif self.method == "mc_dropout":
            return MCDropoutUQ(self.model, **kwargs)
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
        """Optional calibration step."""
        if hasattr(self.uq_model, "calibrate"):
            return self.uq_model.calibrate(*args, **kwargs)
        else:
            raise NotImplementedError(f"{self.method} does not support calibrate().")
