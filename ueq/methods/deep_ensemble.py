import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np


class DeepEnsembleUQ:
    """
    Deep Ensembles for Uncertainty Quantification.
    (Lakshminarayanan et al., 2017)

    Trains multiple neural networks independently with different initializations
    and aggregates their predictions to estimate uncertainty.

    Parameters
    ----------
    model_fn : callable
        A function that returns a fresh instance of the base model (nn.Module).
    n_models : int, default=5
        Number of ensemble members.
    device : str, default="cpu"
        Device to train models on ("cpu" or "cuda").
    """

    def __init__(self, model_fn, n_models=5, device="cpu"):
        self.model_fn = model_fn
        self.n_models = n_models
        self.device = device
        self.models = []
        self.is_fitted = False

    def fit(self, train_loader, criterion, optimizer_fn, epochs=10):
        """
        Train all ensemble members independently.

        Parameters
        ----------
        train_loader : DataLoader
            Training data.
        criterion : loss function
            Loss function (e.g., nn.MSELoss).
        optimizer_fn : callable
            Function that takes model parameters and returns an optimizer.
        epochs : int
            Number of training epochs per model.
        """
        self.models = []

        for i in range(self.n_models):
            model = self.model_fn().to(self.device)
            optimizer = optimizer_fn(model.parameters())

            for epoch in range(epochs):
                model.train()
                for X, y in train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    preds = model(X)
                    loss = criterion(preds, y)
                    loss.backward()
                    optimizer.step()

            self.models.append(deepcopy(model))

        self.is_fitted = True
        return self

    def predict(self, X, return_interval=True, alpha=0.05):
        """
        Predict with uncertainty estimates.

        Parameters
        ----------
        X : torch.Tensor
            Input data.
        return_interval : bool
            Whether to return prediction intervals.
        alpha : float
            Confidence level (e.g., 0.05 = 95% CI).

        Returns
        -------
        mean : np.ndarray
            Mean predictions across ensemble.
        intervals : list of tuples
            Prediction intervals (lower, upper) if return_interval=True.
        """
        if not self.is_fitted:
            raise RuntimeError("DeepEnsembleUQ is not fitted yet.")

        self.models = [m.eval() for m in self.models]
        with torch.no_grad():
            preds = [m(X.to(self.device)).cpu().numpy().squeeze() for m in self.models]

        preds = np.array(preds)  # shape: (n_models, n_samples)
        mean = preds.mean(axis=0)

        if return_interval:
            lower = np.percentile(preds, 100 * alpha / 2, axis=0)
            upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
            return mean, list(zip(lower, upper))

        return mean

    def predict_dist(self, X):
        """
        Return full predictive distribution (all ensemble predictions).
        """
        if not self.is_fitted:
            raise RuntimeError("DeepEnsembleUQ is not fitted yet.")

        with torch.no_grad():
            preds = [m(X.to(self.device)).cpu().numpy().squeeze() for m in self.models]

        return np.array(preds)  # (n_models, n_samples)
