import numpy as np
import torch
import torch.nn as nn


class MCDropoutUQ:
    """
    Monte Carlo Dropout for Uncertainty Quantification.

    Runs multiple stochastic forward passes with dropout enabled at inference time,
    producing predictive mean and uncertainty estimates.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model with dropout layers.
    n_forward_passes : int, default=50
        Number of stochastic passes to estimate uncertainty.
    device : str, default="cpu"
        Device to run on ("cpu" or "cuda").
    """

    def __init__(self, model, n_forward_passes=50, device="cpu"):
        self.model = model.to(device)
        self.n_forward_passes = n_forward_passes
        self.device = device
        self.is_fitted = False

    def fit(self, train_loader, criterion, optimizer, epochs=10):
        """
        Train the PyTorch model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        criterion : torch.nn.Module
            Loss function.
        optimizer : torch.optim.Optimizer
            Optimizer.
        epochs : int
            Number of training epochs.
        """
        self.model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def predict(self, X):
        """
        Predict with MC Dropout uncertainty estimates.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        mean_preds : np.ndarray
            Predictive mean.
        std_preds : np.ndarray
            Predictive standard deviation (uncertainty).
        """
        if not self.is_fitted:
            raise RuntimeError("MCDropoutUQ model is not trained yet.")

        X = X.to(self.device)

        self.model.eval()
        self._enable_dropout()

        preds = []
        with torch.no_grad():
            for _ in range(self.n_forward_passes):
                outputs = self.model(X)
                preds.append(outputs.cpu().numpy())

        preds = np.array(preds)  # shape: (n_passes, batch_size, output_dim)

        mean_preds = preds.mean(axis=0)
        std_preds = preds.std(axis=0)

        return mean_preds, std_preds
