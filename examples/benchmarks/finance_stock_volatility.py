import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from ueq import UQ
from ueq.utils.metrics import coverage, sharpness, expected_calibration_error

# 1. Load stock data (Apple)
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
returns = data["Close"].pct_change().dropna().values.reshape(-1, 1)
X, y = returns[:-1], returns[1:]

# 2. Train-test split
n = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:n], X[n:], y[:n], y[n:]

# 3. Deep Ensemble
uq = UQ(
    model=lambda: MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=300),
    method="deep_ensemble",
    n_models=5
)
uq.fit(X_train, y_train)
mean_pred, intervals = uq.predict(X_test, return_interval=True)

print("Finance Benchmark")
print("Coverage:", coverage(y_test, intervals))
print("Sharpness:", sharpness(intervals))
print("ECE:", expected_calibration_error(y_test, intervals))
