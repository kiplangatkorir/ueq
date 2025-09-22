import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from ueq import UQ
from ueq.utils.metrics import coverage, sharpness, expected_calibration_error

# 1. Load stock data (Apple)
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
returns = data["Close"].pct_change().dropna().values.reshape(-1, 1)

# Features: lagged returns, Target: next-day return
X, y = returns[:-1], returns[1:].ravel()

# 2. Train-test split
n = int(0.8 * len(X))
X_train_full, X_test = X[:n], X[n:]
y_train_full, y_test = y[:n], y[n:]

# 2a. Further split training into train + calibration
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_full, y_train_full, test_size=0.2, shuffle=False
)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cal = scaler.transform(X_cal)
X_test = scaler.transform(X_test)

# 4. Bootstrap UQ with MLP
uq = UQ(model=MLPRegressor(max_iter=500), method="bootstrap", n_models=10)
uq.fit(X_train, y_train)

# Predictions
_, cal_intervals = uq.predict(X_cal, return_interval=True)
_, test_intervals = uq.predict(X_test, return_interval=True)

# 5. Conformal calibration
alpha = 0.05  # target miscoverage (95% coverage)
residuals = np.maximum(
    y_cal - cal_intervals[:, 0],  # below lower bound
    cal_intervals[:, 1] - y_cal   # above upper bound
)
q = np.quantile(residuals, 1 - alpha)

# Expand test intervals by conformal adjustment
conf_intervals = np.column_stack([
    test_intervals[:, 0] - q,
    test_intervals[:, 1] + q
])

# 6. Metrics
print("Finance Benchmark (Raw Intervals)")
print("Coverage:", coverage(y_test, test_intervals))
print("Sharpness:", sharpness(test_intervals))
print("ECE:", expected_calibration_error(y_test, test_intervals))

print("\nFinance Benchmark (Conformal-Calibrated Intervals)")
print("Coverage:", coverage(y_test, conf_intervals))
print("Sharpness:", sharpness(conf_intervals))
print("ECE:", expected_calibration_error(y_test, conf_intervals))
