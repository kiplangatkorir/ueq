# examples/benchmarks/climate_forecasting.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_openml

from ueq import UQ
from ueq.utils.metrics import coverage, sharpness, expected_calibration_error

# 1. Load dataset (Beijing PM2.5 Air Quality, regression task)
print("🌍 Loading climate dataset...")
data = fetch_openml("Beijing PM2.5", version=1, as_frame=True)
X = data.data.select_dtypes(include=[np.number]).fillna(0).values
y = data.target.values.astype(float)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. MC Dropout UQ
uq = UQ(
    model=lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
    method="mc_dropout",
    dropout_rate=0.2,
    n_forward_passes=30
)

uq.fit(X_train, y_train)
mean_pred, intervals = uq.predict(X_test, return_interval=True)

# 4. Metrics
print("🌍 Climate Forecasting Benchmark")
print("Coverage:", coverage(y_test, intervals))
print("Sharpness:", sharpness(intervals))
print("ECE:", expected_calibration_error(y_test, intervals))

# 5. Visualization
plt.figure(figsize=(10, 5))
plt.plot(y_test[:200], label="True", alpha=0.7)
plt.plot(mean_pred[:200], label="Predicted Mean", alpha=0.7)
plt.fill_between(
    np.arange(200),
    intervals[:200, 0],
    intervals[:200, 1],
    color="orange",
    alpha=0.3,
    label="Uncertainty Interval"
)
plt.legend()
plt.title("Climate Forecasting with MC Dropout UQ")
plt.show()
