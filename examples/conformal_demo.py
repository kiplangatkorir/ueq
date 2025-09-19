from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from ueq import UQ

# Generate data
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Base model
model = LinearRegression()

# Wrap with Uncertainty Everywhere (conformal)
uq = UQ(model, method="conformal", alpha=0.1)

# Fit with calibration split
uq.fit(X_train, y_train, X_calib, y_calib)

# Predict with intervals
pred, interval = uq.predict(X_test[:5])

print("Predictions:", pred)
print("Intervals:", interval)
