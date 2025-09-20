import numpy as np
import matplotlib.pyplot as plt

from ueq.methods.bayesian_linear import BayesianLinearUQ


def main():
    # --- 1. Generate synthetic data (linear with noise) ---
    np.random.seed(42)
    n = 30
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    y = 2.5 * X.squeeze() + np.random.normal(0, 1, n)

    # --- 2. Fit Bayesian Linear UQ ---
    model = BayesianLinearUQ(alpha=2.0, beta=25.0)  # stronger prior, lower noise
    model.fit(X, y)

    # --- 3. Predictions with uncertainty ---
    X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
    mean, intervals = model.predict(X_test, return_interval=True)

    lower, upper = zip(*intervals)

    # --- 4. Plot ---
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X_test, mean, color="red", label="Predictive mean")
    plt.fill_between(
        X_test.squeeze(),
        lower,
        upper,
        color="orange",
        alpha=0.3,
        label="95% predictive interval"
    )
    plt.title("Bayesian Linear Regression with Uncertainty")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
