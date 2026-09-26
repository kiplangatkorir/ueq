import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

from ueq.methods.conformal import ConformalUQ
from ueq.methods.online_conformal import OnlineConformalUQ


def regression_splits(n_calib, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200 + n_calib + 50, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.normal(size=len(X))
    return (X[:200], y[:200]), (X[200:200 + n_calib], y[200:200 + n_calib]), (X[-50:], y[-50:])


def test_small_calibration_gives_infinite_symmetric_bounds():
    train, calib, test = regression_splits(n_calib=10)
    uq = ConformalUQ(LinearRegression(), alpha=0.05)
    with pytest.warns(UserWarning, match="too small"):
        uq.fit(*train, *calib)
    _, intervals = uq.predict(test[0], return_interval=True)
    assert all(lo == -np.inf and hi == np.inf for lo, hi in intervals)


def test_calibration_at_exact_minimum_is_finite():
    # n = 9 at alpha = 0.1 is the smallest valid size; float error must not
    # push ceil(0.9 * 10) to 10.
    train, calib, test = regression_splits(n_calib=9)
    uq = ConformalUQ(LinearRegression(), alpha=0.1).fit(*train, *calib)
    assert np.isfinite(uq.q)


def test_small_calibration_gives_infinite_asymmetric_bounds():
    train, calib, test = regression_splits(n_calib=10)
    uq = ConformalUQ(LinearRegression(), alpha=0.1, nonconformity="quantile")
    with pytest.warns(UserWarning, match="too small"):
        uq.fit(*train, *calib)
    assert uq.q == np.inf
    assert uq.q_lower == -np.inf


def test_small_calibration_gives_full_label_sets():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] > 0).astype(int)
    uq = ConformalUQ(LogisticRegression(), alpha=0.05, task_type="classification")
    with pytest.warns(UserWarning, match="too small"):
        uq.fit(X[:200], y[:200], X[200:210], y[200:210])
    assert all(s == {0, 1} for s in uq.predict(X[250:]))


def test_online_buffer_gives_infinite_bounds_until_full():
    train, calib, test = regression_splits(n_calib=5)
    uq = OnlineConformalUQ(LinearRegression(), alpha=0.1)
    with pytest.warns(UserWarning, match="buffer holds 5"):
        uq.fit(*train, *calib)
    assert uq.q == np.inf

    uq.update(test[0][:4], test[1][:4])
    assert np.isfinite(uq.q)


@pytest.mark.parametrize(
    "kwargs",
    [{"nonconformity": "normalized"},
     {"task_type": "classification", "nonconformity": "margin"},
     {"task_type": "classification", "class_conditional": True}],
)
def test_disabled_options_raise(kwargs):
    with pytest.raises(NotImplementedError):
        ConformalUQ(LogisticRegression(), **kwargs)


def test_multi_output_regression_raises():
    X, y = make_regression(n_samples=300, n_features=3, n_targets=2, random_state=0)
    uq = ConformalUQ(LinearRegression())
    with pytest.raises(ValueError, match="single regression target"):
        uq.fit(X[:150], y[:150], X[150:], y[150:])


def test_column_vector_target_is_accepted():
    train, calib, test = regression_splits(n_calib=100)
    uq = ConformalUQ(LinearRegression(), alpha=0.1)
    uq.fit(train[0], train[1][:, None], calib[0], calib[1][:, None])
    preds, intervals = uq.predict(test[0], return_interval=True)
    assert preds.shape == (50,)
    assert np.isfinite(uq.q)


@pytest.mark.statistical
@pytest.mark.parametrize("nonconformity", ["residual", "quantile"])
def test_regression_coverage_over_repeated_trials(nonconformity):
    alpha, n_calib, n_test, trials = 0.1, 200, 200, 50
    coverages = []
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(400 + n_calib + n_test, 3))
        noise = rng.normal(size=len(X)) * (1 + np.abs(X[:, 0]))
        y = X @ [1.0, -2.0, 0.5] + noise
        a, b = 400, 400 + n_calib
        uq = ConformalUQ(LinearRegression(), alpha=alpha, nonconformity=nonconformity)
        uq.fit(X[:a], y[:a], X[a:b], y[a:b])
        _, intervals = uq.predict(X[b:], return_interval=True)
        lo, hi = np.asarray(intervals).T
        coverages.append(np.mean((y[b:] >= lo) & (y[b:] <= hi)))

    # The asymmetric score splits alpha between two one-sided quantiles, so its
    # guarantee is 1 - alpha with slightly more slack; both sit near 1 - alpha.
    sd = np.sqrt(alpha * (1 - alpha) * (1 / (n_calib + 2) + 1 / n_test) / trials)
    assert np.mean(coverages) >= 1 - alpha - 4 * sd
    assert np.mean(coverages) <= 1 - alpha + 2 / (n_calib + 1) + 4 * sd
