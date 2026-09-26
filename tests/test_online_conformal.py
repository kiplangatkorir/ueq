import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from ueq import ExperimentalWarning
from ueq.methods.conformal import ConformalUQ
from ueq.methods.online_conformal import AdaptiveConformalUQ, OnlineConformalUQ


def noise_shift_stream(seed, n_fit=600, n_stream=3000, scale_after=3.0):
    """Linear data whose noise level jumps at the start of the stream."""
    rng = np.random.default_rng(seed)
    n = n_fit + n_stream
    X = rng.normal(size=(n, 3))
    noise = rng.normal(size=n)
    noise[n_fit:] *= scale_after
    y = X @ [1.0, -2.0, 0.5] + noise
    half = n_fit // 2
    return (X[:half], y[:half]), (X[half:n_fit], y[half:n_fit]), (X[n_fit:], y[n_fit:])


def stream_coverage(uq, X, y, batch=50):
    """Predict each batch, then reveal its labels; returns per-point coverage."""
    covered = []
    for i in range(0, len(y), batch):
        Xb, yb = X[i:i + batch], y[i:i + batch]
        _, intervals = uq.predict(Xb, return_interval=True)
        lo, hi = np.asarray(intervals).T
        covered.extend((yb >= lo) & (yb <= hi))
        if hasattr(uq, "update"):
            uq.update(Xb, yb)
    return np.asarray(covered)


@pytest.mark.statistical
def test_rolling_window_recovers_coverage_after_noise_shift():
    alpha, static_cov, rolling_cov = 0.1, [], []
    for seed in range(5):
        train, calib, stream = noise_shift_stream(seed)
        static = ConformalUQ(LinearRegression(), alpha=alpha).fit(*train, *calib)
        rolling = OnlineConformalUQ(LinearRegression(), alpha=alpha,
                                    calibration_window=500).fit(*train, *calib)
        static_cov.append(stream_coverage(static, *stream)[1000:].mean())
        rolling_cov.append(stream_coverage(rolling, *stream)[1000:].mean())

    # Once the window holds only post-shift scores, rolling is back near 1 - alpha
    # while the static quantile keeps under-covering.
    assert np.mean(static_cov) < 0.6
    assert np.mean(rolling_cov) > 0.86


def test_online_classification_uses_model_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(900, 2))
    y = np.where(X[:, 0] + 0.3 * rng.normal(size=900) > 0, "up", "down")
    uq = OnlineConformalUQ(LogisticRegression(), alpha=0.1, task_type="classification")
    uq.fit(X[:300], y[:300], X[300:600], y[300:600])
    uq.update(X[600:700], y[600:700])
    sets = uq.predict(X[700:])
    assert set().union(*sets) <= {"up", "down"}
    assert np.mean([t in s for t, s in zip(y[700:].tolist(), sets)]) >= 0.85


def test_adaptive_conformal_is_experimental():
    with pytest.warns(ExperimentalWarning, match="AdaptiveConformalUQ"):
        AdaptiveConformalUQ(LinearRegression())
