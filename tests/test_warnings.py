"""Warnings on known-invalid defaults and experimental modules, and specs for the
recalibration behaviour those modules still lack."""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from ueq import UQ, ExperimentalWarning, DriftAwareRecalibrator, UncertaintyInflator, UQMonitor
from ueq.methods.bayesian_linear import BayesianLinearUQ
from ueq.methods.bootstrap import BootstrapUQ
from ueq.methods.online_conformal import AdaptiveConformalUQ, OnlineConformalUQ


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.normal(size=300)
    return X, y


def test_auto_bootstrap_warns_about_the_default_change():
    with pytest.warns(FutureWarning, match="split conformal in 1.1.0"):
        uq = UQ(LinearRegression())
    assert uq.method == "bootstrap"


def test_explicit_method_does_not_warn_about_the_default():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        UQ(LinearRegression(), method="conformal")


def test_bootstrap_interval_warning_is_issued_once(data):
    X, y = data
    uq = BootstrapUQ(LinearRegression(), n_models=5, random_state=0).fit(X, y)
    with pytest.warns(UserWarning, match="mean prediction"):
        uq.predict(X[:5])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        uq.predict(X[:5])
        uq.predict(X[:5], return_interval=False)


def test_bayesian_linear_warns_about_fixed_noise(data):
    X, y = data
    uq = BayesianLinearUQ().fit(X, y)
    with pytest.warns(UserWarning, match="fixed noise precision"):
        uq.predict(X[:5])


@pytest.mark.parametrize(
    "make",
    [lambda: UQMonitor(),
     lambda: DriftAwareRecalibrator(OnlineConformalUQ(LinearRegression())),
     lambda: UncertaintyInflator(OnlineConformalUQ(LinearRegression()))],
    ids=["UQMonitor", "DriftAwareRecalibrator", "UncertaintyInflator"],
)
def test_experimental_modules_warn(make):
    with pytest.warns(ExperimentalWarning):
        make()


def test_cross_framework_ensemble_is_experimental():
    from ueq.methods.cross_ensemble import CrossFrameworkEnsembleUQ

    with pytest.warns(ExperimentalWarning, match="CrossFrameworkEnsembleUQ"):
        CrossFrameworkEnsembleUQ([LinearRegression(), LinearRegression()])


def test_uq_monitor_method_warns_once(data):
    X, y = data
    uq = UQ(LinearRegression(), method="conformal", alpha=0.1)
    uq.fit(X[:150], y[:150], X[150:250], y[150:250])
    with pytest.warns(ExperimentalWarning) as record:
        uq.monitor(X[250:])
    assert sum(issubclass(w.category, ExperimentalWarning) for w in record) == 1


def shifted_stream(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2600, 3))
    noise = rng.normal(size=2600)
    noise[600:] *= 3.0
    y = X @ [1.0, -2.0, 0.5] + noise
    return (X[:300], y[:300]), (X[300:600], y[300:600]), (X[600:], y[600:])


def feed(uq, X, y, batch=50):
    for i in range(0, len(y), batch):
        uq.update(X[i:i + batch], y[i:i + batch])


@pytest.mark.xfail(strict=True, reason="AdaptiveConformalUQ recalibration is a no-op (ROADMAP Phase 2)")
def test_adaptive_recalibration_changes_the_quantile():
    train, calib, stream = shifted_stream()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adaptive = AdaptiveConformalUQ(LinearRegression(), alpha=0.1).fit(*train, *calib)
        rolling = OnlineConformalUQ(LinearRegression(), alpha=0.1).fit(*train, *calib)
        adaptive_q, rolling_q = [], []
        for i in range(0, 600, 50):
            adaptive.update(stream[0][i:i + 50], stream[1][i:i + 50])
            rolling.update(stream[0][i:i + 50], stream[1][i:i + 50])
            adaptive_q.append(adaptive.q)
            rolling_q.append(rolling.q)

    assert adaptive.recalibration_count > 0
    # A recalibration should adapt faster than the plain rolling window.
    assert not np.allclose(adaptive_q, rolling_q)


@pytest.mark.xfail(strict=True, reason="DriftAwareRecalibrator does not change the quantile (ROADMAP Phase 2)")
def test_drift_aware_recalibration_changes_the_quantile():
    train, calib, stream = shifted_stream()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recal = DriftAwareRecalibrator(
            OnlineConformalUQ(LinearRegression(), alpha=0.1),
            coverage_check_interval=5, target_coverage=0.9,
        ).fit(*train, *calib)
        twin = OnlineConformalUQ(LinearRegression(), alpha=0.1).fit(*train, *calib)
        feed(recal, *stream)
        feed(twin, *stream)

    assert recal.total_recalibrations > 0
    assert recal.uq_method.q != twin.q


def test_network_class_is_treated_as_a_constructor():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 1)

        def forward(self, x):
            return self.fc(x)

    uq = UQ(Net)
    assert uq.model_type == "constructor"
    assert uq.method == "deep_ensemble"
    assert UQ(Net()).model_type == "pytorch"
