import numpy as np
import pytest

from ueq.benchmarks import (
    make_concept_drift_data,
    make_covariate_shift_data,
    make_heteroscedastic_data,
    make_synthetic_regression,
)


def test_synthetic_regression_shapes_and_metadata():
    X, y, meta = make_synthetic_regression(n_samples=200, n_features=4,
                                           noise="heteroscedastic", seed=0)
    assert X.shape == (200, 4)
    assert y.shape == (200,)
    assert meta["y_true"].shape == (200,)
    assert np.all(meta["noise_std"] > 0)
    assert meta["noise_type"] == "heteroscedastic"
    assert meta["shift_point"] is None


@pytest.mark.parametrize("shift", ["covariate", "label_noise", "concept_drift"])
def test_shifts_record_the_shift_point(shift):
    _, _, meta = make_synthetic_regression(n_samples=100, shift=shift, seed=0)
    assert meta["shift_point"] == 50
    assert meta["shift_type"] == shift


def test_covariate_shift_keeps_y_given_x():
    # Same seed: same X draws, same weights. The shifted rows must follow the
    # same noise-free function of X as the unshifted data.
    X0, _, m0 = make_synthetic_regression(n_samples=400, n_features=3, seed=1)
    X1, _, m1 = make_synthetic_regression(n_samples=400, n_features=3,
                                          shift="covariate", shift_strength=2.0, seed=1)
    sp = m1["shift_point"]
    assert np.allclose(X1[sp:, 0], X0[sp:, 0] + 2.0)

    def interaction(X):
        return 0.5 * np.sin(X[:, 0]) * X[:, 1]

    w = np.linalg.lstsq(X0, m0["y_true"] - interaction(X0), rcond=None)[0]
    assert np.allclose(m1["y_true"], X1 @ w + interaction(X1))


@pytest.mark.parametrize(
    "make",
    [lambda: make_synthetic_regression(n_samples=50, seed=3),
     lambda: make_heteroscedastic_data(n_samples=50, seed=3),
     lambda: make_concept_drift_data(n_samples=50, seed=3),
     lambda: make_covariate_shift_data(n_train=50, n_test=20, seed=3)],
    ids=["regression", "heteroscedastic", "concept_drift", "covariate_shift"],
)
def test_generators_are_reproducible_and_leave_global_rng_alone(make):
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    first = make()
    assert np.array_equal(np.random.get_state()[1], before)

    second = make()
    for a, b in zip(first, second):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)


def test_heteroscedastic_noise_grows_with_first_feature():
    X, _, meta = make_heteroscedastic_data(n_samples=500, variance_function="linear", seed=0)
    order = np.argsort(X[:, 0])
    assert np.all(np.diff(meta["noise_std"][order]) >= -1e-12)


def test_covariate_shift_data_shifts_first_feature_only():
    X_train, y_train, X_test, y_test, meta = make_covariate_shift_data(
        n_train=2000, n_test=2000, shift_strength=2.0, seed=0)
    assert abs(X_test[:, 0].mean() - X_train[:, 0].mean() - 2.0) < 0.15
    assert abs(X_test[:, 1].mean() - X_train[:, 1].mean()) < 0.15
    assert y_train.shape == (2000,) and y_test.shape == (2000,)
