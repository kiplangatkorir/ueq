import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

from ueq import UQ
from ueq.methods.conformal import ConformalUQ


def make_labelled_data(labels, n=3000, seed=0):
    """Three-way split of a noisy multiclass problem using the given labels."""
    rng = np.random.default_rng(seed)
    k = len(labels)
    centers = rng.normal(scale=1.5, size=(k, 4))
    idx = rng.integers(k, size=n)
    X = centers[idx] + rng.normal(size=(n, 4))
    y = np.asarray(labels)[idx]
    a, b = n // 3, 2 * n // 3
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def set_coverage(y, sets):
    return np.mean([label in s for label, s in zip(y.tolist(), sets)])


def test_uq_routes_classifier_to_conformal_classification():
    uq = UQ(LogisticRegression())
    assert uq.method == "conformal"
    assert uq.uq_model.task_type == "classification"
    assert uq.uq_model.nonconformity == "inverse_probability"


@pytest.mark.parametrize("model", [SVC(), RidgeClassifier()])
def test_classifier_without_predict_proba_raises(model):
    with pytest.raises(ValueError, match="predict_proba"):
        UQ(model)


def test_conformal_classification_defaults_to_lac():
    uq = ConformalUQ(LogisticRegression(), task_type="classification")
    assert uq.nonconformity == "inverse_probability"


@pytest.mark.parametrize(
    "labels",
    [[0, 1, 2], [1, 2, 3], [-1, 1], ["cat", "dog", "fox"]],
    ids=["0..K-1", "1..K", "-1/+1", "strings"],
)
def test_prediction_sets_contain_labels(labels):
    train, calib, test = make_labelled_data(labels)
    uq = UQ(LogisticRegression(), alpha=0.1)
    uq.fit(*train, *calib)
    point, sets = uq.predict(test[0], return_interval=True)

    assert set().union(*sets) <= set(labels)
    assert set_coverage(test[1], sets) >= 0.87
    singletons = [next(iter(s)) for s in sets if len(s) == 1]
    assert all(p in labels for p in point if p != -1)
    assert len(singletons) > 0


def test_unknown_calibration_label_raises():
    train, calib, _ = make_labelled_data(["a", "b"])
    uq = ConformalUQ(LogisticRegression(), task_type="classification")
    with pytest.raises(ValueError, match="not among the model's classes"):
        uq.fit(train[0], train[1], calib[0], np.where(calib[1] == "a", "a", "z"))


@pytest.mark.statistical
def test_lac_marginal_coverage_over_repeated_trials():
    alpha, n_calib, n_test, trials = 0.1, 500, 500, 40
    coverages = []
    for seed in range(trials):
        train, calib, test = make_labelled_data(["a", "b", "c"], n=3 * n_calib, seed=seed)
        uq = ConformalUQ(LogisticRegression(), alpha=alpha, task_type="classification")
        uq.fit(*train, *calib)
        coverages.append(set_coverage(test[1], uq.predict(test[0])))

    # Mean coverage of split conformal is ceil((1-alpha)(n+1))/(n+1); per-trial
    # spread combines the Beta calibration term and the binomial test term.
    expected = np.ceil((1 - alpha) * (n_calib + 1)) / (n_calib + 1)
    sd = np.sqrt(alpha * (1 - alpha) * (1 / (n_calib + 2) + 1 / n_test) / trials)
    assert abs(np.mean(coverages) - expected) < 4 * sd
