"""Shared helpers for split and online conformal prediction."""

import warnings

import numpy as np

# Guards the rank computations against float error, e.g. 0.9 * 10 = 9.000000000000002.
_EPS = 1e-9


def min_calibration_size(level):
    """Smallest n for which the ceil((1 - level)(n + 1))-th score exists."""
    return int(np.ceil(1 / level - _EPS)) - 1


def _warn_small_calibration(n, level):
    needed = min_calibration_size(level)
    warnings.warn(
        f"Calibration set of size {n} is too small for this alpha: at least "
        f"{needed} scores are needed. Returning infinite bounds (or the full "
        "label set) to keep the coverage guarantee.",
        UserWarning,
        stacklevel=3,
    )


def upper_quantile(scores, level):
    """The ceil((1 - level)(n + 1))-th smallest score, the split conformal quantile.

    Returns +inf when that rank exceeds the number of scores.
    """
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    k = int(np.ceil((1 - level) * (n + 1) - _EPS))
    if k > n:
        _warn_small_calibration(n, level)
        return np.inf
    return scores[k - 1]


def lower_quantile(scores, level):
    """The floor(level * (n + 1))-th smallest score; -inf when that rank is 0."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    k = int(np.floor(level * (n + 1) + _EPS))
    if k < 1:
        _warn_small_calibration(n, level)
        return -np.inf
    return scores[k - 1]


def model_classes(model, n_columns):
    """Class labels in the column order of model.predict_proba."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        return np.arange(n_columns)
    return np.asarray(classes)


def encode_labels(y, classes):
    """Map labels to their column index in predict_proba output."""
    y = np.asarray(y)
    sorter = np.argsort(classes)
    pos = np.searchsorted(classes, y, sorter=sorter)
    idx = sorter[np.clip(pos, 0, len(classes) - 1)]
    unknown = classes[idx] != y
    if np.any(unknown):
        raise ValueError(
            f"Labels {np.unique(y[unknown]).tolist()} are not among the "
            f"model's classes {classes.tolist()}."
        )
    return idx


def prediction_sets(probas, q, classes):
    """LAC prediction sets {label : 1 - p(label) <= q}, as sets of class labels."""
    included = probas >= 1 - q
    return [set(classes[row].tolist()) for row in included]


def point_labels(pred_sets):
    """The single label of a singleton set, else -1 (the 1.0.x convention)."""
    return [next(iter(s)) if len(s) == 1 else -1 for s in pred_sets]
