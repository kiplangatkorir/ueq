# Contributing to UEQ

First of all, thank you for your interest in contributing to **UEQ (Uncertainty Estimation & Quantification)**.

UEQ is an open-source project for **statistically sound, practical uncertainty quantification for machine learning**: prediction intervals and sets, and checks of whether they hold. Contributions are welcome from ML engineers, researchers, and motivated students.

This document explains **how to get started, what kinds of contributions we are looking for, and how to work effectively with the project**. The project's direction is set out in [docs/ROADMAP.md](docs/ROADMAP.md).

## Contribution Templates

To make contributing easier, we provide templates for common contribution types:

- **New UQ Methods**: See `templates/uq_method/TEMPLATE.md`
- **New Benchmarks**: See `templates/benchmark/TEMPLATE.md`
- **New Metrics**: See `templates/metric/TEMPLATE.md`

These templates include:
- Code structure and best practices
- Documentation requirements
- Testing guidelines
- Examples and common patterns

A PR built from the method or metric templates must also pass the [statistical review gate](#statistical-review-gate) below.

## What This Is (and Is Not)

**UEQ contributions are:**

* Open-source and public
* Self-directed (no interviews, no applications)
* Focused on real ML uncertainty problems
* Reviewed for correctness and clarity

**UEQ contributions are NOT:**

* A paid job (at least for now)
* A coursework dumping ground
* A resume submission process

If this works for you, you are very welcome here.

## How to Start (Recommended Path)

1. **Star the repository** (helps visibility)
2. **Read the README** to understand UEQ’s goals
3. **Browse the Issues**
4. Pick one of:

   * `good-first-issue`
   * `core`
   * `research`
5. Comment on the issue to signal you are working on it
6. Open a Pull Request when ready

That’s it — no permission required.

## Types of Contributions We Value

### 1. Core Features

* Correctness fixes and validity tests for the existing methods
* Conformal prediction (split conformal regression and prediction sets)
* Coverage evaluation and monitoring of intervals and sets, including ones produced by other libraries
* Evaluation metrics

### 2. Research & Experimental Work

* Evaluating coverage per segment and over time, with delayed labels
* Recalibration that is shown to restore coverage

UEQ is not adding new UQ method families (evidential, Bayesian neural networks, Laplace, structured outputs, new nonconformity scores). Other libraries already implement them well; see section 9 of [docs/ROADMAP.md](docs/ROADMAP.md).

Research contributions **do not need to be perfect**, but they must be:

* Clearly documented
* Empirically evaluated
* Honest about limitations

Anything that lands in the statistical code paths must also pass the [statistical review gate](#statistical-review-gate).

### 3. Benchmarks & Datasets

* Synthetic datasets with known uncertainty
* Real-world benchmarks
* Reproducible evaluation protocols

### 4. Documentation & Examples

* Tutorials
* Example notebooks
* Visualization utilities

## Issue Labels Explained

* `good-first-issue` – Well-scoped, beginner-friendly
* `core` – Planned core work (see docs/ROADMAP.md)
* `research` – Exploratory/experimental
* `benchmark` – Evaluation & datasets
* `architecture` – Design & extensibility
* `rfc` – Requires discussion before implementation

Please ensure that you respect the intent of each label.

## Coding Guidelines

* Follow existing project structure
* Write clear, readable Python
* Prefer explicit over clever
* Add docstrings for public APIs
* Add tests for every change in behaviour
* No bare `except:` and no `print()` in library code under `ueq/`. Catch specific exceptions, and report to the user with `warnings.warn` or `logging`. CI enforces this with `ruff check ueq --select E722,T201`.

Statistical correctness > performance > convenience.

## Statistical Review Gate

This gate applies to any PR that touches `ueq/methods/`, `ueq/utils/metrics.py` or `ueq/diagnostics.py`, or any other code that computes intervals, prediction sets, coverage or calibration. These paths are listed in `.github/CODEOWNERS`. Passing CI is necessary, but not sufficient.

Such a PR needs all of the following.

1. **A repeated-trial coverage test.** Run the method on freshly generated data over many seeds (at least 50 in the per-PR suite) and compare the mean empirical coverage with the target `1 - alpha`. Fix the seeds so the test is deterministic.
2. **A tolerance derived from the sampling distribution, not picked by hand.**
   * For split conformal with `n` calibration points, coverage given the calibration set follows Beta(k, n + 1 - k) with k = ceil((1 - alpha)(n + 1)). Empirical coverage on `m` test points is then beta-binomial, and the mean over `T` independent trials is approximately normal with that variance divided by `T`. The helper below computes the band.
   * For a method without a finite-sample guarantee (for example a rolling window after a shift), state the property being tested and use a binomial interval (such as Clopper-Pearson) for the estimated coverage.
   * Choose the false-failure rate (for example 0.001) and show the computation in the test.
   * Hand-picked ranges such as `0.7 <= coverage <= 1.0` or `0 <= ece <= 1` are not accepted, and neither are tests that only check shapes.
3. **Strict xfail for known-wrong behaviour.** If a PR documents a bug without fixing it, write the test for the correct behaviour and mark it `@pytest.mark.xfail(strict=True, reason="...")`. Do not write a passing test that pins the bug.
4. **Maintainer sign-off.** A maintainer (the code owner) approves the PR after reading the validity test.
5. **One issue per PR.** Do not bundle several statistical changes into one PR.

```python
import numpy as np
from scipy import stats


def split_conformal_coverage_band(n_calib, n_test, alpha, n_trials, false_failure=1e-3):
    """Two-sided band for the mean empirical coverage of split conformal over n_trials."""
    k = int(np.ceil((1 - alpha) * (n_calib + 1)))
    if k > n_calib:
        raise ValueError("Calibration set too small: the interval should be infinite.")
    a, b = k, n_calib + 1 - k
    mean = a / (a + b)                                     # expected coverage
    var_cond = a * b / ((a + b) ** 2 * (a + b + 1))        # coverage given the calibration set
    var_trial = var_cond + (mean * (1 - mean) - var_cond) / n_test   # beta-binomial / n_test
    half = stats.norm.ppf(1 - false_failure / 2) * np.sqrt(var_trial / n_trials)
    return mean - half, mean + half
```

With 200 calibration points, 500 test points, `alpha=0.1` and 50 trials, the band is about 0.889 to 0.912.

Some examples in `TESTING_GUIDE.md` use a single seed and a fixed band (for example 0.85 to 0.95). They predate this gate; where the two differ, this section applies.

### Automated contributors

Coding bots and other automated agents may open **draft** PRs. They may not take an issue from open to closed on their own: a human maintainer reviews the PR, decides whether it resolves the issue, merges it and closes the issue. A bot-authored PR should reference issues with `Refs #N`, not `Closes #N`, so that merging does not close the issue automatically. The review gate above applies to bot PRs in full.

## Pull Request Guidelines

A good PR:

* References an existing issue
* Explains *what* was done and *why*
* Includes tests or validation; statistical changes include the repeated-trial test described above
* Updates documentation if behavior changes

Draft PRs are welcome.

## Review Process

* Maintainers will review for correctness and clarity
* PRs covered by the statistical review gate need a maintainer's approval before merging
* Feedback may be technical and direct
* Iteration is expected — this is normal

The goal is **quality and long-term maintainability**, not speed.

## Community & Conduct

Be respectful, professional, and constructive.

This project follows a simple rule:

> Be rigorous with ideas, kind with people.

Harassment, plagiarism, or bad-faith contributions will not be tolerated.

## Recognition

Contributors will be:

* Credited in release notes
* Listed in the repository
* Acknowledged in documentation where appropriate

Consistent contributors may be invited to become maintainers.

## Questions & Discussion

* Use GitHub Issues for technical discussion
* Open RFC issues for design proposals
* Discord may be created once the community grows

If you are unsure where to start, open an issue and ask.

**Thank you for helping build trustworthy machine learning.**
