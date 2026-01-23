# Testing Guide for UEQ

This guide explains how to test UEQ and contribute test cases for new features.

## Table of Contents

1. [Installation for Testing](#installation-for-testing)
2. [Running Tests](#running-tests)
3. [Writing Tests](#writing-tests)
4. [Testing UQ Methods](#testing-uq-methods)
5. [Testing Benchmarks](#testing-benchmarks)
6. [Continuous Integration](#continuous-integration)

## Installation for Testing

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/kiplangatkorir/ueq.git
cd ueq

# Install in development mode with test dependencies
pip install -e ".[dev]"
```

### Verify Installation

```python
import ueq
print(f"UEQ version: {ueq.__version__}")

# Try a quick example
from ueq import make_synthetic_regression
X, y, meta = make_synthetic_regression(n_samples=100, seed=42)
print(f"Generated dataset: X shape={X.shape}, y shape={y.shape}")
```

## Running Tests

### Run All Tests

```bash
# Run the full test suite
pytest tests/

# Run with coverage report
pytest tests/ --cov=ueq --cov-report=html
```

### Run Specific Tests

```bash
# Test a specific module
pytest tests/test_conformal.py

# Test a specific function
pytest tests/test_conformal.py::test_conformal_regression

# Run tests matching a pattern
pytest -k "conformal"
```

### Verbose Output

```bash
# See detailed output
pytest tests/ -v

# See print statements
pytest tests/ -s
```

## Writing Tests

### Test Structure

All tests should follow this structure:

```python
import numpy as np
import pytest
from ueq.methods.your_method import YourMethod

def test_your_method_basic():
    """Test basic functionality of your method."""
    # 1. Setup: Create test data
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    # 2. Execute: Run the method
    method = YourMethod()
    method.fit(X[:80], y[:80])
    predictions, intervals = method.predict(X[80:], return_interval=True)
    
    # 3. Assert: Verify results
    assert len(predictions) == 20
    assert len(intervals) == 20
    
    # 4. Verify properties
    for lower, upper in intervals:
        assert lower <= upper  # Intervals should be valid

def test_your_method_coverage():
    """Test that coverage is near nominal level."""
    from ueq.benchmarks import make_synthetic_regression
    
    # Use synthetic benchmark with known uncertainty
    X, y, meta = make_synthetic_regression(n_samples=500, seed=42)
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]
    
    method = YourMethod(alpha=0.1)  # 90% coverage target
    method.fit(X_train, y_train)
    predictions, intervals = method.predict(X_test, return_interval=True)
    
    # Compute empirical coverage
    covered = sum(1 for y, (l, u) in zip(y_test, intervals) if l <= y <= u)
    empirical_coverage = covered / len(y_test)
    
    # Coverage should be close to 0.9 (allow some deviation)
    assert 0.85 <= empirical_coverage <= 0.95

def test_your_method_errors():
    """Test error handling."""
    method = YourMethod()
    
    # Should raise error if predict called before fit
    with pytest.raises(RuntimeError):
        method.predict(np.random.randn(10, 5))
    
    # Should raise error for invalid parameters
    with pytest.raises(ValueError):
        YourMethod(alpha=-0.1)  # Invalid alpha
```

### Testing Guidelines

1. **Test the Happy Path**: Normal usage should work
2. **Test Edge Cases**: Empty arrays, single samples, etc.
3. **Test Error Conditions**: Invalid inputs should raise appropriate errors
4. **Test Statistical Properties**: Coverage, calibration, etc.
5. **Use Synthetic Benchmarks**: For reproducible tests with known properties

### Fixtures for Common Test Data

```python
import pytest
import numpy as np

@pytest.fixture
def regression_data():
    """Generate standard regression test data."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = X @ np.random.randn(10) + np.random.randn(200) * 0.5
    return X, y

@pytest.fixture
def classification_data():
    """Generate standard classification test data."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=10, 
                               n_classes=3, random_state=42)
    return X, y

# Use in tests
def test_with_fixture(regression_data):
    X, y = regression_data
    # Your test code here
```

## Testing UQ Methods

### Coverage Tests

Every UQ method should maintain nominal coverage:

```python
def test_method_coverage():
    """Verify method achieves nominal coverage."""
    from ueq.benchmarks import make_synthetic_regression
    
    # Generate data with known uncertainty
    X, y, meta = make_synthetic_regression(
        n_samples=1000,
        noise="heteroscedastic",
        seed=42
    )
    
    # Split data
    X_train = X[:600]
    y_train = y[:600]
    X_calib = X[600:800]
    y_calib = y[600:800]
    X_test = X[800:]
    y_test = y[800:]
    
    # Test at different alpha levels
    for alpha in [0.05, 0.1, 0.2]:
        method = YourMethod(alpha=alpha)
        method.fit(X_train, y_train, X_calib, y_calib)
        
        preds, intervals = method.predict(X_test, return_interval=True)
        
        covered = sum(1 for y, (l, u) in zip(y_test, intervals) if l <= y <= u)
        coverage = covered / len(y_test)
        
        target = 1 - alpha
        # Allow 5% tolerance
        assert target - 0.05 <= coverage <= target + 0.05
```

### Drift Tests

Test that methods handle drift appropriately:

```python
def test_method_under_drift():
    """Test method behavior under distribution drift."""
    from ueq.benchmarks import make_concept_drift_data
    
    X, y, meta = make_concept_drift_data(
        n_samples=1000,
        drift_type="gradual",
        seed=42
    )
    
    # Train on first half (before drift)
    method = YourMethod()
    method.fit(X[:500], y[:500])
    
    # Test on second half (after drift)
    preds, intervals = method.predict(X[500:], return_interval=True)
    
    # Intervals should be wider or method should detect drift
    # (Specific assertions depend on method type)
    assert len(intervals) == 500
```

## Testing Benchmarks

### Benchmark Properties

Test that benchmarks have expected properties:

```python
def test_benchmark_reproducibility():
    """Benchmark should be reproducible with same seed."""
    from ueq.benchmarks import make_synthetic_regression
    
    X1, y1, meta1 = make_synthetic_regression(seed=42)
    X2, y2, meta2 = make_synthetic_regression(seed=42)
    
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(meta1['y_true'], meta2['y_true'])

def test_benchmark_metadata():
    """Benchmark should include required metadata."""
    from ueq.benchmarks import make_synthetic_regression
    
    X, y, meta = make_synthetic_regression()
    
    # Check required metadata
    assert 'y_true' in meta
    assert 'noise_std' in meta
    assert len(meta['y_true']) == len(y)
    assert len(meta['noise_std']) == len(y)
    
    # Verify noise is applied correctly
    noise = y - meta['y_true']
    # Empirical std should be close to mean of noise_std
    assert abs(noise.std() - meta['noise_std'].mean()) < 0.5
```

## Continuous Integration

### GitHub Actions

Tests run automatically on every push and pull request:

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest tests/ --cov=ueq
```

### Pre-commit Checks

Run tests before committing:

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Hooks will run automatically on git commit
# Or run manually:
pre-commit run --all-files
```

## Test Coverage Goals

- **Minimum Coverage**: 80% for new code
- **Core Modules**: 90%+ coverage for `ueq/methods/` and `ueq/utils/metrics.py`
- **Benchmarks**: 100% coverage for synthetic data generators

### Checking Coverage

```bash
# Generate coverage report
pytest tests/ --cov=ueq --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Common Testing Pitfalls

### 1. Random Seeds

Always set random seeds for reproducibility:

```python
# Good
np.random.seed(42)
X = np.random.randn(100, 5)

# Bad - test might fail randomly
X = np.random.randn(100, 5)
```

### 2. Floating Point Comparisons

Use appropriate tolerances:

```python
# Good
np.testing.assert_allclose(result, expected, rtol=1e-5)

# Bad - might fail due to floating point errors
assert result == expected
```

### 3. Coverage Tests

Allow reasonable tolerance for statistical tests:

```python
# Good - allows for statistical variation
assert 0.85 <= coverage <= 0.95  # Target: 0.9

# Bad - too strict, will fail randomly
assert coverage == 0.9
```

## Getting Help

If you encounter issues:

1. Check existing tests for examples
2. Review the contribution templates in `templates/`
3. Ask in GitHub Discussions
4. Open an issue with the `testing` label

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [numpy testing utilities](https://numpy.org/doc/stable/reference/routines.testing.html)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
