# Contributing a New UQ Method

This template helps you add a new uncertainty quantification method to UEQ.

## Step 1: Create Your Method Class

Copy this template to `ueq/methods/your_method_name.py`:

```python
"""
Brief description of the UQ method.

References
----------
[1] Author et al. "Paper Title" Journal/Conference (Year)
"""

import numpy as np


class YourMethodUQ:
    """
    One-line description of your UQ method.
    
    Parameters
    ----------
    model : object
        Base prediction model.
    
    Examples
    --------
    >>> uq = YourMethodUQ(model)
    >>> uq.fit(X_train, y_train)
    >>> predictions, intervals = uq.predict(X_test, return_interval=True)
    """
    
    def __init__(self, model):
        self.base_model = model
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        """Fit the UQ method."""
        self.base_model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X, return_interval=False):
        """Predict with uncertainty."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted yet")
        predictions = self.base_model.predict(X)
        if return_interval:
            intervals = []  # compute intervals
            return predictions, intervals
        return predictions
```

See full template in this directory for complete documentation guidelines.
