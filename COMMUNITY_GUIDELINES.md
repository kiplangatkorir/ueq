# Community Guidelines for UEQ

Welcome to the UEQ community! These guidelines help us maintain a productive, respectful, and inclusive environment for everyone contributing to uncertainty quantification in machine learning.

## Our Principles

### 1. Rigor with Kindness

> **Be rigorous with ideas, kind with people.**

- Challenge technical arguments, not people
- Provide constructive feedback with specific suggestions
- Acknowledge the effort in all contributions

### 2. Collaborative Science

- UQ is a developing field - we're learning together
- Share knowledge generously
- Credit others' work appropriately
- Engage with curiosity, not criticism

### 3. Practical Impact

- Focus on methods that work in production
- Balance theoretical rigor with practical usability
- Document limitations honestly
- Prioritize reproducibility

## Communication

### GitHub Issues

**Good Issues:**
- Clear problem statement
- Reproducible example when reporting bugs
- Motivation for feature requests
- Acknowledgment of existing work

**Example:**

```markdown
## Problem
Current conformal prediction doesn't handle time-series data well.

## Motivation
Time-series forecasting is common in production, but standard CP 
assumes i.i.d. data, leading to coverage violations.

## Proposed Solution
Implement rolling-window conformal prediction with temporal splits.

## Related Work
- [Paper reference]
- Similar to issue #X but focuses on...
```

### Pull Requests

**Before Opening a PR:**
1. Check if there's an existing issue
2. For large changes, discuss in an issue first
3. Follow the contribution templates in `templates/`

**PR Best Practices:**
- One PR per logical change
- Reference related issues
- Include tests for new functionality
- Update documentation
- Add example usage in docstrings

**PR Description Template:**

```markdown
## Summary
Brief description of changes.

## Related Issue
Fixes #123

## Changes
- Added X feature
- Fixed Y bug
- Updated Z documentation

## Testing
- [ ] Tests added and passing
- [ ] Coverage maintained/improved
- [ ] Example notebook included (for new features)

## Breaking Changes
None / List any API changes
```

### Discussions

Use GitHub Discussions for:
- Design proposals
- Research questions
- Implementation advice
- General UQ discussions

**Discussion Guidelines:**
- Search existing discussions first
- Use descriptive titles
- Tag appropriately (research, implementation, etc.)
- Share references and code examples

## Code Standards

### Statistical Correctness

**This is non-negotiable.** UQ methods must be statistically sound.

Before implementing:
1. Understand the mathematical foundations
2. Read the original papers
3. Validate on synthetic data with known properties
4. Document assumptions and limitations clearly

### Code Quality

**Style:**
- Follow PEP 8
- Use `black` for formatting: `black ueq/`
- Use `pylint` for linting
- Write docstrings for all public APIs

**Documentation:**
- Clear mathematical notation
- Explain **what** the method does
- Explain **when** to use it
- Explain **what** the limitations are
- Include references to papers

**Example:**

```python
def evidential_regression(model, alpha=0.05):
    """
    Evidential regression using Normal-Inverse-Gamma distribution.
    
    Produces uncertainty estimates in a single forward pass by predicting
    the parameters of a NIG distribution over the target.
    
    Parameters
    ----------
    model : torch.nn.Module
        Neural network outputting (mu, v, alpha, beta) per sample.
    alpha : float, default=0.05
        Significance level for prediction intervals.
    
    Returns
    -------
    predictions : ndarray
        Point predictions (posterior mean).
    intervals : list of tuples
        Prediction intervals at (1-alpha) confidence.
    
    Mathematical Background
    -----------------------
    Models the posterior p(y|x) as NIG(μ, v, α, β) where:
    - μ: predicted mean
    - v: data uncertainty (aleatoric)
    - α, β: epistemic uncertainty (evidence strength)
    
    References
    ----------
    [1] Amini et al. "Deep Evidential Regression" NeurIPS 2020
    
    Limitations
    -----------
    - Requires model architecture changes (4 outputs instead of 1)
    - Assumes Gaussian likelihood
    - May underestimate uncertainty without proper regularization
    
    Examples
    --------
    >>> model = EvidentialNN(input_dim=10)
    >>> preds, intervals = evidential_regression(model)
    """
```

### Testing Requirements

All new code must include tests:

**Minimum Tests:**
1. Basic functionality test
2. Error handling test
3. Reproducibility test (with seed)

**For UQ Methods:**
4. Coverage test on synthetic data
5. Calibration test
6. Comparison with baseline method

See `TESTING_GUIDE.md` for details.

## Review Process

### For Contributors

**What to Expect:**
- Maintainers will review within 1 week
- Feedback may request changes
- Multiple review iterations are normal
- Not all PRs will be merged

**During Review:**
- Respond to feedback constructively
- Ask questions if feedback is unclear
- Make requested changes promptly
- Push updates to the same branch

### For Reviewers

**Review Focus:**
1. **Correctness**: Is the math/stats correct?
2. **Quality**: Is the code well-written?
3. **Testing**: Are tests adequate?
4. **Documentation**: Is it clear and complete?

**Review Tone:**
- Be specific about issues
- Suggest improvements, don't just criticize
- Acknowledge good work
- Explain the "why" behind requests

**Example Feedback:**

```markdown
❌ Bad: "This is wrong."
✅ Good: "The quantile calculation here doesn't account for the finite 
         sample correction. See Vovk et al. 2005 equation 3 for the 
         correct formula: k = ceil((1-α)(n+1))"

❌ Bad: "Needs tests."
✅ Good: "Could you add a test that verifies coverage on heteroscedastic 
         data? The current tests only use homoscedastic noise, which 
         might not catch issues with adaptive uncertainty."
```

## Recognition

Contributors are valued and acknowledged:

**Attribution:**
- Listed in release notes
- Added to CONTRIBUTORS file
- Mentioned in related documentation

**Growing Responsibility:**
- Consistent high-quality contributors may be invited as maintainers
- Maintainers have merge rights and help guide the project

## Conflict Resolution

If disagreements arise:

1. **Assume Good Intent**: Everyone wants UEQ to succeed
2. **Focus on Ideas**: Debate the technical merits, not personalities
3. **Seek Common Ground**: Find mutually acceptable solutions
4. **Escalate if Needed**: Tag a maintainer to mediate

## Unacceptable Behavior

We do not tolerate:

- Harassment or discrimination
- Plagiarism or uncredited work
- Deliberately incorrect statistical claims
- Bad faith arguments or trolling
- Spam or off-topic content

**Consequences:**
- Warning for first offense
- Temporary ban for repeated violations
- Permanent ban for severe violations

Report violations to project maintainers.

## Research Contributions

UEQ bridges research and production. Research contributions are welcome with these expectations:

**Research PRs Should:**
- Be clearly marked as experimental/research
- Include mathematical notation and derivations
- Reference original papers
- Provide synthetic validation
- Document limitations extensively

**Research PRs May:**
- Have lower test coverage (but still need basic tests)
- Include prototype implementations
- Focus on specific use cases
- Evolve over multiple iterations

**Label Your PR:**
```markdown
[RESEARCH] Evidential Classification with Dirichlet Priors

This is a research implementation of...
Limitations: ...
Future work: ...
```

## Questions?

- **Technical questions**: Open an issue or discussion
- **Community questions**: Tag @maintainers in discussions
- **Private concerns**: Email project maintainers

## Resources

- [CONTRIBUTING.md](CONTRIBUTING.md) - Technical contribution guide
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - How to write tests
- [templates/](templates/) - Contribution templates
- [GitHub Discussions](https://github.com/kiplangatkorir/ueq/discussions)

---

**Thank you for helping make UQ practical, rigorous, and accessible!**

*These guidelines may evolve as the community grows. Last updated: January 2026*
