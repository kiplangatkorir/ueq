import warnings


class ExperimentalWarning(UserWarning):
    """Warning for APIs whose statistical behaviour has not been validated yet."""


def warn_experimental(name, detail=""):
    message = (
        f"{name} is experimental: its statistical behaviour has not been "
        "validated and it may change or be removed. See docs/ROADMAP.md."
    )
    if detail:
        message = f"{message} {detail}"
    warnings.warn(message, ExperimentalWarning, stacklevel=3)
