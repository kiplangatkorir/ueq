def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "statistical: repeated-trial coverage tests (slower; they check validity, not shapes)",
    )
