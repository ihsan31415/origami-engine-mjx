import pytest

collect_ignore_glob = []


def pytest_collection_modifyitems(config, items):
    """Deselect 'slow' tests unless explicitly requested via -m 'slow'."""
    if config.getoption("-m") and "slow" in config.getoption("-m"):
        return
    skip_slow = pytest.mark.skip(reason="slow MJX-CPU test; run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
