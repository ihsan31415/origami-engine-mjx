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
    # GPU tests need CUDA + mujoco-warp; they skip cleanly on CPU-only machines.
    if config.getoption("-m") and "gpu" in config.getoption("-m"):
        return
    from origami_engine_mjx.backends import warp_backend

    skip_gpu = pytest.mark.skip(reason="needs an NVIDIA GPU with mujoco-warp")
    try:
        gpu_ready = warp_backend.WarpBackend().is_available()
    except Exception:  # noqa: BLE001 - no backend => nothing GPU-testable
        gpu_ready = False
    for item in items:
        if "gpu" in item.keywords and not gpu_ready:
            item.add_marker(skip_gpu)
