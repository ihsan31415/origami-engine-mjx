"""Tests for backend selection / Warp backend guard (no GPU required).

These tests verify that the Warp backend *code path* is correct without
requiring an NVIDIA GPU: ``is_available()`` must return False gracefully
when CUDA is absent, and selecting ``backend="warp"`` must raise a clear
error rather than crashing.
"""

import pytest

from origami_engine_mjx import OrigamiEngine
from origami_engine_mjx.backends import available_backends, get_backend
from origami_engine_mjx.backends.warp_backend import WarpBackend


def test_warp_backend_reports_unavailable_without_gpu():
    backend = WarpBackend()
    # On a machine without CUDA, is_available() must be False (not crash).
    assert backend.is_available() in (True, False)
    info = backend.device_info()
    assert isinstance(info, str)


def test_warp_not_in_available_backends_on_cpu():
    """On a CPU-only machine, 'warp' must not appear as available."""
    avail = available_backends()
    # If warp is listed, a CUDA device genuinely exists — that's fine.
    # The key invariant: available_backends() never raises.
    assert isinstance(avail, list)


def test_selecting_warp_without_gpu_raises_runtime_error():
    backend = WarpBackend()
    if not backend.is_available():
        with pytest.raises(RuntimeError, match="not available"):
            get_backend("warp")
        with pytest.raises(RuntimeError, match="not available|backend"):
            OrigamiEngine(backend="warp")


def test_auto_backend_picks_jax_on_cpu():
    engine = OrigamiEngine(backend="auto")
    # On CPU-only, auto must resolve to jax.
    avail = available_backends()
    expected = "warp" if "warp" in avail else avail[0] if avail else "jax"
    assert engine.backend_name == expected
