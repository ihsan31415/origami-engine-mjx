"""Tests for the MJX-JAX backend (CPU-executable).

Uses a minimal 2-face pattern because MJX-JAX on CPU is slow for
mesh-contact models (~1.4 s/step even for tiny patterns; the 16-face
reference model costs ~70 s/step on CPU). GPU execution is the intended
target for full-scale models.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from origami_engine_mjx import Crease, OrigamiEngine
from origami_engine_mjx.backends import available_backends, get_backend


@pytest.fixture(scope="module")
def tiny_creases():
    """Minimal valid pattern: one mountain fold -> 2 faces, 1 hinge."""
    return [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]


def test_jax_backend_available_on_cpu():
    backends = available_backends()
    assert "jax" in backends


def test_engine_creation_jax():
    engine = OrigamiEngine(backend="jax")
    assert engine.backend_name == "jax"
    info = engine.device_info()
    assert "jax" in info.lower()


@pytest.mark.slow
def test_single_simulation_runs(tiny_creases):
    """Short simulation must produce finite centroids."""
    engine = OrigamiEngine(backend="jax")
    result = engine.run(tiny_creases, duration=0.003, fold_duration=0.002, vibrate=False)
    assert result.backend == "jax"
    assert np.isfinite(result.initial_centroid).all()
    assert np.isfinite(result.final_centroid).all()
    assert result.pattern.n_faces == 2


@pytest.mark.slow
def test_evaluate_returns_finite_float(tiny_creases):
    engine = OrigamiEngine(backend="jax")
    score = engine.evaluate(tiny_creases, duration=0.003, fold_duration=0.002, vibrate=False)
    assert isinstance(score, float)
    assert np.isfinite(score)


@pytest.mark.slow
def test_population_evaluation(tiny_creases):
    engine = OrigamiEngine(backend="jax")
    pop = [tiny_creases, tiny_creases]
    result = engine.evaluate_population(pop, duration=0.003, fold_duration=0.002, vibrate=False)
    assert result.n_individuals == 2
    assert len(result.scores) == 2
    assert np.isfinite(result.scores).all()
    assert result.scores[0] == pytest.approx(result.scores[1])


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("nonexistent")
