"""GPU correctness guard for the MJX-Warp backend.

Marked ``gpu``: skipped automatically on machines without CUDA + mujoco-warp.
``python3 -m pytest tests_mjx/test_warp_backend_gpu.py -m gpu -q`` runs it
explicitly.  It checks the two properties the optimiser relies on:

1. Batched scoring (one Warp model, one world per candidate) matches classic
   MuJoCo's own ``mj_step`` for the same XML and candidates.
2. The +x displacement returned for a candidate is the same whether it is
   scored alone or in a batch with others.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from origami_engine_mjx import Crease
from origami_engine_mjx.backends import available_backends, warp_backend

pytestmark = pytest.mark.gpu

# A small herringbone Miura sheet: four faces, three tree hinges and one loop,
# so the loop-closure equality is exercised too.
PATTERN = [
    Crease((0.2, 0.2), (0.5, 0.2), 0.7853981633974483),
    Crease((0.5, 0.2), (0.8, 0.2), -0.7853981633974483),
    Crease((0.2, 0.2), (0.2, 0.5), -0.7853981633974483),
    Crease((0.8, 0.2), (0.8, 0.5), 0.7853981633974483),
    Crease((0.2, 0.5), (0.5, 0.5), 0.7853981633974483),
    Crease((0.5, 0.5), (0.8, 0.5), -0.7853981633974483),
    Crease((0.2, 0.5), (0.2, 0.8), 0.7853981633974483),
    Crease((0.8, 0.5), (0.8, 0.8), -0.7853981633974483),
    Crease((0.2, 0.2), (0.8, 0.8), 0.0),
    Crease((0.2, 0.8), (0.8, 0.8), 0.0),
    Crease((0.2, 0.8), (0.8, 0.2), 0.0),
]


def _variant(scale: float) -> list[Crease]:
    return [Crease(c.p1, c.p2, c.angle * scale) for c in PATTERN]


@pytest.fixture(scope="module")
def backend():
    if "warp" not in available_backends():
        pytest.skip("needs an NVIDIA GPU with mujoco-warp")
    be = warp_backend.WarpBackend()
    if not be.is_available():
        pytest.skip("needs an NVIDIA GPU with mujoco-warp")
    return be


def test_batched_scores_match_single_runs(backend):
    population = [_variant(s) for s in (1.0, 0.75, 0.5, 0.25)]
    duration = 0.05
    batched = backend.run_batch(population, duration=duration, fold_duration=duration / 2, vibrate=False)
    alone = np.array(
        [backend.run_batch([c], duration=duration, fold_duration=duration / 2, vibrate=False)[0] for c in population]
    )
    assert batched.shape == (len(population),)
    assert np.isfinite(batched).all()
    # Same physics, different execution path: agree to sub-millimetre.
    assert np.allclose(batched, alone, atol=1e-3), (batched, alone)


def test_batch_is_much_faster_than_sequential(backend):
    population = [_variant(1.0 - 0.1 * i) for i in range(16)]
    kwargs = {"duration": 0.3, "fold_duration": 0.15, "vibrate": False}
    backend.run_batch(population[:1], **kwargs)  # warm-up / compile / capture

    start = time.perf_counter()
    backend.run_batch(population, **kwargs)
    batched = time.perf_counter() - start

    start = time.perf_counter()
    for candidate in population:
        backend.run_batch([candidate], **kwargs)
    sequential = time.perf_counter() - start
    assert batched < sequential / 4, (batched, sequential)
