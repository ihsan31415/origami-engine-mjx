from __future__ import annotations

"""Evaluation interfaces — backend-aware fitness functions.

These wrap :class:`~origami_engine_mjx.engine.OrigamiEngine` so callers can
pick a backend once and evaluate individuals or populations.
"""

import numpy as np

from .constants import VIB_AMP_X, VIB_AMP_Z, VIB_FREQ
from .engine import OrigamiEngine, compute_forward_displacement
from .types import BackendName, PopulationResult, SimulationResult


def evaluate_individual(
    creases,
    duration: float = 8.0,
    trajectory: tuple[float, ...] = (1.0, 0.0, 0.0),
    fold_duration: float = 3.0,
    vibrate: bool = True,
    amp_x: float = VIB_AMP_X,
    amp_z: float = VIB_AMP_Z,
    freq_hz: float = VIB_FREQ,
    backend: BackendName = "jax",
    return_details: bool = False,
):
    """Evaluate one crease configuration and return net forward displacement."""
    engine = OrigamiEngine(backend=backend)
    result = engine.run(
        creases,
        duration=duration,
        fold_duration=fold_duration,
        vibrate=vibrate,
        amp_x=amp_x,
        amp_z=amp_z,
        freq_hz=freq_hz,
    )
    displacement = compute_forward_displacement(result, trajectory=trajectory)
    if return_details:
        return {
            "displacement": float(displacement),
            "initial_centroid": np.asarray(result.initial_centroid, dtype=float),
            "final_centroid": np.asarray(result.final_centroid, dtype=float),
            "duration": float(duration),
            "backend": result.backend,
            "n_faces": int(result.pattern.n_faces),
            "n_loops": len(result.pattern.loop_edges or []),
        }
    return float(displacement)


def evaluate_population(
    individuals,
    duration: float = 8.0,
    trajectory: tuple[float, ...] = (1.0, 0.0, 0.0),
    fold_duration: float = 3.0,
    vibrate: bool = True,
    amp_x: float = VIB_AMP_X,
    amp_z: float = VIB_AMP_Z,
    freq_hz: float = VIB_FREQ,
    backend: BackendName = "jax",
    batch_size: int | None = None,
    error_policy: str = "raise",
) -> list[float] | PopulationResult:
    """Evaluate a population of crease patterns using the given backend."""
    engine = OrigamiEngine(backend=backend)
    pop_result = engine.evaluate_population(
        individuals,
        duration=duration,
        trajectory=trajectory,
        fold_duration=fold_duration,
        vibrate=vibrate,
        amp_x=amp_x,
        amp_z=amp_z,
        freq_hz=freq_hz,
        batch_size=batch_size,
    )
    scores = pop_result.scores
    if error_policy == "raise":
        for i, s in enumerate(scores):
            if not np.isfinite(s):
                raise RuntimeError(f"individual {i} produced non-finite score: {s}")
    return list(scores)


__all__ = ["evaluate_individual", "evaluate_population"]
