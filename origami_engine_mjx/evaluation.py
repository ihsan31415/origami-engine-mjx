from __future__ import annotations

"""Evaluation interfaces — backend-aware fitness functions.

These wrap :class:`~origami_engine_mjx.engine.OrigamiEngine` so callers can
pick a backend once and evaluate individuals or populations.
"""

import numpy as np

from .constants import VIB_AMP_X, VIB_AMP_Z, VIB_FREQ
from .engine import OrigamiEngine, compute_forward_displacement
from .types import BackendName


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
    actuated=None,
    actuation_schedule=None,
    fold_weight: float = 0.0,
    movement_weight: float = 1.0,
    movement_scale: float = 1.0,
):
    """Evaluate one crease configuration and return net forward displacement.

    A non-finite displacement is returned as-is (no finite-score check).
    ``actuated`` / ``actuation_schedule`` control which hinges are driven (see
    :meth:`origami_engine_mjx.OrigamiEngine.run`).  With ``fold_weight > 0``
    the returned number is the weighted blend of movement and hinge-target
    fitness instead of the bare displacement; ``return_details=True`` then also
    reports ``movement``, ``fold`` and ``combined``.
    """
    engine = OrigamiEngine(backend=backend)
    result = engine.run(
        creases,
        duration=duration,
        fold_duration=fold_duration,
        vibrate=vibrate,
        amp_x=amp_x,
        amp_z=amp_z,
        freq_hz=freq_hz,
        actuated=actuated,
        actuation_schedule=actuation_schedule,
    )
    if fold_weight > 0:
        from .fitness import blend_fitness

        breakdown = blend_fitness(
            result,
            trajectory=trajectory,
            movement_weight=movement_weight,
            fold_weight=fold_weight,
            movement_scale=movement_scale,
        )
        displacement = breakdown.combined
    else:
        breakdown = None
        displacement = compute_forward_displacement(result, trajectory=trajectory)
    if return_details:
        details = {
            "displacement": float(displacement),
            "initial_centroid": np.asarray(result.initial_centroid, dtype=float),
            "final_centroid": np.asarray(result.final_centroid, dtype=float),
            "duration": float(duration),
            "backend": result.backend,
            "n_faces": int(result.pattern.n_faces),
            "n_loops": len(result.pattern.loop_edges or []),
        }
        if breakdown is not None:
            details.update(
                {
                    "movement": breakdown.movement,
                    "fold": breakdown.fold,
                    "combined": breakdown.combined,
                    "fold_report": breakdown.fold_report,
                }
            )
        return details
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
    actuated=None,
    actuation_schedule=None,
    fold_weight: float = 0.0,
    movement_weight: float = 1.0,
    movement_scale: float = 1.0,
) -> list[float]:
    """Evaluate a population of crease patterns using the given backend.

    ``trajectory`` is validated but ignored by the built-in backends (scores
    are always +x) unless ``fold_weight > 0`` asks for the blended fitness,
    which recomputes movement along ``trajectory`` per individual.
    ``batch_size`` is currently ignored.  With ``error_policy="raise"`` any
    non-finite score raises ``RuntimeError``; any other value returns the raw
    scores.
    """
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
        actuated=actuated,
        actuation_schedule=actuation_schedule,
        fold_weight=fold_weight,
        movement_weight=movement_weight,
        movement_scale=movement_scale,
    )
    scores = pop_result.scores
    if error_policy == "raise":
        for i, s in enumerate(scores):
            if not np.isfinite(s):
                raise RuntimeError(f"individual {i} produced non-finite score: {s}")
    return list(scores)


__all__ = ["evaluate_individual", "evaluate_population"]
