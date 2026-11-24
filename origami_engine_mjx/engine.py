from __future__ import annotations

"""Public engine facade: :class:`OrigamiEngine`.

Usage
-----
>>> from origami_engine_mjx import OrigamiEngine
>>> engine = OrigamiEngine(backend="jax")          # CPU or GPU via JAX
>>> engine = OrigamiEngine(backend="warp")         # NVIDIA GPU via Warp
>>> result = engine.run(creases, duration=2.0)
>>> scores = engine.evaluate_population(population, duration=2.0)
"""

import warnings
from typing import Literal

import numpy as np

from .backends import available_backends, get_backend
from .backends.base import SimulationBackend
from .constants import VIB_AMP_X, VIB_AMP_Z, VIB_FREQ
from .types import BackendName, PopulationResult, SimulationResult

BackendChoice = Literal["jax", "warp", "auto"]


def _warn_failed_individual(index: int, exc: BaseException) -> None:
    """Report a per-individual failure on the blended population path.

    The failure still becomes a ``nan`` score (the batch continues); the
    warning keeps it visible instead of silently dropping an individual.
    """
    warnings.warn(
        f"individual {index} failed during blended evaluation: {type(exc).__name__}: {exc}",
        RuntimeWarning,
        stacklevel=3,
    )


class OrigamiEngine:
    """Unified origami simulation engine with pluggable MJX backends.

    Parameters
    ----------
    backend : {"jax", "warp", "auto"}
        ``"jax"`` — MJX-JAX (CPU/GPU/TPU through JAX; differentiable).
        ``"warp"`` — MJX-Warp (NVIDIA CUDA GPU; high-throughput batched).
        ``"auto"`` — pick the first available backend (warp preferred if a CUDA
        GPU is present, otherwise jax).
    """

    def __init__(self, backend: BackendChoice = "jax") -> None:
        if backend == "auto":
            avail = available_backends()
            if not avail:
                raise RuntimeError(
                    "No MJX backend available. Install one with: "
                    "pip install 'origami-engine-mjx[jax]'  or  "
                    "pip install 'origami-engine-mjx[warp]'"
                )
            # Prefer warp if available (implies CUDA GPU), else jax.
            backend = "warp" if "warp" in avail else avail[0]  # type: ignore[assignment]
        self._backend_name: BackendName = backend  # type: ignore[assignment]
        self._backend: SimulationBackend = get_backend(self._backend_name)

    @property
    def backend(self) -> SimulationBackend:
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def device_info(self) -> str:
        return self._backend.device_info()

    # ------------------------------------------------------------------
    # Single simulation
    # ------------------------------------------------------------------

    def run(
        self,
        creases,
        duration: float = 8.0,
        fold_duration: float = 3.0,
        vibrate: bool = True,
        amp_x: float = VIB_AMP_X,
        amp_z: float = VIB_AMP_Z,
        freq_hz: float = VIB_FREQ,
        coll_thick: float | None = None,
        record: bool = False,
        record_every: int = 1,
        actuated=None,
        actuation_schedule=None,
    ) -> SimulationResult:
        """Run a single origami fold simulation.

        Pass ``record=True`` to keep the per-frame state trajectory in
        ``result.extra["trajectory"]`` for rendering
        (:func:`origami_engine_mjx.rendering.render_video`) or movement plots
        (:func:`origami_engine_mjx.rendering.plot_trajectory`).  Recording does
        not change the final state or the score.

        ``actuated`` chooses which hinges are driven — ``None`` (all, the
        default), a bool, or one bool per hinge in ``result.info["hinges"]``
        order.  ``actuation_schedule`` makes that a function of time:
        ``f(t) -> actuated``, evaluated at every physics step, so a hinge can be
        actuated for part of the fold and passive for the rest.  A hinge that is
        not driven is not braked: it keeps its target angle for scoring but
        folds only through the forces its neighbours transmit.
        """
        return self._backend.run(
            creases,
            duration=duration,
            fold_duration=fold_duration,
            vibrate=vibrate,
            amp_x=amp_x,
            amp_z=amp_z,
            freq_hz=freq_hz,
            coll_thick=coll_thick,
            record=record,  # type: ignore[call-arg]  # backend-specific, see SimulationBackend.run
            record_every=record_every,  # type: ignore[call-arg]
            actuated=actuated,  # type: ignore[call-arg]
            actuation_schedule=actuation_schedule,  # type: ignore[call-arg]
        )

    def evaluate(
        self,
        creases,
        duration: float = 8.0,
        trajectory: tuple[float, ...] = (1.0, 0.0, 0.0),
        fold_duration: float = 3.0,
        vibrate: bool = True,
        amp_x: float = VIB_AMP_X,
        amp_z: float = VIB_AMP_Z,
        freq_hz: float = VIB_FREQ,
        return_trajectory: bool = False,
        actuated=None,
        actuation_schedule=None,
        fold_weight: float = 0.0,
        movement_weight: float = 1.0,
        movement_scale: float = 1.0,
    ) -> float | tuple[float, dict]:
        """Evaluate one crease configuration; return forward displacement.

        With ``return_trajectory=True`` returns ``(score, trajectory)`` where
        ``trajectory`` is the recorded per-frame dict from
        ``result.extra["trajectory"]`` (so one simulation serves both scoring
        and rendering).

        With a positive ``fold_weight`` the score is the weighted blend of the
        movement and hinge-target fitness described in
        :mod:`origami_engine_mjx.fitness` (``fold_weight=0``, the default, is
        exactly the movement score).  The weights are relative, so
        ``fold_weight=0.3`` with the default ``movement_weight=1.0`` is a
        70 % / 30 % split.
        """
        result = self.run(
            creases,
            duration=duration,
            fold_duration=fold_duration,
            vibrate=vibrate,
            amp_x=amp_x,
            amp_z=amp_z,
            freq_hz=freq_hz,
            record=return_trajectory,
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
            score = breakdown.combined
        else:
            score = compute_forward_displacement(result, trajectory=trajectory)
        if return_trajectory:
            return score, result.extra["trajectory"]
        return score

    # ------------------------------------------------------------------
    # Population evaluation
    # ------------------------------------------------------------------

    def evaluate_population(
        self,
        population: list,
        duration: float = 8.0,
        trajectory: tuple[float, ...] = (1.0, 0.0, 0.0),
        fold_duration: float = 3.0,
        vibrate: bool = True,
        amp_x: float = VIB_AMP_X,
        amp_z: float = VIB_AMP_Z,
        freq_hz: float = VIB_FREQ,
        batch_size: int | None = None,
        actuated=None,
        actuation_schedule=None,
        fold_weight: float = 0.0,
        movement_weight: float = 1.0,
        movement_scale: float = 1.0,
    ) -> PopulationResult:
        """Evaluate a population of crease patterns.

        ``trajectory`` is validated (it must be non-zero) but is not forwarded
        to the backends, which score the +x centroid displacement.
        ``batch_size`` is forwarded but ignored by both built-in backends.

        ``actuated`` / ``actuation_schedule`` are forwarded to every
        individual's simulation (see :meth:`run`).  With a positive
        ``fold_weight`` each individual is also scored on how closely it folds
        to its target hinge angles, and the per-individual blended score is
        returned in ``scores``; the components are available as
        ``extra["movement_scores"]`` and ``extra["fold_scores"]``.
        """
        direction = np.asarray(trajectory, dtype=float)
        if direction.size == 2:
            direction = np.array([direction[0], direction[1], 0.0])
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("trajectory direction must be non-zero.")
        direction = direction / norm

        blend = fold_weight > 0
        if blend:
            scores, movement_scores, fold_scores = self._evaluate_population_blended(
                population,
                duration=duration,
                trajectory=trajectory,
                fold_duration=fold_duration,
                vibrate=vibrate,
                amp_x=amp_x,
                amp_z=amp_z,
                freq_hz=freq_hz,
                actuated=actuated,
                actuation_schedule=actuation_schedule,
                fold_weight=fold_weight,
                movement_weight=movement_weight,
                movement_scale=movement_scale,
            )
        else:
            scores = self._backend.run_batch(
                population,
                duration=duration,
                fold_duration=fold_duration,
                vibrate=vibrate,
                amp_x=amp_x,
                amp_z=amp_z,
                freq_hz=freq_hz,
                batch_size=batch_size,
                actuated=actuated,  # type: ignore[call-arg]
                actuation_schedule=actuation_schedule,  # type: ignore[call-arg]
            )
            movement_scores = np.asarray(scores, dtype=float)
            fold_scores = np.full(len(population), np.nan)
        extra = {"movement_scores": movement_scores, "fold_scores": fold_scores}
        return PopulationResult(
            scores=np.asarray(scores, dtype=float),
            backend=self._backend_name,
            n_individuals=len(population),
            duration=duration,
            extra=extra,
        )

    def _evaluate_population_blended(
        self,
        population: list,
        *,
        duration: float,
        trajectory: tuple[float, ...],
        fold_duration: float,
        vibrate: bool,
        amp_x: float,
        amp_z: float,
        freq_hz: float,
        actuated,
        actuation_schedule,
        fold_weight: float,
        movement_weight: float,
        movement_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-individual blend of movement and hinge-target scores.

        Runs one simulation per individual (the backends' ``run_batch`` only
        returns movement), so movement is recomputed along ``trajectory``
        rather than read from the +x fast path.  Failures yield ``nan``.
        """
        n = len(population)
        movement = np.full(n, np.nan)
        fold = np.full(n, np.nan)
        from .fitness import FitnessWeights, blend_fitness, blend_scores

        for i, creases in enumerate(population):
            try:
                result = self.run(
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
                breakdown = blend_fitness(
                    result,
                    trajectory=trajectory,
                    movement_weight=movement_weight,
                    fold_weight=fold_weight,
                    movement_scale=movement_scale,
                )
            except Exception as exc:  # noqa: BLE001 - one bad individual must not kill the batch
                _warn_failed_individual(i, exc)
                continue
            movement[i] = breakdown.movement
            fold[i] = breakdown.fold
        combined = np.empty(n, dtype=float)
        weights = FitnessWeights(movement_weight, fold_weight, movement_scale)
        for i in range(n):
            combined[i] = blend_scores(movement[i], fold[i], weights) if np.isfinite(movement[i]) else np.nan
        return combined, movement, fold


def compute_forward_displacement(
    sim_result: SimulationResult,
    trajectory: tuple[float, ...] = (1.0, 0.0, 0.0),
) -> float:
    """Project centroid displacement onto the trajectory direction.

    A 2-element ``trajectory`` is used as-is (no normalization); a 3-element
    ``trajectory`` is normalized before the dot product.
    """
    direction = np.asarray(trajectory, dtype=float)
    initial = np.asarray(sim_result.initial_centroid, dtype=float)
    final = np.asarray(sim_result.final_centroid, dtype=float)
    delta = final - initial
    if direction.size == 2:
        return float(np.dot(delta[:2], direction))
    delta_3d = np.zeros(3)
    delta_3d[:2] = delta
    return float(np.dot(delta_3d, direction / np.linalg.norm(direction)))


__all__ = ["OrigamiEngine", "compute_forward_displacement"]
