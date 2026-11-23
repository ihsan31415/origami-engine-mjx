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

from typing import Literal

import numpy as np

from .backends import available_backends, get_backend
from .backends.base import SimulationBackend
from .constants import VIB_AMP_X, VIB_AMP_Z, VIB_FREQ
from .types import BackendName, PopulationResult, SimulationResult

BackendChoice = Literal["jax", "warp", "auto"]


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
            backend = "warp" if "warp" in avail else avail[0]
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
    ) -> SimulationResult:
        """Run a single origami fold simulation."""
        return self._backend.run(
            creases,
            duration=duration,
            fold_duration=fold_duration,
            vibrate=vibrate,
            amp_x=amp_x,
            amp_z=amp_z,
            freq_hz=freq_hz,
            coll_thick=coll_thick,
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
    ) -> float:
        """Evaluate one crease configuration; return forward displacement."""
        result = self.run(
            creases,
            duration=duration,
            fold_duration=fold_duration,
            vibrate=vibrate,
            amp_x=amp_x,
            amp_z=amp_z,
            freq_hz=freq_hz,
        )
        return compute_forward_displacement(result, trajectory=trajectory)

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
    ) -> PopulationResult:
        """Evaluate a population of crease patterns in batch."""
        direction = np.asarray(trajectory, dtype=float)
        if direction.size == 2:
            direction = np.array([direction[0], direction[1], 0.0])
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("trajectory direction must be non-zero.")
        direction = direction / norm

        scores = self._backend.run_batch(
            population,
            duration=duration,
            fold_duration=fold_duration,
            vibrate=vibrate,
            amp_x=amp_x,
            amp_z=amp_z,
            freq_hz=freq_hz,
            batch_size=batch_size,
        )
        return PopulationResult(
            scores=scores,
            backend=self._backend_name,
            n_individuals=len(population),
            duration=duration,
        )


def compute_forward_displacement(
    sim_result: SimulationResult,
    trajectory: tuple[float, ...] = (1.0, 0.0, 0.0),
) -> float:
    """Project centroid displacement onto the trajectory direction."""
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
