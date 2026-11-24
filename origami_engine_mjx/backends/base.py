from __future__ import annotations

"""Abstract simulation backend interface.

Each backend (MJX-JAX, MJX-Warp) implements this interface so that the public
:class:`~origami_engine_mjx.engine.OrigamiEngine` API can be backend-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..types import SimulationResult


class SimulationBackend(ABC):
    """Common interface for MJX simulation backends."""

    name: str = "abstract"

    @abstractmethod
    def build_model(self, xml: str, info: dict[str, Any]) -> Any:
        """Convert MJCF XML into a backend-native model object.

        Returns a backend-specific model handle (not necessarily a PyTree).
        """

    @abstractmethod
    def make_state(self, model: Any) -> Any:
        """Create an initial simulation state for the given model."""

    @abstractmethod
    def run(
        self,
        creases,
        duration: float = 8.0,
        fold_duration: float = 3.0,
        vibrate: bool = True,
        amp_x: float = 0.0,
        amp_z: float = 0.0013,
        freq_hz: float = 75.0,
        coll_thick: float | None = None,
        *,
        actuated=None,
        actuation_schedule=None,
    ) -> SimulationResult:
        """Run a single simulation and return the result.

        Backends also accept ``record=True`` / ``record_every=int`` to store a
        per-frame trajectory in ``result.extra["trajectory"]``.

        ``actuated`` chooses which hinges are driven (``None`` = all, a bool,
        or one bool per hinge).  ``actuation_schedule`` is an optional
        ``f(time_seconds) -> actuated`` evaluated every physics step, so a
        hinge can be driven for part of the run and free for the rest.
        """

    @abstractmethod
    def run_batch(
        self,
        population: list,
        duration: float = 8.0,
        fold_duration: float = 3.0,
        vibrate: bool = True,
        amp_x: float = 0.0,
        amp_z: float = 0.0013,
        freq_hz: float = 75.0,
        batch_size: int | None = None,
        actuated=None,
        actuation_schedule=None,
    ) -> np.ndarray:
        """Run a batch of crease patterns and return fitness scores.

        Implementations currently evaluate individuals sequentially, one
        ``run`` call each.  Returns an array of forward-displacement scores,
        one per individual.  ``actuated`` / ``actuation_schedule`` are
        forwarded to :meth:`run`.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend's dependencies are installed and usable."""

    @abstractmethod
    def device_info(self) -> str:
        """Return a human-readable description of the compute device in use."""


__all__ = ["SimulationBackend"]
