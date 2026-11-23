from __future__ import annotations

"""Core domain types for crease patterns and simulation results."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

BackendName = Literal["jax", "warp"]


@dataclass(frozen=True)
class Crease:
    """A single crease segment with a signed target fold angle (radians).

    Negative angles are mountain folds, positive angles are valley folds.
    """

    p1: tuple[float, float]
    p2: tuple[float, float]
    angle: float


@dataclass
class Pattern:
    """Validated face/edge graph reconstructed from a set of creases."""

    valid: bool
    reason: str = ""
    faces: list | None = None
    edges: list | None = None
    root: int = 0
    parent: dict | None = None
    depth: dict | None = None
    tree_edge: dict | None = None
    loop_edges: list | None = None
    n_faces: int = 0
    n_loops: int = 0


@dataclass
class SimulationResult:
    """Final simulation state plus the centroid bookkeeping used for scoring.

    ``model`` and ``data`` are backend-specific objects (MJX-JAX Model/Data,
    MJX-Warp model/state, or classic MuJoCo MjModel/MjData for rendering).
    """

    model: Any
    data: Any
    pattern: Pattern
    info: dict[str, Any]
    initial_centroid: np.ndarray
    final_centroid: np.ndarray
    backend: str = "jax"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PopulationResult:
    """Result of batch population evaluation."""

    scores: np.ndarray
    backend: str
    n_individuals: int
    duration: float
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BackendName",
    "Crease",
    "Pattern",
    "PopulationResult",
    "SimulationResult",
]
