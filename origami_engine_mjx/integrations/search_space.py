from __future__ import annotations

"""Search-space / genome specification for optimization pipelines.

This module defines :class:`SearchSpace`, a documented, serializable
description of the design space that optimizers (ES, CMA-ES, RL policies, …)
search over.  It ships **no** optimization algorithm; it only encodes and
decodes genomes and describes their bounds.

Design space
------------
For the current geometry backend the crease **topology is fixed**: the segment
endpoints (``p1``/``p2``) of every crease are held constant and only the signed
fold **angles** vary.  This is exactly what
:func:`origami_engine_mjx.geometry.build_pattern` and
:func:`origami_engine_mjx.model.generate_xml` support today — the face graph is
reconstructed from the segment geometry, so moving endpoints can change the
number of faces and hinges, which the simulators cannot batch over.

Active vs. fixed creases
------------------------
Only creases that become **spanning-tree hinges** are actuated (see
:func:`origami_engine_mjx.model.generate_xml`: loop edges become equality
``connect`` constraints and their angle is unused).  With ``active_only=True``
(default) the genome therefore contains one angle per *actuated* hinge and the
remaining (loop) crease angles are held at their base values.  With
``active_only=False`` every crease angle is a genome variable, including the
inert loop creases.

The default 4×4 Miura-ori reference pattern has 24 creases, of which 15 are
actuated and 9 are loop edges, so the default genome is 15-dimensional.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..constants import MAX_FOLD
from ..geometry import build_pattern
from ..patterns import load_default_creases
from ..serialization import normalize_creases
from ..types import Crease

__all__ = ["SearchSpace", "VectorLike"]


def _as_rng(rng) -> np.random.Generator:
    """Coerce ``None`` / ``int`` / :class:`numpy.random.Generator` to a Generator."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


#: A 1-D genome: either a NumPy array or any Python sequence of floats.
VectorLike = np.ndarray | Sequence[float]


@dataclass(eq=False)
class SearchSpace:
    """Fixed-topology fold-angle design space.

    Instances are plain dataclasses over NumPy arrays and Python lists, so they
    are picklable and can be shipped to ``multiprocessing`` workers.

    Parameters
    ----------
    topology : list[tuple[tuple[float, float], tuple[float, float]]]
        Fixed ``(p1, p2)`` endpoints for every crease in the pattern.
    base_angles : numpy.ndarray
        Full-length ``(n_creases,)`` base fold angles (radians).  Inactive
        creases keep these values in :meth:`decode`.
    active_indices : numpy.ndarray
        Indices into ``topology`` that are variable genome entries.  Sorted
        ascending.  ``dim == len(active_indices)``.
    lower, upper : numpy.ndarray
        Per-variable angle bounds (radians), both ``(dim,)``.
    parameter_names : list[str]
        Human-readable name per variable, e.g. ``"crease_3.angle"``.
    name : str
        Optional label for logs.
    """

    topology: list[tuple[tuple[float, float], tuple[float, float]]]
    base_angles: np.ndarray
    active_indices: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    parameter_names: list[str] = field(default_factory=list)
    name: str = "origami"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_creases(
        cls,
        creases: Iterable,
        active_only: bool = True,
        angle_limits: tuple[float, float] = (-MAX_FOLD, MAX_FOLD),
        name: str = "origami",
    ) -> SearchSpace:
        """Build a search space from a template crease pattern.

        Parameters
        ----------
        creases :
            Any input accepted by
            :func:`origami_engine_mjx.serialization.normalize_creases`
            (list of :class:`~origami_engine_mjx.types.Crease`, list of dicts,
            or a JSON path).  Its geometry defines the fixed topology and its
            angles define the base genome.
        active_only :
            When ``True`` (default) only actuated (spanning-tree) crease angles
            are variable.  When ``False`` every crease angle is variable.
        angle_limits :
            ``(low, high)`` radian bounds applied to every variable.  Defaults
            to ``(-MAX_FOLD, +MAX_FOLD)`` with
            ``MAX_FOLD = π·0.98`` (the hinge target clip in
            :func:`origami_engine_mjx.model.generate_xml`).
        name :
            Label stored on the space.

        Raises
        ------
        ValueError
            If the pattern is invalid or ``angle_limits`` is malformed.
        """
        normalized = normalize_creases(creases)
        pat = build_pattern(normalized)
        if not pat.valid:
            raise ValueError(f"cannot build a search space from an invalid pattern: {pat.reason}")

        topology = [(tuple(c.p1), tuple(c.p2)) for c in normalized]
        base_angles = np.asarray([float(c.angle) for c in normalized], dtype=float)

        n = len(normalized)
        if active_only:
            active = sorted({int(pat.edges[k]["crease"]) for k in pat.tree_edge.values()})
            active = [i for i in active if i >= 0]
            if not active:
                raise ValueError("pattern has no actuated creases to vary.")
        else:
            active = list(range(n))

        lo, hi = float(angle_limits[0]), float(angle_limits[1])
        if not lo < hi:
            raise ValueError(f"angle_limits must be increasing, got {angle_limits!r}")

        active_indices = np.asarray(active, dtype=int)
        lower = np.full(active_indices.size, lo, dtype=float)
        upper = np.full(active_indices.size, hi, dtype=float)
        names = [f"crease_{int(i)}.angle" for i in active_indices]
        return cls(
            topology=topology,
            base_angles=base_angles,
            active_indices=active_indices,
            lower=lower,
            upper=upper,
            parameter_names=names,
            name=name,
        )

    @classmethod
    def from_default_pattern(cls, active_only: bool = True, **kwargs) -> SearchSpace:
        """Convenience constructor over the packaged reference Miura-ori."""
        return cls.from_creases(
            load_default_creases(),
            active_only=active_only,
            name="reference_miura_4x4",
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Shape / metadata
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Number of genome variables (``len(active_indices)``)."""
        return int(self.active_indices.size)

    @property
    def n_creases(self) -> int:
        """Total number of creases in the fixed topology."""
        return len(self.topology)

    @property
    def normalized_lower(self) -> np.ndarray:
        """Lower bounds of the normalized ``[-1, 1]`` view."""
        return -np.ones(self.dim, dtype=float)

    @property
    def normalized_upper(self) -> np.ndarray:
        """Upper bounds of the normalized ``[-1, 1]`` view."""
        return np.ones(self.dim, dtype=float)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a ``(lower, upper)`` copy of the angle bounds."""
        return self.lower.copy(), self.upper.copy()

    def __len__(self) -> int:
        return self.dim

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SearchSpace(name={self.name!r}, dim={self.dim}, "
            f"n_creases={self.n_creases}, active={self.active_indices.tolist()})"
        )

    def __eq__(self, other: object) -> bool:
        """Value equality over the topology, bounds and base angles.

        The auto-generated dataclass ``__eq__`` would compare the NumPy array
        fields elementwise and raise ``ValueError``; this override compares them
        with :func:`numpy.array_equal` instead.  Instances are therefore
        unhashable (defining ``__eq__`` clears ``__hash__``).
        """
        if not isinstance(other, SearchSpace):
            return NotImplemented
        return (
            self.name == other.name
            and self.topology == other.topology
            and np.array_equal(self.active_indices, other.active_indices)
            and np.array_equal(self.base_angles, other.base_angles)
            and np.array_equal(self.lower, other.lower)
            and np.array_equal(self.upper, other.upper)
        )

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def decode(self, vector: VectorLike) -> list[Crease]:
        """Map a genome ``(dim,)`` to a full list of :class:`Crease` objects.

        Inactive creases are filled from :attr:`base_angles`.  No clipping is
        performed here; the physics clips hinge targets to ``±MAX_FOLD``
        internally, and :meth:`clip` / :meth:`repair` are available when a
        caller wants the vector itself to stay in bounds.

        Raises
        ------
        ValueError
            If ``vector`` does not have length :attr:`dim`.
        """
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"genome has length {v.size}, expected {self.dim}")
        angles = self.base_angles.copy()
        angles[self.active_indices] = v
        return [Crease(p1, p2, float(a)) for (p1, p2), a in zip(self.topology, angles)]

    def encode(self, creases: Iterable) -> np.ndarray:
        """Map creases back to a genome vector.

        The crease geometry must match :attr:`topology` (same creases in the
        same order); otherwise a :class:`ValueError` is raised.  Only the
        angles at :attr:`active_indices` are returned.

        Raises
        ------
        ValueError
            If the number of creases or any endpoint differs from the topology.
        """
        normalized = normalize_creases(creases)
        if len(normalized) != self.n_creases:
            raise ValueError(f"pattern has {len(normalized)} creases, expected {self.n_creases}")
        for idx, c in enumerate(normalized):
            p1, p2 = self.topology[idx]
            if not (np.allclose(c.p1, p1) and np.allclose(c.p2, p2)):
                raise ValueError(f"crease {idx} endpoints do not match the search-space topology")
        angles = np.asarray([float(c.angle) for c in normalized], dtype=float)
        return angles[self.active_indices].copy()

    # ------------------------------------------------------------------
    # Bounds / repair
    # ------------------------------------------------------------------

    def clip(self, vector: VectorLike) -> np.ndarray:
        """Clip a genome to the angle bounds (no NaN handling)."""
        return np.clip(np.asarray(vector, dtype=float).reshape(-1), self.lower, self.upper)

    def repair(self, vector: VectorLike) -> np.ndarray:
        """Clip to bounds and replace non-finite entries with the base angle.

        Safe to call inside an optimizer loop; never raises for a
        correctly-sized input.
        """
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"genome has length {v.size}, expected {self.dim}")
        bad = ~np.isfinite(v)
        if np.any(bad):
            v = v.copy()
            v[bad] = self.base_angles[self.active_indices][bad]
        return np.clip(v, self.lower, self.upper)

    def is_valid(self, vector: VectorLike) -> bool:
        """Return whether the genome decodes to a valid crease pattern."""
        try:
            return bool(build_pattern(self.decode(vector)).valid)
        except Exception:  # noqa: BLE001 - validity probe must never raise
            return False

    # ------------------------------------------------------------------
    # Normalized [-1, 1] view
    # ------------------------------------------------------------------

    def to_normalized(self, vector: VectorLike) -> np.ndarray:
        """Map angle-space genome to the normalized ``[-1, 1]`` view."""
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"genome has length {v.size}, expected {self.dim}")
        span = self.upper - self.lower
        return (2.0 * (v - self.lower) / span) - 1.0

    def from_normalized(self, normalized: VectorLike) -> np.ndarray:
        """Map a normalized ``[-1, 1]`` vector back to angle space (clipped)."""
        z = np.asarray(normalized, dtype=float).reshape(-1)
        if z.size != self.dim:
            raise ValueError(f"normalized genome has length {z.size}, expected {self.dim}")
        z = np.clip(z, -1.0, 1.0)
        span = self.upper - self.lower
        return self.lower + (z + 1.0) * 0.5 * span

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, rng=None, size: int | None = None) -> np.ndarray:
        """Draw uniform random genomes in angle space.

        Parameters
        ----------
        rng :
            ``None``, an integer seed, or a :class:`numpy.random.Generator`.
        size :
            ``None`` returns a single ``(dim,)`` genome; an integer ``N``
            returns an ``(N, dim)`` batch.
        """
        g = _as_rng(rng)
        if size is None:
            return g.uniform(self.lower, self.upper)
        return g.uniform(self.lower, self.upper, size=(int(size), self.dim))

    def sample_normalized(self, rng=None, size: int | None = None) -> np.ndarray:
        """Draw uniform random genomes in the normalized ``[-1, 1]`` view."""
        g = _as_rng(rng)
        if size is None:
            return g.uniform(-1.0, 1.0, size=self.dim)
        return g.uniform(-1.0, 1.0, size=(int(size), self.dim))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-friendly description of the space (topology + bounds)."""
        return {
            "name": self.name,
            "dim": self.dim,
            "n_creases": self.n_creases,
            "active_indices": [int(i) for i in self.active_indices],
            "parameter_names": list(self.parameter_names),
            "lower": [float(x) for x in self.lower],
            "upper": [float(x) for x in self.upper],
            "base_angles": [float(x) for x in self.base_angles],
            "topology": [[list(p1), list(p2)] for p1, p2 in self.topology],
        }
