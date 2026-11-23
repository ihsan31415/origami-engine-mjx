from __future__ import annotations

"""Configurable design-space specification for optimization pipelines.

:class:`DesignSpace` generalizes
:class:`~origami_engine_mjx.integrations.search_space.SearchSpace` (the
fixed-topology, actuated-angle-only genome) so a pipeline can choose *what* it
optimizes:

* **fold angles** of the actuated (spanning-tree) creases — the default, which
  reproduces ``SearchSpace`` exactly;
* **fold angles** of the inert loop edges;
* **crease endpoints** (geometry) — either per crease or as shared vertices
  (whole-pattern);
* **crease presence**, which gives a variable crease *count*.

This module ships **no** optimization algorithm: it only encodes and decodes
genomes, describes their bounds, and reports whether a genome is geometrically
valid.

Selection
---------
A *variable* is selected by crease index (``int`` / ``slice`` / list of ints /
boolean mask), by group tag, by parameter name, or by a predicate over
:class:`Variable` objects.  The four presets :meth:`DesignSpace.angles_only`,
:meth:`DesignSpace.angles_and_geometry`, :meth:`DesignSpace.geometry_only` and
:meth:`DesignSpace.everything` cover the common cases; the fluent
:class:`DesignSpaceBuilder` and the dict/JSON form cover the rest.

Constraints
-----------
Optional equality / symmetry constraints **reduce** the dimension: the genome
holds only the *free* variables and :meth:`DesignSpace.decode` re-expands the
dependent ones.  A symmetry ``var_j = -var_i`` halves the variables it links;
an equality ``var_j = var_i`` collapses them to one.

Variable crease count
---------------------
Adding a ``"presence"`` variable per slot lets a genome drop creases.  Two
views are supported: the padded view (:meth:`DesignSpace.presence_mask`, one
entry per base slot) and the list view (:meth:`DesignSpace.decode`, only the
present creases).  Trade-off: a reduced pattern may no longer tile the sheet
(:func:`origami_engine_mjx.geometry.build_pattern` then reports it invalid), so
use :meth:`DesignSpace.is_valid` / :meth:`DesignSpace.rejection_sample` and let
the evaluator's ``penalty``/``status`` fallback absorb the rest.

The default ``angles_only`` preset produces exactly the same vectors, bounds
and decoded creases as :class:`SearchSpace`.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..constants import MAX_FOLD, S
from ..geometry import build_pattern
from ..patterns import load_default_creases
from ..serialization import normalize_creases
from ..types import Crease
from .search_space import SearchSpace

__all__ = [
    "ANGLE",
    "COORD",
    "PRESENCE",
    "DesignSpace",
    "DesignSpaceBuilder",
    "Variable",
]

#: Variable kinds.
ANGLE = "angle"
COORD = "coord"
PRESENCE = "presence"

_KINDS = (ANGLE, COORD, PRESENCE)
_ENDPOINTS = ("p1", "p2")
_COMPONENTS = ("x", "y")

#: A genome: a NumPy array or any sequence of floats.
VectorLike = np.ndarray | Sequence[float]


def _as_rng(rng) -> np.random.Generator:
    """Coerce ``None`` / ``int`` / :class:`numpy.random.Generator` to a Generator."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _resolve_indices(selector, n: int) -> list[int]:
    """Resolve a crease selection into a sorted list of indices.

    Accepts an ``int``, a ``slice``, a list/array of ints, a boolean mask, or
    ``None`` (meaning *all*).  Raises :class:`ValueError` on an out-of-range or
    wrong-length selection.
    """
    if selector is None:
        return list(range(n))
    if isinstance(selector, (int, np.integer)):
        idx = int(selector)
        if not -n <= idx < n:
            raise ValueError(f"crease index {idx} out of range for {n} creases")
        return [idx % n]
    if isinstance(selector, slice):
        return list(range(*selector.indices(n)))
    arr = np.asarray(selector)
    if arr.dtype == bool:
        if arr.shape != (n,):
            raise ValueError(f"boolean mask must have length {n}, got {arr.shape}")
        return [i for i in range(n) if bool(arr[i])]
    if arr.ndim == 0:
        return _resolve_indices(int(arr), n)
    out: list[int] = []
    for raw in np.asarray(selector, dtype=int).reshape(-1).tolist():
        i = int(raw)
        if not -n <= i < n:
            raise ValueError(f"crease index {i} out of range for {n} creases")
        out.append(i % n)
    return sorted(set(out))


@dataclass(eq=False)
class Variable:
    """A single optimized quantity in a :class:`DesignSpace`.

    Parameters
    ----------
    kind : {"angle", "coord", "presence"}
        What the variable controls.
    targets : list[tuple[int, str]]
        ``(crease_index, endpoint)`` pairs the variable writes to.  ``endpoint``
        is ``"p1"``/``"p2"`` for ``kind="coord"`` and ``""`` otherwise.  A coord
        variable may have several targets (shared vertex across creases).
    component : {"x", "y", ""}
        Which endpoint component a coord variable writes.
    group : str
        Selection tag (e.g. ``"angles"``, ``"geometry"``, ``"presence"``).
    name : str
        Human-readable name, e.g. ``"crease_3.angle"``.
    lower, upper : float
        Per-parameter bounds.
    """

    kind: str
    targets: list[tuple[int, str]]
    component: str = ""
    group: str = ""
    name: str = ""
    lower: float = 0.0
    upper: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown variable kind {self.kind!r}; expected one of {_KINDS}")
        if self.kind == COORD:
            if self.component not in _COMPONENTS:
                raise ValueError(f"coord component must be one of {_COMPONENTS}, got {self.component!r}")
            for _, endpoint in self.targets:
                if endpoint not in _ENDPOINTS:
                    raise ValueError(f"coord endpoint must be one of {_ENDPOINTS}, got {endpoint!r}")
        if not self.lower < self.upper:
            raise ValueError(f"variable bounds must be increasing, got ({self.lower}, {self.upper})")

    @property
    def n_targets(self) -> int:
        """Number of crease endpoints (or angles) the variable writes."""
        return len(self.targets)

    def to_dict(self) -> dict:
        """JSON-friendly description of the variable."""
        return {
            "kind": self.kind,
            "targets": [[int(i), str(ep)] for i, ep in self.targets],
            "component": self.component,
            "group": self.group,
            "name": self.name,
            "lower": float(self.lower),
            "upper": float(self.upper),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Variable:
        """Rebuild a variable from :meth:`to_dict` output."""
        return cls(
            kind=str(payload["kind"]),
            targets=[(int(i), str(ep)) for i, ep in payload["targets"]],
            component=str(payload.get("component", "")),
            group=str(payload.get("group", "")),
            name=str(payload.get("name", "")),
            lower=float(payload["lower"]),
            upper=float(payload["upper"]),
        )


def _angle_variable(index: int, lower: float, upper: float, group: str = "angles") -> Variable:
    return Variable(ANGLE, [(index, "")], group=group, name=f"crease_{index}.angle", lower=lower, upper=upper)


def _coord_variable(
    targets: list[tuple[int, str]], component: str, lower: float, upper: float, group: str = "geometry", name: str = ""
) -> Variable:
    if not name:
        name = f"vertex_{component}:{targets[0][0]}{targets[0][1]}" if len(targets) == 1 else f"vertex_{component}"
    return Variable(COORD, targets, component=component, group=group, name=name, lower=lower, upper=upper)


def _presence_variable(index: int, group: str = "presence") -> Variable:
    return Variable(PRESENCE, [(index, "")], group=group, name=f"crease_{index}.present", lower=0.0, upper=1.0)


@dataclass(eq=False)
class DesignSpace:
    """Configurable design space over fold angles, endpoints and presence.

    Instances are plain dataclasses over NumPy arrays, Python lists and
    :class:`Variable` objects, so they are picklable and can be shipped to
    ``multiprocessing`` workers.

    Parameters
    ----------
    base_creases : list[Crease]
        The template pattern.  Every slot keeps its base angle and endpoints
        unless a variable overwrites them.
    variables : list[Variable]
        Full variable list (free + dependent).
    name : str
        Optional label for logs.
    dependent : dict[int, tuple[int, float]]
        Constraint expansion: ``dependent[j] = (i, sign)`` means variable ``j``
        is ``sign * variable[i]``.  Dependent indices are removed from the
        genome and re-expanded by :meth:`decode`.
    count_mode : {"fixed", "range", "free"}
        How the crease count may vary (only meaningful with presence
        variables).
    count_bounds : tuple[int, int] | None
        ``(min, max)`` present-crease count for ``count_mode="range"``.
    presence_threshold : float
        A presence variable at or above this value keeps its crease.
    angle_limits, coord_limits : tuple[float, float]
        Default bounds used by the constructors.
    """

    base_creases: list[Crease]
    variables: list[Variable]
    name: str = "origami"
    dependent: dict[int, tuple[int, float]] = field(default_factory=dict)
    count_mode: str = "fixed"
    count_bounds: tuple[int, int] | None = None
    presence_threshold: float = 0.5
    angle_limits: tuple[float, float] = (-MAX_FOLD, MAX_FOLD)
    coord_limits: tuple[float, float] = (0.0, S)

    def __post_init__(self) -> None:
        if not self.base_creases:
            raise ValueError("base_creases must not be empty")
        if not self.variables:
            raise ValueError("a design space must contain at least one variable")
        if self.count_mode not in {"fixed", "range", "free"}:
            raise ValueError("count_mode must be 'fixed', 'range' or 'free'")
        n = len(self.variables)
        for dep, (src, _) in self.dependent.items():
            if not 0 <= int(dep) < n or not 0 <= int(src) < n:
                raise ValueError(f"constraint index out of range: dependent[{dep}] = {src}")
            if int(dep) == int(src):
                raise ValueError(f"constraint links variable {dep} to itself")
        for v in self.variables:
            for ci, _ in v.targets:
                if not 0 <= ci < len(self.base_creases):
                    raise ValueError(f"variable {v.name!r} targets crease {ci}, but there are {len(self.base_creases)}")
        if self.count_mode == "range":
            if self.count_bounds is None:
                raise ValueError("count_mode='range' requires count_bounds=(min, max)")
            lo, hi = int(self.count_bounds[0]), int(self.count_bounds[1])
            if not 0 <= lo <= hi <= len(self.base_creases):
                raise ValueError(f"count_bounds must satisfy 0 <= min <= max <= {len(self.base_creases)}")

    # ------------------------------------------------------------------
    # Shape / metadata
    # ------------------------------------------------------------------

    @property
    def n_vars(self) -> int:
        """Total number of variables (free + dependent)."""
        return len(self.variables)

    @property
    def n_slots(self) -> int:
        """Number of crease slots in the template (``len(base_creases)``)."""
        return len(self.base_creases)

    @property
    def free_indices(self) -> list[int]:
        """Indices of the variables stored in the genome (dependent removed)."""
        return [i for i in range(self.n_vars) if i not in self.dependent]

    @property
    def dim(self) -> int:
        """Number of genome variables (free variables)."""
        return len(self.free_indices)

    @property
    def n_free(self) -> int:
        """Alias of :attr:`dim` (free variables)."""
        return self.dim

    @property
    def n_fixed(self) -> int:
        """Number of variables removed by constraints (dependent variables)."""
        return len(self.dependent)

    @property
    def n_creases(self) -> int:
        """Alias of :attr:`n_slots` (template crease count)."""
        return self.n_slots

    @property
    def parameter_names(self) -> list[str]:
        """Human-readable name per free variable."""
        return [self.variables[i].name for i in self.free_indices]

    @property
    def groups(self) -> list[str]:
        """Group tag per free variable."""
        return [self.variables[i].group for i in self.free_indices]

    @property
    def lower(self) -> np.ndarray:
        """Per-free-variable lower bounds."""
        return np.asarray([self.variables[i].lower for i in self.free_indices], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        """Per-free-variable upper bounds."""
        return np.asarray([self.variables[i].upper for i in self.free_indices], dtype=float)

    @property
    def normalized_lower(self) -> np.ndarray:
        """Lower bounds of the normalized ``[-1, 1]`` view."""
        return -np.ones(self.dim, dtype=float)

    @property
    def normalized_upper(self) -> np.ndarray:
        """Upper bounds of the normalized ``[-1, 1]`` view."""
        return np.ones(self.dim, dtype=float)

    @property
    def has_presence(self) -> bool:
        """Whether any variable controls crease presence (variable count)."""
        return any(v.kind == PRESENCE for v in self.variables)

    @property
    def variable_kinds(self) -> list[str]:
        """Kind per free variable."""
        return [self.variables[i].kind for i in self.free_indices]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a ``(lower, upper)`` copy of the free-variable bounds."""
        return self.lower.copy(), self.upper.copy()

    def group_slice(self, group: str) -> list[int]:
        """Genome indices whose group equals ``group``."""
        return [k for k, i in enumerate(self.free_indices) if self.variables[i].group == group]

    def select(
        self,
        *,
        group: str | None = None,
        name: str | None = None,
        indices: Any = None,
        predicate: Callable[[Variable], bool] | None = None,
    ) -> list[int]:
        """Return genome indices matching a selection (AND of the given filters).

        Parameters
        ----------
        group : str | None
            Keep variables whose ``group`` equals this tag.
        name : str | None
            Keep variables whose ``name`` equals this string.
        indices : int | slice | sequence | boolean mask | None
            Keep genome positions in this selection.
        predicate : callable(Variable) -> bool | None
            Keep variables for which the predicate is true.
        """
        sel = set(range(self.dim))
        if indices is not None:
            sel &= set(_resolve_indices(indices, self.dim))
        if group is not None:
            sel &= set(self.group_slice(group))
        if name is not None:
            sel &= {k for k, i in enumerate(self.free_indices) if self.variables[i].name == name}
        if predicate is not None:
            sel &= {k for k, i in enumerate(self.free_indices) if predicate(self.variables[i])}
        return sorted(sel)

    def __len__(self) -> int:
        return self.dim

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DesignSpace(name={self.name!r}, dim={self.dim}, n_slots={self.n_slots}, "
            f"kinds={sorted(set(self.variable_kinds))}, n_fixed={self.n_fixed})"
        )

    def __eq__(self, other: object) -> bool:
        """Value equality over base creases, variables and constraints."""
        if not isinstance(other, DesignSpace):
            return NotImplemented
        return (
            self.name == other.name
            and self.base_creases == other.base_creases
            and [v.to_dict() for v in self.variables] == [v.to_dict() for v in other.variables]
            and self.dependent == other.dependent
            and self.count_mode == other.count_mode
            and self.count_bounds == other.count_bounds
        )

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def _expand(self, vector: VectorLike) -> np.ndarray:
        """Expand a genome ``(dim,)`` into the full ``(n_vars,)`` variable values."""
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"genome has length {v.size}, expected {self.dim}")
        full = np.empty(self.n_vars, dtype=float)
        full[self.free_indices] = v
        for dep, (src, sign) in self.dependent.items():
            full[dep] = sign * full[src]
        return full

    def decode(self, vector: VectorLike) -> list[Crease]:
        """Map a genome ``(dim,)`` to the present :class:`Crease` list.

        Angles, endpoints and presence are overwritten from the genome; absent
        creases (presence below :attr:`presence_threshold`) are dropped, so the
        returned list is the *list view* of the pattern.  No clipping is
        performed; :meth:`clip` / :meth:`repair` are available when the vector
        itself must stay in bounds.

        Raises
        ------
        ValueError
            If ``vector`` does not have length :attr:`dim`.
        """
        full = self._expand(vector)
        n = self.n_slots
        angles = np.asarray([c.angle for c in self.base_creases], dtype=float)
        p1 = np.asarray([c.p1 for c in self.base_creases], dtype=float)
        p2 = np.asarray([c.p2 for c in self.base_creases], dtype=float)
        present = np.ones(n, dtype=bool)
        for var, val in zip(self.variables, full):
            if var.kind == ANGLE:
                for ci, _ in var.targets:
                    angles[ci] = val
            elif var.kind == COORD:
                col = 0 if var.component == "x" else 1
                for ci, endpoint in var.targets:
                    (p1 if endpoint == "p1" else p2)[ci, col] = val
            else:  # PRESENCE
                for ci, _ in var.targets:
                    present[ci] = bool(val >= self.presence_threshold)
        return [
            Crease((float(p1[i, 0]), float(p1[i, 1])), (float(p2[i, 0]), float(p2[i, 1])), float(angles[i]))
            for i in range(n)
            if present[i]
        ]

    def decode_padded(self, vector: VectorLike) -> list[Crease]:
        """Decode to the *padded* view: one crease per slot, absent slots dropped too.

        This is the same crease list as :meth:`decode` but the presence mask is
        exposed separately by :meth:`presence_mask`, so a fixed-length pipeline
        can align individuals slot by slot.
        """
        return self.decode(vector)

    def presence_mask(self, vector: VectorLike) -> np.ndarray:
        """Return the ``(n_slots,)`` boolean presence mask for a genome.

        Every slot is ``True`` unless a presence variable drops it, so a space
        without presence variables always returns all-``True``.
        """
        full = self._expand(vector)
        mask = np.ones(self.n_slots, dtype=bool)
        for var, val in zip(self.variables, full):
            if var.kind == PRESENCE:
                for ci, _ in var.targets:
                    mask[ci] = bool(val >= self.presence_threshold)
        return mask

    def base_vector(self) -> np.ndarray:
        """Return the genome that reproduces the template pattern exactly.

        Angles come from the base creases, endpoints from the base geometry and
        presence is ``1.0`` (all creases present).
        """
        out = np.empty(self.dim, dtype=float)
        for k, i in enumerate(self.free_indices):
            var = self.variables[i]
            if var.kind == ANGLE:
                out[k] = float(self.base_creases[var.targets[0][0]].angle)
            elif var.kind == COORD:
                crease = self.base_creases[var.targets[0][0]]
                pt = crease.p1 if var.targets[0][1] == "p1" else crease.p2
                out[k] = float(pt[0] if var.component == "x" else pt[1])
            else:  # PRESENCE
                out[k] = 1.0
        return out

    def encode(self, creases: Iterable) -> np.ndarray:
        """Map creases back to a genome vector.

        Creases are matched to template slots by their endpoints (so presence
        may drop slots).  Only the free variables are returned; dependent values
        are reconstructed by :meth:`decode`.

        Raises
        ------
        ValueError
            If a crease does not match any template slot or endpoints are
            ambiguous (duplicate template creases).
        """
        normalized = normalize_creases(creases)
        slot_of: dict[tuple, int] = {}
        for i, c in enumerate(self.base_creases):
            key = _slot_key(c.p1, c.p2)
            if key in slot_of:
                raise ValueError("cannot encode with duplicate template creases; endpoints are ambiguous")
            slot_of[key] = i
        by_slot: dict[int, Crease] = {}
        for c in normalized:
            key = _slot_key(c.p1, c.p2)
            if key not in slot_of:
                raise ValueError("a crease does not match any template slot; cannot encode into this design space")
            by_slot[slot_of[key]] = c
        full = np.zeros(self.n_vars, dtype=float)
        for i, var in enumerate(self.variables):
            if var.kind == PRESENCE:
                full[i] = 1.0 if var.targets[0][0] in by_slot else 0.0
                continue
            ci = var.targets[0][0]
            crease = by_slot.get(ci)
            if crease is None:
                raise ValueError(f"variable {var.name!r} needs crease {ci}, which is absent from the input")
            if var.kind == ANGLE:
                full[i] = float(crease.angle)
            else:  # COORD
                pt = crease.p1 if var.targets[0][1] == "p1" else crease.p2
                full[i] = float(pt[0] if var.component == "x" else pt[1])
        return full[self.free_indices].copy()

    # ------------------------------------------------------------------
    # Bounds / repair / validity
    # ------------------------------------------------------------------

    def clip(self, vector: VectorLike) -> np.ndarray:
        """Clip a genome to the free-variable bounds (no NaN handling)."""
        return np.clip(np.asarray(vector, dtype=float).reshape(-1), self.lower, self.upper)

    def repair(self, vector: VectorLike) -> np.ndarray:
        """Clip to bounds and replace non-finite entries with the base vector.

        Safe to call inside an optimizer loop; never raises for a correctly
        sized input.  This is a *cheap projection* (bounds only); it does not
        guarantee geometric validity — use :meth:`is_valid` /
        :meth:`rejection_sample` for that.
        """
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"genome has length {v.size}, expected {self.dim}")
        bad = ~np.isfinite(v)
        if np.any(bad):
            v = v.copy()
            v[bad] = self.base_vector()[bad]
        return np.clip(v, self.lower, self.upper)

    def is_valid(self, vector: VectorLike) -> bool:
        """Whether the genome decodes to a pattern that tiles the sheet.

        Uses :func:`origami_engine_mjx.geometry.build_pattern`; a decode or
        geometry error is reported as ``False`` rather than raised.
        """
        try:
            pat = build_pattern(self.decode(vector))
            return bool(pat.valid)
        except Exception:  # noqa: BLE001 - validity probe must never raise
            return False

    def validity_reason(self, vector: VectorLike) -> str:
        """Human-readable reason a genome is invalid (``""`` when valid)."""
        try:
            pat = build_pattern(self.decode(vector))
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            return f"{type(exc).__name__}: {exc}"
        return "" if pat.valid else pat.reason

    def rejection_sample(
        self, rng=None, size: int | None = None, max_tries: int = 100, normalized: bool = False
    ) -> np.ndarray:
        """Draw valid genomes by rejection sampling.

        Parameters
        ----------
        rng :
            ``None``, an integer seed, or a :class:`numpy.random.Generator`.
        size :
            ``None`` returns a single ``(dim,)`` genome; an integer ``N``
            returns an ``(N, dim)`` batch (which may contain fewer than ``N``
            rows if the budget runs out).
        max_tries : int
            Maximum draws per requested genome.
        normalized : bool
            Sample in the normalized ``[-1, 1]`` view when ``True``.

        Returns
        -------
        numpy.ndarray
            A ``(dim,)`` or ``(N, dim)`` array of valid genomes.
        """
        g = _as_rng(rng)
        single = size is None
        want = 1 if size is None else int(size)
        if want < 0:
            raise ValueError("size must be non-negative")
        accepted: list[np.ndarray] = []
        attempts = 0
        budget = max_tries * want if want else 0
        while len(accepted) < want and attempts < budget:
            attempts += 1
            cand = self.sample_normalized(g) if normalized else self.sample(g)
            if self.is_valid(cand):
                accepted.append(np.asarray(cand, dtype=float))
        if not accepted:
            raise RuntimeError(f"could not draw a valid genome in {budget} attempts; is the space feasible?")
        arr = np.asarray(accepted, dtype=float)
        return arr[0] if single else arr

    # ------------------------------------------------------------------
    # Normalized [-1, 1] view
    # ------------------------------------------------------------------

    def to_normalized(self, vector: VectorLike) -> np.ndarray:
        """Map a genome to the normalized ``[-1, 1]`` view."""
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"genome has length {v.size}, expected {self.dim}")
        span = self.upper - self.lower
        return (2.0 * (v - self.lower) / span) - 1.0

    def from_normalized(self, normalized: VectorLike) -> np.ndarray:
        """Map a normalized ``[-1, 1]`` vector back to genome space (clipped)."""
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
        """Draw uniform random genomes in genome space."""
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
    # Constraints
    # ------------------------------------------------------------------

    def with_constraints(
        self,
        equalities: Iterable[tuple[int, int]] = (),
        symmetries: Iterable[tuple[int, int]] = (),
    ) -> DesignSpace:
        """Return a copy with equality / symmetry constraints applied.

        ``(i, j)`` in ``equalities`` means variable ``j`` equals variable ``i``;
        ``(i, j)`` in ``symmetries`` means variable ``j`` equals ``-variable[i]``
        (a symmetry).  The dependent variable ``j`` is removed from the genome
        and re-expanded by :meth:`decode`, reducing :attr:`dim`.

        Raises
        ------
        ValueError
            If a constraint references an out-of-range index, links a variable
            to itself, or creates a cycle.
        """
        dependent: dict[int, tuple[int, float]] = dict(self.dependent)
        n = self.n_vars

        def resolve(idx: int) -> tuple[int, float]:
            sign = 1.0
            seen = set()
            while idx in dependent:
                if idx in seen:
                    raise ValueError("constraint cycle detected")
                seen.add(idx)
                src, s = dependent[idx]
                sign *= s
                idx = src
            return idx, sign

        for i, j in equalities:
            i, j = int(i), int(j)
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"constraint index out of range: ({i}, {j}) for {n} variables")
            if i == j:
                raise ValueError(f"constraint links variable {i} to itself")
            src, sign = resolve(i)
            dependent[j] = (src, sign)
        for i, j in symmetries:
            i, j = int(i), int(j)
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"constraint index out of range: ({i}, {j}) for {n} variables")
            if i == j:
                raise ValueError(f"constraint links variable {i} to itself")
            src, sign = resolve(i)
            dependent[j] = (src, -sign)
        # Re-resolve so every source is free and acyclic.
        resolved: dict[int, tuple[int, float]] = {}
        for dep in dependent:
            src, sign = resolve(dep)
            if src == dep:
                raise ValueError(f"constraint makes variable {dep} depend on itself")
            resolved[dep] = (src, sign)
        return replace(self, dependent=resolved)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the space."""
        counts: dict[str, int] = {}
        for kind in self.variable_kinds:
            counts[kind] = counts.get(kind, 0) + 1
        return {
            "name": self.name,
            "dim": self.dim,
            "n_vars": self.n_vars,
            "n_free": self.n_free,
            "n_fixed": self.n_fixed,
            "n_slots": self.n_slots,
            "kinds": counts,
            "groups": sorted(set(self.groups)),
            "has_presence": self.has_presence,
            "count_mode": self.count_mode,
            "count_bounds": None
            if self.count_bounds is None
            else [int(self.count_bounds[0]), int(self.count_bounds[1])],
            "parameter_names": list(self.parameter_names),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-friendly description of the space (template + variables)."""
        return {
            "name": self.name,
            "dim": self.dim,
            "n_slots": self.n_slots,
            "count_mode": self.count_mode,
            "count_bounds": None
            if self.count_bounds is None
            else [int(self.count_bounds[0]), int(self.count_bounds[1])],
            "presence_threshold": float(self.presence_threshold),
            "angle_limits": [float(self.angle_limits[0]), float(self.angle_limits[1])],
            "coord_limits": [float(self.coord_limits[0]), float(self.coord_limits[1])],
            "base_creases": [[list(c.p1), list(c.p2), float(c.angle)] for c in self.base_creases],
            "variables": [v.to_dict() for v in self.variables],
            "dependent": {str(int(k)): [int(v[0]), float(v[1])] for k, v in self.dependent.items()},
            "parameter_names": list(self.parameter_names),
            "lower": [float(x) for x in self.lower],
            "upper": [float(x) for x in self.upper],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> DesignSpace:
        """Rebuild a space from :meth:`to_dict` output."""
        base = [Crease(tuple(c[0]), tuple(c[1]), float(c[2])) for c in payload["base_creases"]]
        variables = [Variable.from_dict(v) for v in payload["variables"]]
        dependent = {int(k): (int(v[0]), float(v[1])) for k, v in payload.get("dependent", {}).items()}
        cb = payload.get("count_bounds")
        return cls(
            base_creases=base,
            variables=variables,
            name=str(payload.get("name", "origami")),
            dependent=dependent,
            count_mode=str(payload.get("count_mode", "fixed")),
            count_bounds=None if cb is None else (int(cb[0]), int(cb[1])),
            presence_threshold=float(payload.get("presence_threshold", 0.5)),
            angle_limits=tuple(float(x) for x in payload.get("angle_limits", (-MAX_FOLD, MAX_FOLD))),  # type: ignore[arg-type]
            coord_limits=tuple(float(x) for x in payload.get("coord_limits", (0.0, S))),  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Constructors / presets
    # ------------------------------------------------------------------

    @classmethod
    def from_search_space(cls, space: SearchSpace) -> DesignSpace:
        """Convert a fixed-topology :class:`SearchSpace` into a design space."""
        base = [Crease(p1, p2, float(a)) for (p1, p2), a in zip(space.topology, space.base_angles)]
        variables = [
            _angle_variable(int(i), float(space.lower[k]), float(space.upper[k]))
            for k, i in enumerate(space.active_indices)
        ]
        return cls(base_creases=base, variables=variables, name=space.name)

    @property
    def is_angle_only(self) -> bool:
        """Whether every variable is a fold angle (RL-compatible, fixed topology)."""
        return all(v.kind == ANGLE for v in self.variables)

    def to_search_space(self) -> SearchSpace:
        """Convert to a fixed-topology :class:`SearchSpace`.

        Only valid when every variable is a fold angle and there are no
        constraints (i.e. :attr:`is_angle_only` and :attr:`n_fixed` == 0).  This
        is how the RL environments consume a :class:`DesignSpace`: RL needs one
        shared topology and an angle-only genome, so geometry / crease-count
        spaces are rejected with a clear error rather than silently ignored.

        Raises
        ------
        ValueError
            If the space varies geometry or presence, or carries constraints.
        """
        if not self.is_angle_only:
            kinds = sorted({v.kind for v in self.variables})
            raise ValueError(
                "the RL environments need a fixed topology and an angle-only genome, but this "
                f"design space also varies {kinds}. Use DesignSpace.angles_only(...) / a SearchSpace "
                "for RL (see docs/RL.md), or run the evaluator instead."
            )
        if self.dependent:
            raise ValueError(
                "cannot convert a design space with equality/symmetry constraints to a SearchSpace; "
                "constraints reduce the genome dimension, which the RL environments do not support."
            )
        active = [int(v.targets[0][0]) for v in self.variables]
        if len(set(active)) != len(active):
            raise ValueError("design space has more than one angle variable for the same crease")
        active_sorted = sorted(active)
        by_index = {int(v.targets[0][0]): v for v in self.variables}
        base_angles = np.asarray([float(c.angle) for c in self.base_creases], dtype=float)
        topology = [((float(c.p1[0]), float(c.p1[1])), (float(c.p2[0]), float(c.p2[1]))) for c in self.base_creases]
        return SearchSpace(
            topology=topology,
            base_angles=base_angles,
            active_indices=np.asarray(active_sorted, dtype=int),
            lower=np.asarray([by_index[i].lower for i in active_sorted], dtype=float),
            upper=np.asarray([by_index[i].upper for i in active_sorted], dtype=float),
            parameter_names=[by_index[i].name for i in active_sorted],
            name=self.name,
        )

    @classmethod
    def angles_only(
        cls,
        creases: Iterable | None = None,
        active_only: bool = True,
        angle_limits: tuple[float, float] = (-MAX_FOLD, MAX_FOLD),
        name: str = "origami",
    ) -> DesignSpace:
        """Preset: actuated fold angles only (default; matches ``SearchSpace``).

        With ``active_only=True`` (default) only spanning-tree hinges vary, so
        the vectors, bounds and decoded creases are identical to
        :meth:`SearchSpace.from_creases`.  With ``active_only=False`` every
        crease angle varies (including inert loop edges).
        """
        normalized = normalize_creases(load_default_creases() if creases is None else creases)
        pat = build_pattern(normalized)
        if not pat.valid:
            raise ValueError(f"cannot build a design space from an invalid pattern: {pat.reason}")
        if active_only:
            active = sorted({int(pat.edges[k]["crease"]) for k in pat.tree_edge.values()})
            active = [i for i in active if i >= 0]
            if not active:
                raise ValueError("pattern has no actuated creases to vary.")
        else:
            active = list(range(len(normalized)))
        lo, hi = float(angle_limits[0]), float(angle_limits[1])
        if not lo < hi:
            raise ValueError(f"angle_limits must be increasing, got {angle_limits!r}")
        variables = [_angle_variable(i, lo, hi) for i in active]
        return cls(base_creases=normalized, variables=variables, name=name, angle_limits=(lo, hi))

    @classmethod
    def from_creases(cls, creases: Iterable, **kwargs) -> DesignSpace:
        """Alias of :meth:`angles_only` (the default preset)."""
        return cls.angles_only(creases, **kwargs)

    @classmethod
    def from_default_pattern(cls, active_only: bool = True, **kwargs) -> DesignSpace:
        """Preset over the packaged reference Miura-ori (15-d by default)."""
        return cls.angles_only(load_default_creases(), active_only=active_only, name="reference_miura_4x4", **kwargs)

    @classmethod
    def angles_and_geometry(
        cls,
        creases: Iterable | None = None,
        active_only: bool = True,
        geometry_mode: str = "vertex",
        axes: str = "xy",
        angle_limits: tuple[float, float] = (-MAX_FOLD, MAX_FOLD),
        coord_limits: tuple[float, float] = (0.0, S),
        name: str = "origami",
    ) -> DesignSpace:
        """Preset: actuated angles plus crease endpoints (geometry)."""
        space = cls.angles_only(creases, active_only=active_only, angle_limits=angle_limits, name=name)
        builder = DesignSpaceBuilder(space.base_creases)
        for v in space.variables:
            builder.add(v)
        builder.geometry(mode=geometry_mode, axes=axes, bounds=coord_limits)
        return builder.build(name=name, angle_limits=angle_limits, coord_limits=coord_limits)

    @classmethod
    def geometry_only(
        cls,
        creases: Iterable | None = None,
        geometry_mode: str = "vertex",
        axes: str = "xy",
        coord_limits: tuple[float, float] = (0.0, S),
        name: str = "origami",
    ) -> DesignSpace:
        """Preset: crease endpoints only (no angles)."""
        normalized = normalize_creases(load_default_creases() if creases is None else creases)
        pat = build_pattern(normalized)
        if not pat.valid:
            raise ValueError(f"cannot build a design space from an invalid pattern: {pat.reason}")
        builder = DesignSpaceBuilder(normalized)
        builder.geometry(mode=geometry_mode, axes=axes, bounds=coord_limits)
        return builder.build(name=name, coord_limits=coord_limits)

    @classmethod
    def everything(
        cls,
        creases: Iterable | None = None,
        active_only: bool = True,
        geometry_mode: str = "vertex",
        axes: str = "xy",
        angle_limits: tuple[float, float] = (-MAX_FOLD, MAX_FOLD),
        coord_limits: tuple[float, float] = (0.0, S),
        presence: Any = None,
        count_mode: str = "free",
        count_bounds: tuple[int, int] | None = None,
        name: str = "origami",
    ) -> DesignSpace:
        """Preset: angles, geometry and (optionally) crease presence.

        With ``presence`` set (``None`` means every slot), the space also has a
        presence variable per selected slot, giving a variable crease count.
        """
        space = cls.angles_only(creases, active_only=active_only, angle_limits=angle_limits, name=name)
        builder = DesignSpaceBuilder(space.base_creases)
        for v in space.variables:
            builder.add(v)
        builder.geometry(mode=geometry_mode, axes=axes, bounds=coord_limits)
        if presence is not None:
            builder.presence(indices=presence)
        return builder.build(
            name=name,
            angle_limits=angle_limits,
            coord_limits=coord_limits,
            count_mode=count_mode if presence is not None else "fixed",
            count_bounds=count_bounds,
        )


def _slot_key(p1: Sequence[float], p2: Sequence[float], ndigits: int = 9) -> tuple:
    """Order-independent rounded key for a crease segment."""
    a = (round(float(p1[0]), ndigits), round(float(p1[1]), ndigits))
    b = (round(float(p2[0]), ndigits), round(float(p2[1]), ndigits))
    return (a, b) if a <= b else (b, a)


class DesignSpaceBuilder:
    """Fluent builder for :class:`DesignSpace`.

    Example
    -------
    >>> from origami_engine_mjx.integrations.design_space import DesignSpaceBuilder
    >>> from origami_engine_mjx.patterns import load_default_creases
    >>> b = DesignSpaceBuilder(load_default_creases())
    >>> b.angles(active_only=True)
    >>> b.geometry(mode="vertex")
    >>> space = b.build()                       # doctest: +SKIP
    """

    def __init__(self, creases: Iterable) -> None:
        self.base_creases: list[Crease] = normalize_creases(creases)
        self._variables: list[Variable] = []
        self._angle_limits: tuple[float, float] = (-MAX_FOLD, MAX_FOLD)
        self._coord_limits: tuple[float, float] = (0.0, S)
        self._count_mode = "fixed"
        self._count_bounds: tuple[int, int] | None = None
        self._presence_threshold = 0.5

    # -- variable sources -------------------------------------------------

    def add(self, variable: Variable) -> DesignSpaceBuilder:
        """Append an already-constructed :class:`Variable`."""
        self._variables.append(variable)
        return self

    def angles(
        self,
        active_only: bool = True,
        indices: Any = None,
        bounds: tuple[float, float] = (-MAX_FOLD, MAX_FOLD),
        group: str = "angles",
    ) -> DesignSpaceBuilder:
        """Add fold-angle variables.

        ``active_only=True`` (default) restricts to spanning-tree hinges;
        ``indices`` overrides the selection (crease index / slice / list / mask).
        """
        lo, hi = float(bounds[0]), float(bounds[1])
        if not lo < hi:
            raise ValueError(f"angle bounds must be increasing, got {bounds!r}")
        self._angle_limits = (lo, hi)
        if indices is not None:
            chosen = _resolve_indices(indices, len(self.base_creases))
        elif active_only:
            pat = build_pattern(self.base_creases)
            if not pat.valid:
                raise ValueError(f"cannot select actuated creases from an invalid pattern: {pat.reason}")
            chosen = sorted({int(pat.edges[k]["crease"]) for k in pat.tree_edge.values()})
            chosen = [i for i in chosen if i >= 0]
        else:
            chosen = list(range(len(self.base_creases)))
        for i in chosen:
            self._variables.append(_angle_variable(i, lo, hi, group=group))
        return self

    def loop_angles(
        self, bounds: tuple[float, float] = (-MAX_FOLD, MAX_FOLD), group: str = "loop_angles"
    ) -> DesignSpaceBuilder:
        """Add fold-angle variables for the inert loop edges only."""
        pat = build_pattern(self.base_creases)
        if not pat.valid:
            raise ValueError(f"cannot select loop creases from an invalid pattern: {pat.reason}")
        tree = {int(pat.edges[k]["crease"]) for k in pat.tree_edge.values()}
        loop = sorted({int(pat.edges[k]["crease"]) for k in pat.loop_edges} - tree)
        return self.angles(active_only=False, indices=[i for i in loop if i >= 0], bounds=bounds, group=group)

    def geometry(
        self,
        mode: str = "vertex",
        axes: str = "xy",
        bounds: tuple[float, float] = (0.0, S),
        group: str = "geometry",
        indices: Any = None,
    ) -> DesignSpaceBuilder:
        """Add endpoint-coordinate variables.

        ``mode="vertex"`` shares one variable per unique (rounded) vertex across
        every crease that uses it (whole-pattern); ``mode="segment"`` gives each
        crease endpoint its own variable (per-crease).  ``axes`` is any subset of
        ``"xy"``.
        """
        if mode not in {"vertex", "segment"}:
            raise ValueError("geometry mode must be 'vertex' or 'segment'")
        if any(a not in "xy" for a in axes) or not axes:
            raise ValueError("axes must be a non-empty subset of 'xy'")
        lo, hi = float(bounds[0]), float(bounds[1])
        if not lo < hi:
            raise ValueError(f"coord bounds must be increasing, got {bounds!r}")
        self._coord_limits = (lo, hi)
        chosen = set(_resolve_indices(indices, len(self.base_creases)))
        if mode == "segment":
            for ci in sorted(chosen):
                for ep in _ENDPOINTS:
                    for comp in axes:
                        self._variables.append(_coord_variable([(ci, ep)], comp, lo, hi, group=group))
        else:
            vertex_targets: dict[tuple[float, float], list[tuple[int, str]]] = {}
            for ci in sorted(chosen):
                for ep in _ENDPOINTS:
                    pt = self.base_creases[ci].p1 if ep == "p1" else self.base_creases[ci].p2
                    key = (round(float(pt[0]), 9), round(float(pt[1]), 9))
                    vertex_targets.setdefault(key, []).append((ci, ep))
            for key in sorted(vertex_targets):
                for comp in axes:
                    targets = vertex_targets[key]
                    name = f"vertex_{comp}({key[0]:g},{key[1]:g})"
                    self._variables.append(_coord_variable(targets, comp, lo, hi, group=group, name=name))
        return self

    def presence(self, indices: Any = None, group: str = "presence") -> DesignSpaceBuilder:
        """Add one presence variable per selected slot (variable crease count)."""
        chosen = _resolve_indices(indices, len(self.base_creases))
        for ci in chosen:
            self._variables.append(_presence_variable(ci, group=group))
        return self

    def select(self, **kwargs: Any) -> list[int]:
        """Return indices of the currently added variables matching a selection.

        Accepts the same keyword filters as :meth:`DesignSpace.select`.
        """
        tmp = DesignSpace(
            base_creases=self.base_creases,
            variables=list(self._variables) or [_angle_variable(0, -1.0, 1.0)],
        )
        return tmp.select(**kwargs)

    def constrain(
        self, equalities: Iterable[tuple[int, int]] = (), symmetries: Iterable[tuple[int, int]] = ()
    ) -> DesignSpaceBuilder:
        """Store equality / symmetry constraints applied by :meth:`build`."""
        self._equalities = list(equalities)
        self._symmetries = list(symmetries)
        return self

    def build(
        self,
        name: str = "origami",
        angle_limits: tuple[float, float] | None = None,
        coord_limits: tuple[float, float] | None = None,
        count_mode: str = "fixed",
        count_bounds: tuple[int, int] | None = None,
    ) -> DesignSpace:
        """Build the :class:`DesignSpace` (applying any constraints)."""
        if not self._variables:
            raise ValueError("builder has no variables; add angles, geometry or presence first")
        space = DesignSpace(
            base_creases=self.base_creases,
            variables=list(self._variables),
            name=name,
            count_mode=count_mode,
            count_bounds=count_bounds,
            presence_threshold=self._presence_threshold,
            angle_limits=angle_limits or self._angle_limits,
            coord_limits=coord_limits or self._coord_limits,
        )
        equalities = getattr(self, "_equalities", [])
        symmetries = getattr(self, "_symmetries", [])
        if equalities or symmetries:
            space = space.with_constraints(equalities=equalities, symmetries=symmetries)
        return space
