from __future__ import annotations

"""Population initialization / seeding helpers (no optimization algorithm).

Everything here *produces* an initial population for an external optimizer; it
ships no mutation, crossover, selection or search.  The helpers cover:

* **Seed loading** — read crease-pattern JSON files (a single file or a whole
  directory), encode them into the design space and report rejected files.
* **Interpolation** — linear interpolation between two seeds, a piecewise path
  through several seeds, and random barycentric combinations.  Compatibility is
  handled explicitly: same topology works directly; seeds whose topology cannot
  be encoded into the space are rejected with a reason (never silently mangled).
* **Random initialization** — uniform, Gaussian perturbation, Latin-hypercube
  and (when SciPy is installed) Sobol sampling.
* **Mixed populations** — e.g. 20% seeds / 50% interpolation / 30% random.
* **Validity-aware generation and de-duplication**, plus diversity-aware
  farthest-point selection.
* **Population I/O** — JSON or ``.npz`` with provenance and the space spec, and
  an export directory of standard crease JSON files.
* **Statistics** — :func:`population_stats`.

The design space is duck-typed: it only needs ``dim``, ``lower``, ``upper``,
``sample``, ``encode``, ``is_valid``, ``to_normalized`` and ``from_normalized``
(all provided by ``SearchSpace`` and ``DesignSpace``).  The Gaussian initializer
additionally uses a centre genome — ``DesignSpace.base_vector()`` or, for a
``SearchSpace``, the ``base_angles``/``active_indices`` slice (falling back to
the bounds midpoint).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..serialization import load_creases_from_json, save_creases_to_json
from .design_space import DesignSpace
from .search_space import SearchSpace

__all__ = [
    "Population",
    "SeedSet",
    "deduplicate",
    "diversity_filter",
    "export_creases_dir",
    "gaussian_population",
    "interpolate",
    "jitter",
    "lhs_population",
    "load_population",
    "load_seeds",
    "mixed_population",
    "piecewise_path",
    "population_stats",
    "random_barycentric",
    "random_population",
    "save_population",
    "sobol_population",
]

Space = SearchSpace | DesignSpace


def _as_rng(rng) -> np.random.Generator:
    """Coerce ``None`` / ``int`` / :class:`numpy.random.Generator` to a Generator."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


@dataclass
class SeedSet:
    """Result of :func:`load_seeds`.

    Attributes
    ----------
    genomes : numpy.ndarray
        ``(n, dim)`` encoded genomes for the accepted seeds.
    sources : list[str]
        Paths of the accepted seed files, in ``genomes`` order.
    rejected : list[tuple[str, str]]
        ``(path, reason)`` for every seed that could not be encoded.
    """

    genomes: np.ndarray
    sources: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.genomes.shape[0])


@dataclass
class Population:
    """An initial population plus provenance.

    Attributes
    ----------
    genomes : numpy.ndarray
        ``(n, dim)`` genome matrix.
    space : Space | None
        The design space the genomes live in.
    provenance : list[dict]
        One dict per genome describing how it was produced (``method`` and any
        source indices).
    name : str
        Optional label.
    """

    genomes: np.ndarray
    space: Space | None = None
    provenance: list[dict] = field(default_factory=list)
    name: str = "population"

    def __len__(self) -> int:
        return int(self.genomes.shape[0])

    @property
    def dim(self) -> int:
        """Genome dimensionality (0 for an empty population)."""
        return int(self.genomes.shape[1]) if self.genomes.ndim == 2 else 0

    def __getitem__(self, item):
        return self.genomes[item]


# ----------------------------------------------------------------------
# Seed loading
# ----------------------------------------------------------------------


def _iter_seed_files(sources) -> list[Path]:
    if isinstance(sources, (str, Path)):
        sources = [sources]
    files: list[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            files.extend(sorted(q for q in p.glob("*.json")))
        else:
            files.append(p)
    return files


def load_seeds(sources, space: Space) -> SeedSet:
    """Load crease-pattern JSON seeds and encode them into ``space``.

    Parameters
    ----------
    sources :
        A file path, a directory (every ``*.json`` is read) or a list of either.
    space :
        Design space used to encode each seed.  A seed whose crease topology
        does not match the space is rejected with a reason rather than raising.

    Returns
    -------
    SeedSet
        Accepted genomes/sources and a ``(path, reason)`` reject list.
    """
    genomes: list[np.ndarray] = []
    accepted: list[str] = []
    rejected: list[tuple[str, str]] = []
    for path in _iter_seed_files(sources):
        try:
            creases = load_creases_from_json(path)
        except Exception as exc:  # noqa: BLE001 - a bad seed must not abort loading
            rejected.append((str(path), f"read error: {type(exc).__name__}: {exc}"))
            continue
        try:
            genomes.append(np.asarray(space.encode(creases), dtype=float))
        except Exception as exc:  # noqa: BLE001 - incompatible topology is a per-seed reject
            rejected.append((str(path), f"encode error: {type(exc).__name__}: {exc}"))
            continue
        accepted.append(str(path))
    arr = np.asarray(genomes, dtype=float) if genomes else np.zeros((0, space.dim), dtype=float)
    return SeedSet(genomes=arr, sources=accepted, rejected=rejected)


# ----------------------------------------------------------------------
# Interpolation
# ----------------------------------------------------------------------


def interpolate(a: Sequence[float], b: Sequence[float], t: float | Sequence[float]) -> np.ndarray:
    """Linear interpolation ``(1 - t) * a + t * b`` (``t`` may be a vector)."""
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"cannot interpolate genomes of shapes {a_arr.shape} and {b_arr.shape}")
    t_arr = np.asarray(t, dtype=float)
    if t_arr.ndim == 0:
        return (1.0 - float(t_arr)) * a_arr + float(t_arr) * b_arr
    t_col = t_arr.reshape(-1, 1)
    return (1.0 - t_col) * a_arr[None, :] + t_col * b_arr[None, :]


def piecewise_path(seeds: np.ndarray, n: int, rng=None, closed: bool = False) -> np.ndarray:
    """Sample ``n`` genomes along the polyline through ``seeds``.

    Seeds are visited in order (optionally looping back to the first when
    ``closed``).  All seeds must share a topology; they are interpolated in
    genome space, so a coordinate that differs between seeds is interpolated
    too.  ``n`` is the number of output rows.
    """
    S = np.asarray(seeds, dtype=float)
    if S.ndim != 2 or S.shape[0] < 2:
        raise ValueError("piecewise_path needs at least two seeds of shape (m, dim)")
    g = _as_rng(rng)
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    pts = np.vstack([S, S[:1]]) if closed else S
    m = pts.shape[0]
    # Random parameter in [0, m-1] (or [0, m) when closed).
    hi = float(m) if closed else float(m - 1)
    u = g.uniform(0.0, hi, size=n)
    seg = np.clip(np.floor(u).astype(int), 0, m - 2)
    frac = u - seg
    out = (1.0 - frac)[:, None] * pts[seg] + frac[:, None] * pts[seg + 1]
    return out


def random_barycentric(seeds: np.ndarray, n: int, rng=None) -> np.ndarray:
    """Random convex combinations of ``seeds`` (Dirichlet weights).

    Weights are drawn from a symmetric Dirichlet distribution, so each output
    genome is a convex combination of the seed genomes.
    """
    S = np.asarray(seeds, dtype=float)
    if S.ndim != 2 or S.shape[0] < 1:
        raise ValueError("random_barycentric needs seeds of shape (m, dim)")
    g = _as_rng(rng)
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    weights = g.dirichlet(np.ones(S.shape[0]), size=n)
    return weights @ S


# ----------------------------------------------------------------------
# Jitter
# ----------------------------------------------------------------------


def jitter(genomes, sigma: float = 0.05, space: Space | None = None, rng=None, normalized: bool = False) -> np.ndarray:
    """Add Gaussian noise to genomes, optionally clipping to the space bounds.

    ``sigma`` is in units of the space span when ``normalized=True`` (noise is
    added in the ``[-1, 1]`` view), otherwise in raw genome units.  When
    ``space`` is given the result is clipped back into bounds.
    """
    X = np.atleast_2d(np.asarray(genomes, dtype=float))
    g = _as_rng(rng)
    if normalized and space is not None:
        Z = np.vstack([space.to_normalized(row) for row in X])
        Z = Z + g.normal(0.0, float(sigma), size=Z.shape)
        Z = np.clip(Z, -1.0, 1.0)
        return np.vstack([space.from_normalized(row) for row in Z])
    Y = X + g.normal(0.0, float(sigma), size=X.shape)
    if space is not None:
        Y = np.clip(Y, space.lower, space.upper)
    return Y


# ----------------------------------------------------------------------
# Random initialization
# ----------------------------------------------------------------------


def _lhs(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Latin-hypercube sample in ``[-1, 1]^dim``."""
    if n < 1:
        return np.zeros((0, dim), dtype=float)
    cut = (np.arange(n)[:, None] + rng.random((n, dim))) / n
    for j in range(dim):
        rng.shuffle(cut[:, j])
    return 2.0 * cut - 1.0


def lhs_population(space: Space, n: int, rng=None) -> np.ndarray:
    """Latin-hypercube sample mapped into the design space."""
    g = _as_rng(rng)
    Z = _lhs(int(n), space.dim, g)
    return np.vstack([space.from_normalized(row) for row in Z]) if len(Z) else Z.reshape(0, space.dim)


def sobol_population(space: Space, n: int, rng=None, scramble: bool = True) -> np.ndarray:
    """Sobol' sample mapped into the design space (SciPy if available).

    Falls back to :func:`lhs_population` when SciPy is not installed, so the
    function always returns a population (the fallback is not flagged, but the
    docstring is explicit that Sobol' needs SciPy).
    """
    try:
        from scipy.stats import qmc  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - SciPy is optional
        return lhs_population(space, n, rng=rng)
    g = _as_rng(rng)
    seed = int(g.integers(0, 2**31 - 1))
    sampler = qmc.Sobol(d=space.dim, scramble=scramble, seed=seed)
    unit = sampler.random(int(n))
    Z = 2.0 * unit - 1.0
    return np.vstack([space.from_normalized(row) for row in Z]) if len(Z) else Z.reshape(0, space.dim)


def _space_base(space: Space) -> np.ndarray:
    """Return a sensible centre genome for ``space`` (base vector or bounds mid)."""
    if hasattr(space, "base_vector"):
        return np.asarray(space.base_vector(), dtype=float)
    if hasattr(space, "base_angles") and hasattr(space, "active_indices"):
        return np.asarray([space.base_angles[i] for i in space.active_indices], dtype=float)
    return 0.5 * (np.asarray(space.lower, dtype=float) + np.asarray(space.upper, dtype=float))


def gaussian_population(space: Space, n: int, sigma: float = 0.25, rng=None) -> np.ndarray:
    """Gaussian perturbation around the space's base genome (clipped to bounds).

    The centre is ``space.base_vector()`` for a :class:`DesignSpace`, the
    ``base_angles``/``active_indices`` slice for a :class:`SearchSpace`, and the
    midpoint of the bounds otherwise.  ``sigma`` is a fraction of each
    variable's span.
    """
    g = _as_rng(rng)
    base = _space_base(space)
    span = np.asarray(space.upper, dtype=float) - np.asarray(space.lower, dtype=float)
    X = base[None, :] + g.normal(0.0, 1.0, size=(int(n), space.dim)) * float(sigma) * span[None, :]
    return np.clip(X, space.lower, space.upper)


def random_population(space: Space, n: int, rng=None, method: str = "uniform") -> np.ndarray:
    """Draw ``n`` random genomes with the requested ``method``.

    ``method`` is one of ``"uniform"``, ``"gaussian"``, ``"lhs"`` or
    ``"sobol"`` (Sobol' falls back to LHS without SciPy).
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if method == "uniform":
        return space.sample(rng, size=n)
    if method == "gaussian":
        return gaussian_population(space, n, rng=rng)
    if method == "lhs":
        return lhs_population(space, n, rng=rng)
    if method == "sobol":
        return sobol_population(space, n, rng=rng)
    raise ValueError("method must be one of 'uniform', 'gaussian', 'lhs', 'sobol'")


# ----------------------------------------------------------------------
# Validity / dedup / diversity
# ----------------------------------------------------------------------


def deduplicate(genomes, space: Space | None = None, ndigits: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Drop duplicate genomes, keeping first occurrences.

    When ``space`` is given, duplicates are detected on the normalized ``[-1, 1]``
    view (so angle and geometry variables are compared consistently); the
    *returned* rows are always the original genome-space rows.  Returns
    ``(unique_genomes, keep_indices)``.
    """
    X = np.atleast_2d(np.asarray(genomes, dtype=float))
    rows = np.vstack([space.to_normalized(row) for row in X]) if space is not None else X
    seen: set[tuple] = set()
    keep: list[int] = []
    for i, row in enumerate(rows):
        key = tuple(np.round(row, ndigits).tolist())
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return X[keep], np.asarray(keep, dtype=int)


def diversity_filter(genomes, k: int, space: Space | None = None, rng=None) -> np.ndarray:
    """Farthest-point selection of ``k`` genomes (diversity-aware).

    Distances are computed in the normalized ``[-1, 1]`` view when ``space`` is
    given (so angle and geometry variables are comparable), otherwise in raw
    genome units.  Returns the selected genome rows.
    """
    X = np.atleast_2d(np.asarray(genomes, dtype=float))
    n = X.shape[0]
    k = int(k)
    if k >= n:
        return X.copy()
    if k <= 0:
        return X[:0].copy()
    # Distances are measured in the normalized view when a space is given, but
    # the *returned* rows are always the original genome-space rows.
    D = np.vstack([space.to_normalized(row) for row in X]) if space is not None else X
    g = _as_rng(rng)
    first = int(g.integers(0, n))
    selected = [first]
    dist = np.linalg.norm(D - D[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(D - D[nxt], axis=1))
    return X[selected]


# ----------------------------------------------------------------------
# Mixed populations
# ----------------------------------------------------------------------


def _split_counts(n: int, fractions: dict[str, float]) -> dict[str, int]:
    keys = list(fractions)
    total = float(sum(fractions.values()))
    if total <= 0:
        raise ValueError("fractions must sum to a positive value")
    raw = {k: round(n * fractions[k] / total) for k in keys}
    # Fix rounding so the parts sum exactly to n.
    diff = n - sum(raw.values())
    order = sorted(keys, key=lambda k: fractions[k], reverse=True)
    i = 0
    while diff != 0 and order:
        k = order[i % len(order)]
        raw[k] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        i += 1
    return raw


def mixed_population(
    space: Space,
    n: int,
    seeds: np.ndarray | None = None,
    fractions: dict[str, float] | None = None,
    rng=None,
    method: str = "uniform",
    valid_only: bool = False,
    dedup: bool = True,
) -> Population:
    """Build a mixed initial population from seeds / interpolation / random.

    Parameters
    ----------
    fractions :
        Mapping like ``{"seed": 0.2, "interp": 0.5, "random": 0.3}``.  Keys are
        a subset of ``seed``, ``interp``, ``random`` (others are ignored);
        values are relative weights and are normalized to ``n``.
    seeds :
        Seed genomes (``(m, dim)``) used by the ``seed`` and ``interp`` parts.
        Required when those fractions are non-zero.
    valid_only :
        Drop genomes the space reports as invalid (via ``space.is_valid``).
    dedup :
        Remove duplicate genomes after generation.

    Returns
    -------
    Population
        Genomes plus per-genome provenance (``method`` and, for the interpolated
        part, the source seed indices).
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    g = _as_rng(rng)
    if fractions is None:
        fractions = {"seed": 0.2, "interp": 0.5, "random": 0.3}
    frac = {k: v for k, v in fractions.items() if k in {"seed", "interp", "random"} and v > 0}
    if not frac:
        frac = {"random": 1.0}
    if ("seed" in frac or "interp" in frac) and (seeds is None or len(np.atleast_2d(seeds)) == 0):
        # No seeds available: fall back to random rather than raising.
        frac = {"random": 1.0}
    counts = _split_counts(n, frac)

    parts: list[np.ndarray] = []
    prov: list[dict] = []

    def _emit(arr: np.ndarray, provenance: list[dict]) -> None:
        if arr is None or len(arr) == 0:
            return
        parts.append(arr)
        prov.extend(provenance)

    if "seed" in counts and counts["seed"] > 0:
        S = np.atleast_2d(np.asarray(seeds, dtype=float))
        idx = g.integers(0, S.shape[0], size=counts["seed"])
        _emit(S[idx], [{"method": "seed", "seed_index": int(j)} for j in idx])
    if "interp" in counts and counts["interp"] > 0:
        S = np.atleast_2d(np.asarray(seeds, dtype=float))
        if S.shape[0] >= 2:
            arr = piecewise_path(S, counts["interp"], rng=g)
        else:
            arr = random_barycentric(S, counts["interp"], rng=g)
        _emit(arr, [{"method": "interp"} for _ in range(len(arr))])
    if "random" in counts and counts["random"] > 0:
        arr = random_population(space, counts["random"], rng=g, method=method)
        _emit(arr, [{"method": "random", "sampler": method} for _ in range(len(arr))])

    genomes = np.vstack(parts) if parts else np.zeros((0, space.dim), dtype=float)

    if valid_only and len(genomes):
        keep = [i for i in range(len(genomes)) if space.is_valid(genomes[i])]
        genomes = genomes[keep]
        prov = [prov[i] for i in keep]
    if dedup and len(genomes):
        genomes, keep_idx = deduplicate(genomes, space=space)
        prov = [prov[int(i)] for i in keep_idx]
    # Shuffle so the parts are not contiguous.
    if len(genomes):
        order = g.permutation(len(genomes))
        genomes = genomes[order]
        prov = [prov[int(i)] for i in order]
    return Population(genomes=genomes, space=space, provenance=prov, name="mixed")


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------


def save_population(
    path, population: Population | np.ndarray, space: Space | None = None, meta: dict | None = None
) -> Path:
    """Save a population to JSON or ``.npz`` (suffix selects the format).

    The payload records the genomes, per-genome provenance (when a
    :class:`Population` is given) and, when ``space`` is provided, the space
    spec (:meth:`DesignSpace.to_dict` / :meth:`SearchSpace.to_dict`).
    """
    if isinstance(population, Population):
        genomes = population.genomes
        provenance = population.provenance
        space = space if space is not None else population.space
    else:
        genomes = np.atleast_2d(np.asarray(population, dtype=float))
        provenance = []
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = None
    if space is not None and hasattr(space, "to_dict"):
        spec = space.to_dict()
    if out.suffix.lower() == ".npz":
        np.savez(
            out,
            genomes=genomes,
            provenance=json.dumps(provenance),
            space_spec=json.dumps(spec),
            meta=json.dumps(dict(meta or {})),
        )
        return out
    payload = {
        "genomes": [[float(x) for x in row] for row in genomes],
        "provenance": provenance,
        "space": spec,
        "meta": dict(meta or {}),
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def load_population(path, space: Space | None = None) -> Population:
    """Load a population written by :func:`save_population`.

    When ``space`` is ``None`` the stored space spec is used if present (a
    :class:`DesignSpace`), otherwise the population has ``space=None``.
    """
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as data:
            genomes = np.asarray(data["genomes"], dtype=float)
            provenance = json.loads(str(data["provenance"]))
            spec = json.loads(str(data["space_spec"]))
            meta = json.loads(str(data["meta"]))
    else:
        payload = json.loads(p.read_text())
        genomes = np.asarray(payload["genomes"], dtype=float).reshape(-1, _infer_dim(payload))
        provenance = payload.get("provenance", [])
        spec = payload.get("space")
        meta = payload.get("meta", {})
    if space is None and isinstance(spec, dict) and "variables" in spec and "base_creases" in spec:
        # ``save_population`` stores whichever ``to_dict`` the space exposes.  A
        # DesignSpace spec round-trips; a SearchSpace spec has no ``from_dict``,
        # so it is kept as metadata and ``space`` stays ``None``.
        space = DesignSpace.from_dict(spec)
    return Population(genomes=genomes, space=space, provenance=provenance, name=str(meta.get("name", "population")))


def _infer_dim(payload: dict) -> int:
    genomes = payload.get("genomes") or []
    if genomes:
        return len(genomes[0])
    spec = payload.get("space") or {}
    return int(spec.get("dim", 0))


def export_creases_dir(directory, population: Population | np.ndarray, space: Space) -> list[Path]:
    """Write one standard crease-pattern JSON per genome into ``directory``.

    Returns the list of written paths.  Useful for handing a population to the
    CLI/other tools that read the crease JSON format.
    """
    if isinstance(population, Population):
        genomes = population.genomes
    else:
        genomes = np.atleast_2d(np.asarray(population, dtype=float))
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, row in enumerate(genomes):
        creases = space.decode(row)
        paths.append(save_creases_to_json(creases, out / f"genome_{i:05d}.json", angle_unit="radian"))
    return paths


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------


def population_stats(genomes, space: Space | None = None) -> dict[str, Any]:
    """Summary statistics of a population matrix.

    Returns ``n``, ``dim``, per-variable ``mean``/``std``/``min``/``max``, the
    mean pairwise distance (in the normalized view when ``space`` is given) and
    the fraction of valid genomes when ``space`` exposes ``is_valid``.
    """
    X = np.atleast_2d(np.asarray(genomes, dtype=float))
    if np.asarray(genomes, dtype=float).size == 0:
        # ``np.atleast_2d([])`` would report a phantom ``(1, 0)`` row; keep the
        # declared dimension for an empty ``(0, dim)`` input.
        arr = np.asarray(genomes, dtype=float)
        X = arr.reshape(0, arr.shape[1] if arr.ndim == 2 else 0)
    n, dim = X.shape if X.ndim == 2 else (0, 0)
    stats: dict[str, Any] = {
        "n": int(n),
        "dim": int(dim),
        "mean": [float(v) for v in X.mean(axis=0)] if n else [],
        "std": [float(v) for v in X.std(axis=0)] if n else [],
        "min": [float(v) for v in X.min(axis=0)] if n else [],
        "max": [float(v) for v in X.max(axis=0)] if n else [],
    }
    if n and space is not None and hasattr(space, "is_valid"):
        stats["valid_fraction"] = float(np.mean([bool(space.is_valid(row)) for row in X]))
    else:
        stats["valid_fraction"] = None
    if n >= 2:
        Y = np.vstack([space.to_normalized(row) for row in X]) if space is not None else X
        d = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=-1)
        stats["mean_pairwise_distance"] = float(d[np.triu_indices(n, k=1)].mean())
    else:
        stats["mean_pairwise_distance"] = 0.0
    return stats
