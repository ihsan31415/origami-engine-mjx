from __future__ import annotations

"""Evaluator / objective interface for optimization pipelines.

:class:`OrigamiEvaluator` adapts the fixed-topology
:class:`~origami_engine_mjx.integrations.search_space.SearchSpace` and the
existing :class:`~origami_engine_mjx.engine.OrigamiEngine` facade into a
stateless, picklable objective that any optimizer (ES, CMA-ES, Bayesian
optimization, MAP-Elites, …) can call.  It contains **no** search algorithm.

Contract
--------
* Input: one genome ``(dim,)`` or a batch ``(N, dim)`` of NumPy arrays (a JAX
  array also works on the JAX backend because it is coerced through
  :func:`numpy.asarray`).
* Output: a scalar *fitness* per genome (``float`` for one, ``(N,)`` array for
  a batch) that is **always higher-is-better**.  ``fitness = raw`` when
  ``maximize=True`` and ``fitness = -raw`` when ``maximize=False``, so a
  minimizer that negates it (``-fitness``) optimizes in the requested
  direction.  Use :meth:`OrigamiEvaluator.to_minimizer` /
  :meth:`OrigamiEvaluator.to_maximizer` for the two conventions.
* Per-genome failures — an invalid crease pattern, a simulation exception, or a
  non-finite score — are reported as a ``status`` string plus the documented
  ``penalty`` value rather than aborting the batch.  Set
  ``error_policy="raise"`` to re-raise decode/simulation *exceptions* instead;
  non-finite scores are still reported as ``"nonfinite"`` (never raised), and a
  wrong-length genome always raises ``ValueError`` regardless of the policy
  (it is a programming error, not a per-genome result).

Reproducibility
---------------
The physics is deterministic: identical creases, duration and vibration give
byte-identical scores on a given backend (verified on the JAX backend).  The
only stochastic part of a pipeline is genome sampling, which is controlled by
the caller's ``rng`` / seed; the evaluator itself never draws randomness.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..constants import VIB_AMP_X, VIB_AMP_Z, VIB_FREQ
from ..engine import OrigamiEngine, compute_forward_displacement
from ..fitness import FitnessWeights, blend_scores, fold_report
from ..types import BackendName
from .design_space import DesignSpace
from .search_space import SearchSpace

__all__ = [
    "DESCRIPTOR_NAMES",
    "BatchResult",
    "EvalRecord",
    "OrigamiEvaluator",
    "topology_signature",
]


def topology_signature(creases, ndigits: int = 9) -> tuple:
    """Return a hashable signature of a crease pattern's *geometry*.

    Two crease lists with the same signature produce the same MJX model (the
    face graph depends only on the endpoints, not the fold angles), so the
    evaluator buckets genomes with this key before calling the backend.
    """
    return tuple(
        (
            round(float(c.p1[0]), ndigits),
            round(float(c.p1[1]), ndigits),
            round(float(c.p2[0]), ndigits),
            round(float(c.p2[1]), ndigits),
        )
        for c in creases
    )


#: Behavior descriptors returned alongside fitness in the structured path.
#: They are computed from the final simulation state and let quality-diversity
#: pipelines (e.g. MAP-Elites) bin solutions.
DESCRIPTOR_NAMES = (
    "displacement_x",
    "displacement_y",
    "displacement_norm",
    "direction_x",
    "direction_y",
    "final_x",
    "final_y",
    "fold_amount",
    "fold_score",
    "fold_mean_abs_error_deg",
)


@dataclass
class EvalRecord:
    """Structured result for a single genome.

    Attributes
    ----------
    fitness : float
        Signed value optimizers consume (``raw`` or ``-raw``).
    raw : float
        Raw forward displacement (metres).  On the structured path this is
        projected onto the evaluator ``trajectory``; the fitness-only fast path
        delegates to the backends, which always score the +x displacement.
    status : str
        ``"ok"``, ``"invalid"`` (decode failed), ``"nonfinite"`` or ``"error"``.
    error : str
        Human-readable reason when ``status != "ok"``.
    time : float
        Wall time for this genome in seconds (0.0 when served from cache, and on
        the fitness-only fast path where per-genome time is not separable).
    descriptors : dict[str, float]
        Behavior descriptors (see :data:`DESCRIPTOR_NAMES`); empty in the
        fitness-only fast path.
    cached : bool
        Whether the record was served from the evaluator cache.
    path : str
        Which evaluation path produced the record: ``"batched"`` (one backend
        call for a same-topology batch), ``"bucketed"`` (a topology bucket of
        more than one individual), ``"single"`` (one individual / one backend
        call), ``"invalid"`` (decode failed) or ``"cached"`` (served from the
        evaluator cache).
    """

    fitness: float
    raw: float
    status: str = "ok"
    error: str = ""
    time: float = 0.0
    descriptors: dict[str, float] = field(default_factory=dict)
    cached: bool = False
    path: str = ""


@dataclass
class BatchResult:
    """Vectorized result of :meth:`OrigamiEvaluator.evaluate_batch`.

    All arrays are indexed by genome and have length ``N``.
    """

    fitness: np.ndarray
    raw: np.ndarray
    status: list[str]
    time: np.ndarray
    descriptors: dict[str, np.ndarray]
    wall_time: float
    n_cache_hits: int = 0
    """Cumulative cache hits on the evaluator, not just this batch."""
    errors: list[str] = field(default_factory=list)
    """Per-genome error text (``""`` when ``status == "ok"``)."""
    cached: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    """Per-genome cache-hit flags."""
    paths: list[str] = field(default_factory=list)
    """Per-genome evaluation path (see :attr:`EvalRecord.path`)."""

    @property
    def n(self) -> int:
        """Number of genomes in the batch."""
        return int(self.fitness.shape[0])

    @property
    def ok(self) -> np.ndarray:
        """Boolean mask of genomes that evaluated successfully."""
        return np.asarray([s == "ok" for s in self.status])

    def best_index(self) -> int:
        """Index of the best (highest signed fitness) genome in the batch."""
        return int(np.argmax(self.fitness))

    def as_records(self) -> list[EvalRecord]:
        """Return the batch as a list of :class:`EvalRecord`."""
        out = []
        for i in range(self.n):
            out.append(
                EvalRecord(
                    fitness=float(self.fitness[i]),
                    raw=float(self.raw[i]),
                    status=self.status[i],
                    error=self.errors[i] if i < len(self.errors) else "",
                    time=float(self.time[i]),
                    descriptors={k: float(v[i]) for k, v in self.descriptors.items()},
                    cached=bool(self.cached[i]) if i < len(self.cached) else False,
                    path=self.paths[i] if i < len(self.paths) else "",
                )
            )
        return out


class OrigamiEvaluator:
    """Stateless, picklable objective mapping genomes to signed fitness.

    Parameters
    ----------
    space : SearchSpace | DesignSpace
        Genome specification; defines the topology (fixed or variable) and the
        per-variable bounds.  A
        :class:`~origami_engine_mjx.integrations.design_space.DesignSpace` lets
        the topology change between individuals (endpoints and crease presence
        may vary); the evaluator buckets by
        :func:`topology_signature` and reports the path per individual.
    backend : {"jax", "warp"}
        Simulation backend used for every evaluation.
    duration, fold_duration, vibrate, amp_x, amp_z, freq_hz :
        Simulation parameters forwarded to :meth:`OrigamiEngine.run`.
    trajectory : tuple
        Direction the forward displacement is projected onto.  The built-in
        backends currently score the +x displacement regardless, so this only
        affects :func:`origami_engine_mjx.engine.compute_forward_displacement`
        used for the single-genome structured path.
    maximize : bool
        ``True`` (default) makes higher ``raw`` displacement better
        (``fitness = raw``); ``False`` makes lower displacement better
        (``fitness = -raw``).  Fitness is always higher-is-better.
    penalty : float | None
        Value returned for failed genomes.  Defaults to ``-1e6`` (the worst
        fitness, since fitness is always higher-is-better).
    error_policy : {"penalty", "raise"}
        Behaviour on per-genome failure.  ``"penalty"`` (default) converts
        decode/simulation failures into penalty records; ``"raise"`` re-raises
        those exceptions.  Non-finite scores always become ``"nonfinite"``
        records, and a wrong-length genome always raises ``ValueError``.
    batch_size : int | None
        Forwarded to :meth:`OrigamiEngine.evaluate_population`.  The Warp backend
        scores a whole population in one batched GPU model, so this bounds the
        chunk it sends at once; ``None`` (default) means "one chunk per batch".
        See ``docs/OPTIMIZATION_ES.md``.
    return_descriptors : bool
        When ``True``, :meth:`evaluate_batch` always uses the slower structured
        path so :data:`DESCRIPTOR_NAMES` are populated.
    use_cache : bool
        Cache results keyed by the exact genome bytes.  Useful for
        evolutionary loops that re-evaluate elites.  ``n_evaluations`` counts
        cache misses only.
    fold_weight, movement_weight, movement_scale : float
        Relative weights of the hinge-target ("fold") and movement components
        of the fitness, see :class:`~origami_engine_mjx.fitness.FitnessWeights`.
        ``fold_weight=0`` (default) means *movement only*: the score is exactly
        the historical displacement.  ``movement_weight=70, fold_weight=30``
        optimises a 70 % / 30 % blend.  ``movement_scale`` divides movement
        before blending (default ``1.0``, the sheet side).
    actuated :
        Which hinges are driven — ``None`` (all, default), a bool, or one bool
        per hinge of the decoded crease pattern.
    """

    def __init__(
        self,
        space: SearchSpace | DesignSpace,
        backend: BackendName = "jax",
        duration: float = 8.0,
        fold_duration: float = 3.0,
        vibrate: bool = True,
        amp_x: float = VIB_AMP_X,
        amp_z: float = VIB_AMP_Z,
        freq_hz: float = VIB_FREQ,
        trajectory: tuple[float, ...] = (1.0, 0.0, 0.0),
        maximize: bool = True,
        penalty: float | None = None,
        error_policy: str = "penalty",
        batch_size: int | None = None,
        return_descriptors: bool = False,
        use_cache: bool = False,
        fold_weight: float = 0.0,
        movement_weight: float = 1.0,
        movement_scale: float = 1.0,
        actuated=None,
    ) -> None:
        if error_policy not in {"penalty", "raise"}:
            raise ValueError("error_policy must be 'penalty' or 'raise'")
        # Validates the weights (non-negative, not both zero, positive scale).
        self.weights = FitnessWeights(movement_weight, fold_weight, movement_scale)
        self.space = space
        self.backend: BackendName = backend
        self.duration = float(duration)
        self.fold_duration = float(fold_duration)
        self.vibrate = bool(vibrate)
        self.amp_x = float(amp_x)
        self.amp_z = float(amp_z)
        self.freq_hz = float(freq_hz)
        self.trajectory = tuple(float(t) for t in trajectory)
        self.maximize = bool(maximize)
        self.error_policy = error_policy
        self.batch_size = batch_size
        self.return_descriptors = bool(return_descriptors)
        self.use_cache = bool(use_cache)
        self.actuated = actuated
        if penalty is None:
            penalty = -1e6
        self.penalty = float(penalty)

        self.n_evaluations = 0
        self.n_cache_hits = 0
        self._cache: dict[bytes, EvalRecord] = {}
        # Lazily built engine (see ``_engine``); excluded from pickling.
        self._engine_cache: OrigamiEngine | None = None

    def __getstate__(self) -> dict[str, object]:
        # Drop the cached engine: a pickled evaluator must not carry a GPU
        # model or captured graph across a process boundary.
        state = dict(self.__dict__)
        state["_engine_cache"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._engine_cache = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def fold_weight(self) -> float:
        """Weight of the hinge-target component (``0`` = movement only)."""
        return self.weights.fold

    @property
    def movement_weight(self) -> float:
        """Weight of the movement component."""
        return self.weights.movement

    @property
    def movement_fraction(self) -> float:
        """Share of the fitness that comes from movement, in ``[0, 1]``."""
        return self.weights.normalized()[0]

    @property
    def fold_fraction(self) -> float:
        """Share of the fitness that comes from folding to target, in ``[0, 1]``."""
        return self.weights.normalized()[1]

    @property
    def dim(self) -> int:
        """Genome dimensionality (``space.dim``)."""
        return self.space.dim

    def clear_cache(self) -> None:
        """Drop all cached records and reset counters."""
        self._cache.clear()
        self.n_evaluations = 0
        self.n_cache_hits = 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OrigamiEvaluator(space={self.space.name!r}, dim={self.dim}, "
            f"backend={self.backend!r}, maximize={self.maximize})"
        )

    # ------------------------------------------------------------------
    # Core simulation helpers
    # ------------------------------------------------------------------

    def _engine(self) -> OrigamiEngine:
        # Built once and reused: an engine holds the backend, which (for Warp)
        # owns a compiled-model cache and a captured CUDA graph.  Rebuilding it
        # per genome would recompile the same model for every individual.  The
        # engine is created lazily so the evaluator stays trivially picklable
        # (``_engine_cache`` is not part of the pickled state).
        engine = self._engine_cache
        if engine is None:
            engine = OrigamiEngine(backend=self.backend)
            self._engine_cache = engine
        return engine

    def _simulate(self, engine: OrigamiEngine, creases) -> tuple[float, dict[str, float]]:
        """Run one simulation and return ``(raw, descriptors)``.

        ``raw`` is the movement displacement when ``fold_weight == 0`` (the
        historical behaviour), otherwise the weighted movement/fold blend.
        """
        result = engine.run(
            creases,
            duration=self.duration,
            fold_duration=self.fold_duration,
            vibrate=self.vibrate,
            amp_x=self.amp_x,
            amp_z=self.amp_z,
            freq_hz=self.freq_hz,
            actuated=self.actuated,
        )
        movement = float(compute_forward_displacement(result, trajectory=self.trajectory))
        report = fold_report(result)
        fold = float(report["score"])
        raw = blend_scores(movement, fold, self.weights) if self.weights.fold > 0 else movement
        initial = np.asarray(result.initial_centroid, dtype=float)
        final = np.asarray(result.final_centroid, dtype=float)
        delta = final - initial
        norm = float(np.linalg.norm(delta))
        direction = delta / norm if norm > 0 else np.zeros_like(delta)
        fold_amount = float(sum(abs(c.angle) for c in creases))
        descriptors = {
            "displacement_x": float(delta[0]),
            "displacement_y": float(delta[1]),
            "displacement_norm": norm,
            "direction_x": float(direction[0]),
            "direction_y": float(direction[1]),
            "final_x": float(final[0]),
            "final_y": float(final[1]),
            "fold_amount": fold_amount,
            "fold_score": fold,
            "fold_mean_abs_error_deg": float(report["mean_abs_error_deg"]),
        }
        return raw, descriptors

    def _make_record(self, raw: float, descriptors: dict[str, float], dt: float) -> EvalRecord:
        status = "ok" if np.isfinite(raw) else "nonfinite"
        if status == "ok":
            fitness = raw if self.maximize else -raw
            return EvalRecord(fitness=float(fitness), raw=float(raw), status="ok", time=dt, descriptors=descriptors)
        return EvalRecord(
            fitness=float(self.penalty),
            raw=float(raw),
            status="nonfinite",
            error=f"non-finite score: {raw}",
            time=dt,
            descriptors=descriptors,
        )

    def _failed_record(self, status: str, error: str, dt: float = 0.0, path: str = "") -> EvalRecord:
        return EvalRecord(fitness=float(self.penalty), raw=float("nan"), status=status, error=error, time=dt, path=path)

    def _evaluate_row(self, engine: OrigamiEngine, row: np.ndarray) -> EvalRecord:
        """Structured single-genome evaluation (used for descriptors)."""
        t0 = time.perf_counter()
        try:
            creases = self.space.decode(row)
        except Exception as exc:
            if self.error_policy == "raise":
                raise
            return self._failed_record("invalid", str(exc), path="invalid")
        try:
            raw, descriptors = self._simulate(engine, creases)
        except Exception as exc:
            if self.error_policy == "raise":
                raise
            return self._failed_record("error", f"{type(exc).__name__}: {exc}", time.perf_counter() - t0, path="single")
        rec = self._make_record(raw, descriptors, time.perf_counter() - t0)
        rec.path = "single"
        return rec

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    def evaluate(self, genome) -> EvalRecord:
        """Evaluate a single genome and return a full :class:`EvalRecord`."""
        row = np.asarray(genome, dtype=float).reshape(-1)
        if row.size != self.dim:
            raise ValueError(f"genome has length {row.size}, expected {self.dim}")
        key = row.tobytes()
        if self.use_cache and key in self._cache:
            self.n_cache_hits += 1
            cached = self._cache[key]
            return EvalRecord(
                fitness=cached.fitness,
                raw=cached.raw,
                status=cached.status,
                error=cached.error,
                time=0.0,
                descriptors=dict(cached.descriptors),
                cached=True,
                path="cached",
            )
        engine = self._engine()
        rec = self._evaluate_row(engine, row)
        self.n_evaluations += 1
        if self.use_cache:
            self._cache[key] = rec
        return rec

    def evaluate_batch(self, genomes, return_details: bool | None = None) -> BatchResult:
        """Evaluate a batch ``(N, dim)`` of genomes.

        Parameters
        ----------
        genomes :
            ``(N, dim)`` array-like; a single ``(dim,)`` genome is promoted to
            ``(1, dim)``.
        return_details :
            Override :attr:`return_descriptors`.  When ``False`` (and caching
            is off) the fast path calls
            :meth:`OrigamiEngine.evaluate_population` once for the whole batch
            and skips descriptors.

        Returns
        -------
        BatchResult
            Signed fitness, raw displacement, status strings, per-genome wall
            time, descriptors (empty on the fast path), error text and cache-hit
            flags.  On the fast path ``raw`` is the backend's +x displacement
            (the ``trajectory`` is validated but not applied).
        """
        X = np.asarray(genomes, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"expected genomes of shape (N, {self.dim}), got {X.shape}")
        n = X.shape[0]
        details = self.return_descriptors if return_details is None else bool(return_details)
        # The fast path only returns movement scores, so a blended objective
        # (and the cache) need the structured per-genome path.
        details = details or self.use_cache or self.weights.fold > 0  # cache needs structured records

        t_start = time.perf_counter()
        records: list[EvalRecord | None] = [None] * n
        todo: list[int] = []
        for i in range(n):
            key = np.ascontiguousarray(X[i]).tobytes()
            if self.use_cache and key in self._cache:
                self.n_cache_hits += 1
                cached = self._cache[key]
                records[i] = EvalRecord(
                    fitness=cached.fitness,
                    raw=cached.raw,
                    status=cached.status,
                    error=cached.error,
                    time=0.0,
                    descriptors=dict(cached.descriptors),
                    cached=True,
                    path="cached",
                )
            else:
                todo.append(i)

        if todo:
            engine = self._engine()
            if not details:
                self._fast_batch(engine, X, todo, records)
            else:
                for i in todo:
                    rec = self._evaluate_row(engine, X[i])
                    records[i] = rec
                    self.n_evaluations += 1
                    if self.use_cache:
                        self._cache[X[i].tobytes()] = rec

        wall = time.perf_counter() - t_start
        final = [r for r in records if r is not None]
        fitness = np.asarray([r.fitness for r in final], dtype=float)
        raw = np.asarray([r.raw for r in final], dtype=float)
        status = [r.status for r in final]
        errors = [r.error for r in final]
        cached_flags = np.asarray([r.cached for r in final], dtype=bool)
        times = np.asarray([r.time for r in final], dtype=float)
        paths = [r.path for r in final]
        descriptors: dict[str, np.ndarray] = {}
        if details:
            for name in DESCRIPTOR_NAMES:
                descriptors[name] = np.asarray([r.descriptors.get(name, np.nan) for r in final], dtype=float)
        return BatchResult(
            fitness=fitness,
            raw=raw,
            status=status,
            time=times,
            descriptors=descriptors,
            wall_time=wall,
            n_cache_hits=self.n_cache_hits,
            errors=errors,
            cached=cached_flags,
            paths=paths,
        )

    def _fast_batch(self, engine: OrigamiEngine, X: np.ndarray, todo: list[int], records: list) -> None:
        """Fitness-only path: decode all, bucket by topology, one call per bucket.

        The crease *topology* (endpoints) determines the MJX model, so genomes
        are grouped by :func:`topology_signature` and each group is sent to the
        backend in a single :meth:`OrigamiEngine.evaluate_population` call.  With
        a fixed-topology space there is one group, i.e. the historical "batched"
        fast path; a geometry/crease-count space yields several "bucketed"
        groups.  Per-individual paths are recorded on the records.
        """
        buckets: dict[tuple, list[int]] = {}
        decoded: dict[int, Any] = {}
        for i in todo:
            try:
                creases = self.space.decode(X[i])
            except Exception as exc:
                if self.error_policy == "raise":
                    raise
                records[i] = self._failed_record("invalid", str(exc), path="invalid")
                continue
            decoded[i] = creases
            buckets.setdefault(topology_signature(creases), []).append(i)
        if not buckets:
            return
        multi = len(buckets) > 1
        for members in buckets.values():
            pop = [decoded[i] for i in members]
            path = "single" if len(members) == 1 else ("bucketed" if multi else "batched")
            try:
                scores = engine.evaluate_population(
                    pop,
                    duration=self.duration,
                    trajectory=self.trajectory,
                    fold_duration=self.fold_duration,
                    vibrate=self.vibrate,
                    amp_x=self.amp_x,
                    amp_z=self.amp_z,
                    freq_hz=self.freq_hz,
                    batch_size=self.batch_size,
                    actuated=self.actuated,
                    fold_weight=self.weights.fold,
                    movement_weight=self.weights.movement,
                    movement_scale=self.weights.movement_scale,
                ).scores
            except Exception as exc:
                # Keep the fast path consistent with the structured path: a
                # population-level failure becomes a penalty record per genome.
                if self.error_policy == "raise":
                    raise
                for i in members:
                    records[i] = self._failed_record("error", f"{type(exc).__name__}: {exc}", path=path)
                    self.n_evaluations += 1
                continue
            share = 0.0  # per-genome time is not separable on this path
            for k, i in enumerate(members):
                rec = self._make_record(float(scores[k]), {}, share)
                rec.path = path
                records[i] = rec
                self.n_evaluations += 1

    # ------------------------------------------------------------------
    # Convenience callables
    # ------------------------------------------------------------------

    def __call__(self, genomes):
        """Return signed fitness: ``float`` for one genome, ``(N,)`` for a batch."""
        arr = np.asarray(genomes, dtype=float)
        if arr.ndim == 1:
            return float(self.evaluate(arr).fitness)
        return self.evaluate_batch(arr).fitness

    def raw(self, genome) -> float:
        """Return the raw (unsigned) forward displacement for one genome."""
        return float(self.evaluate(genome).raw)

    def signed(self, raw: float) -> float:
        """Apply the ``maximize`` sign switch to a raw displacement."""
        return float(raw) if self.maximize else -float(raw)

    def to_minimizer(self):
        """Return ``f(x) -> -fitness``, a scalar for minimizers (pymoo, cma, nevergrad)."""
        return lambda x: -self(x)

    def to_maximizer(self):
        """Return ``f(x) -> fitness``, a scalar for maximizers (evosax-style)."""
        return lambda x: self(x)

    def to_batch_objective(self, sense: str = "minimize"):
        """Return a batched callable ``f(X) -> (N,)`` for ``sense`` in {"minimize","maximize"}."""
        if sense == "minimize":
            return lambda X: -self.evaluate_batch(X).fitness
        if sense == "maximize":
            return lambda X: self.evaluate_batch(X).fitness
        raise ValueError("sense must be 'minimize' or 'maximize'")
