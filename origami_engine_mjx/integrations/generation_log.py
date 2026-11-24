from __future__ import annotations

"""Generation-aware run logging (no optimization algorithms).

:class:`RunLogger` records **evaluations**.  :class:`GenerationLogger` records
**generations**: it keeps the per-generation best/mean/spread, the global best
so far, and an optional best-per-generation checkpoint on disk.  The library
stays algorithm-free — *you* own the loop and call
:meth:`GenerationLogger.end_generation` once per generation::

    with GenerationLogger("log/run.jsonl") as log:            # + run.summary.json
        for gen in range(n_generations):
            genomes = ...                                      # your algorithm
            batch = evaluator.evaluate_batch(genomes)
            log.log_batch(batch, genomes, generation=gen)      # optional per-eval rows
            summary = log.end_generation(gen, batch, genomes)  # one row per gen
        print(log.best_fitness, log.best_genome)

Outputs (all optional, all side files next to ``path``):

``<path>`` (``.jsonl``/``.csv``)
    One JSON object (or CSV row) per call.  Two record kinds share the file:
    ``kind="eval"`` rows carry ``generation``/``index``/``genome`` plus the
    eval fields; ``kind="generation"`` rows carry the summary.  CSV writes a
    union header lazily, so mixing kinds stays valid.
``<stem>.summary.json``   (``summary_path=auto``, default)
    :class:`GenerationSummary` for every generation, plus a ``best`` block.
``<stem>.best.json``      (``best_path=auto``)
    :func:`~origami_engine_mjx.integrations.plumbing.save_best` checkpoint of
    the best genome of the generation that last improved the global best.
``<stem>.best_gen_<N>.json``  (``keep_every_best=True``)
    One checkpoint per generation, so every generation keeps its best
    individual.
"""

import csv
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .evaluator import BatchResult, EvalRecord
from .plumbing import save_best
from .search_space import SearchSpace

__all__ = [
    "GenerationLogger",
    "GenerationSummary",
    "load_generation_history",
]

_EVAL_FIELDS = ["kind", "generation", "index", "genome", "fitness", "raw", "status", "time"]
_GEN_FIELDS = [
    "kind",
    "generation",
    "n_individuals",
    "best_fitness",
    "best_index",
    "mean_fitness",
    "std_fitness",
    "worst_fitness",
    "median_fitness",
    "best_raw",
    "n_ok",
    "n_failed",
    "n_cache_hits",
    "best_genome",
    "global_best_fitness",
    "global_best_generation",
    "wall_time",
    "evals_so_far",
    "gen_wall_time",
]


def _fmt(value: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-friendly Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape(-1)]
    if isinstance(value, tuple):
        return [_fmt(v) for v in value]
    if isinstance(value, list):
        return [_fmt(v) for v in value]
    return value


@dataclass
class GenerationSummary:
    """Per-generation statistics (one row of the generation log).

    Attributes
    ----------
    generation : int
        Generation index as passed to :meth:`GenerationLogger.end_generation`.
    n_individuals : int
        Number of genomes evaluated in this generation.
    best_fitness : float
        Best finite fitness in this generation.
    best_index : int
        Index of the best genome inside the generation (``-1`` if none).
    mean_fitness, std_fitness, worst_fitness, median_fitness : float
        Statistics over finite fitnesses (``nan`` when none are finite).
    best_raw : float
        ``raw`` of the best individual.
    n_ok, n_failed : int
        Counts by ``status`` (``n_failed`` includes invalid/nonfinite/error).
    n_cache_hits : int
        Cache hits reported by the batch.
    best_genome : np.ndarray | None
        The best genome of this generation (a copy).
    global_best_fitness : float
        Best finite fitness seen so far in the whole run.
    global_best_generation : int
        Generation index that produced the global best.
    wall_time : float
        Cumulative logger wall time up to and including this generation.
    gen_wall_time : float
        Wall time between the previous ``end_generation`` and this one.
    evals_so_far : int
        Total evaluations logged so far.
    improved : bool
        True when this generation produced a new global best.
    """

    generation: int
    n_individuals: int
    best_fitness: float
    best_index: int
    mean_fitness: float
    std_fitness: float
    worst_fitness: float
    median_fitness: float
    best_raw: float
    n_ok: int
    n_failed: int
    n_cache_hits: int
    best_genome: np.ndarray | None
    global_best_fitness: float
    global_best_generation: int
    wall_time: float
    gen_wall_time: float
    evals_so_far: int
    improved: bool
    best_path: str | None = None
    generation_best_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the summary as a plain dict (``best_genome`` as a list)."""
        return {
            "kind": "generation",
            "generation": int(self.generation),
            "n_individuals": int(self.n_individuals),
            "best_fitness": float(self.best_fitness),
            "best_index": int(self.best_index),
            "mean_fitness": float(self.mean_fitness),
            "std_fitness": float(self.std_fitness),
            "worst_fitness": float(self.worst_fitness),
            "median_fitness": float(self.median_fitness),
            "best_raw": float(self.best_raw),
            "n_ok": int(self.n_ok),
            "n_failed": int(self.n_failed),
            "n_cache_hits": int(self.n_cache_hits),
            "best_genome": None
            if self.best_genome is None
            else [float(v) for v in np.asarray(self.best_genome).ravel()],
            "global_best_fitness": float(self.global_best_fitness),
            "global_best_generation": int(self.global_best_generation),
            "wall_time": float(self.wall_time),
            "gen_wall_time": float(self.gen_wall_time),
            "evals_so_far": int(self.evals_so_far),
            "improved": bool(self.improved),
            "best_path": self.best_path,
            "generation_best_path": self.generation_best_path,
        }


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


class GenerationLogger:
    """Record per-generation statistics and keep the best of every generation.

    Parameters
    ----------
    path : str | Path | None
        Record file (``.jsonl`` by default, ``.csv`` for CSV).  Existing content
        is truncated on first write, like :class:`RunLogger`.  ``None`` keeps
        everything in memory only.
    space : SearchSpace | None
        Needed to write genome checkpoints (``best_path`` /
        ``keep_every_best=True``).  Generation statistics do not need it.
    maximising : bool
        ``True`` (default): larger ``fitness`` is better.  ``False``: smaller is
        better (e.g. a loss).  Affects "best", ``improved`` and checkpoints
        only; fitness values are logged unmodified.
    best_path, summary_path : str | Path | "auto" | None
        Checkpoint / summary destinations.  ``"auto"`` (default) places them
        next to ``path`` as ``<stem>.best.json`` and ``<stem>.summary.json``.
        ``None`` disables them.  Without ``path`` the defaults are ``None``.
    keep_every_best : bool
        Also write ``<stem>.best_gen_<N>.json`` for **every** generation (needs
        ``space`` and a checkpoint path).
    descriptor_names : sequence[str] | None
        Descriptors to include as extra CSV columns (JSONL always includes
        every descriptor of each eval record).
    flush : bool
        Flush the record file after every write (default ``True``).

    Attributes
    ----------
    best_fitness, best_genome, best_generation : float | np.ndarray | None | int
        Global best after the most recent :meth:`end_generation`.
    history : list[GenerationSummary]
        One entry per logged generation.
    evals : list[dict]
        In-memory copy of every row written (useful in notebooks and tests).
    """

    def __init__(
        self,
        path: str | Path | None,
        space: SearchSpace | None = None,
        *,
        maximising: bool = True,
        best_path: str | Path | Literal["auto"] | None = "auto",
        summary_path: str | Path | Literal["auto"] | None = "auto",
        keep_every_best: bool = False,
        descriptor_names: Iterable[str] | None = None,
        flush: bool = True,
    ) -> None:
        self.path = None if path is None else Path(path)
        self.space = space
        self.maximising = bool(maximising)
        self.keep_every_best = bool(keep_every_best)
        self.flush = bool(flush)
        self._descriptor_names = list(descriptor_names) if descriptor_names is not None else []
        self._t0 = time.perf_counter()
        self._last_end: float | None = None
        self._fh: Any = None
        self._writer: Any = None
        self._fields: list[str] = []

        self.best_path: Path | None = self._side_path(best_path, ".best.json")
        self.summary_path: Path | None = self._side_path(summary_path, ".summary.json")
        if self.best_path is not None and self.space is None:
            # Without a space statistics still work, but genomes cannot be encoded.
            self.best_path = None
        self._gen_stem: Path | None = None if self.path is None else self.path.with_suffix("")

        self.best_fitness: float | None = None
        self.best_genome: np.ndarray | None = None
        self.best_generation: int | None = None
        self.history: list[GenerationSummary] = []
        self.evals: list[dict[str, Any]] = []
        self.evals_so_far = 0
        if self.summary_path is not None:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def _side_path(self, value: Any, suffix: str) -> Path | None:
        """Resolve ``"auto"``/``None``/explicit paths for side files."""
        if value is None:
            return None
        if value == "auto":
            if self.path is None:
                return None
            return self.path.with_suffix("").with_suffix(suffix)
        return Path(value)

    def __enter__(self) -> GenerationLogger:  # noqa: PYI034 - py3.10 has no typing.Self
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _ensure_open(self, row: Mapping[str, Any]) -> None:
        if self.path is None:
            if not self._fields:
                self._fields = list(row)
            return
        if self._fh is not None:
            if self.path.suffix.lower() == ".csv":
                self._maybe_extend_header(row)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields = list(row)
        if self.path.suffix.lower() == ".csv":
            self._fh = self.path.open("w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields, extrasaction="ignore")
            self._writer.writeheader()
        else:
            self._fh = self.path.open("w")

    def _maybe_extend_header(self, row: Mapping[str, Any]) -> None:
        extra = [k for k in row if k not in self._fields]
        if not extra:
            return
        # Rewrite the CSV with the union of columns (only happens when a new
        # record kind or descriptor appears).  Old rows keep their values.
        assert self._fh is not None
        path = self.path
        assert path is not None
        self._fh.close()
        with path.open(newline="") as old_fh:
            existing = list(csv.DictReader(old_fh))
        self._fields += extra
        self._fh = path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fields, extrasaction="ignore")
        self._writer.writeheader()
        for old in existing:
            self._writer.writerow(old)

    def _write(self, row: Mapping[str, Any]) -> None:
        self._ensure_open(row)
        self.evals.append(dict(row))
        if self.path is None or self._fh is None:
            return
        clean = {k: _fmt(v) for k, v in row.items()}
        if self.path.suffix.lower() == ".csv":
            clean = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in clean.items()}
            self._writer.writerow(clean)
        else:
            self._fh.write(json.dumps(clean) + "\n")
        if self.flush:
            self._fh.flush()

    # ------------------------------------------------------------------
    # Evaluation rows
    # ------------------------------------------------------------------

    def log(self, record: EvalRecord, genome=None, *, generation: int | None = None, index: int | None = None) -> None:
        """Append one evaluation row (optionally tagged with its generation)."""
        row: dict[str, Any] = {
            "kind": "eval",
            "generation": None if generation is None else int(generation),
            "index": None if index is None else int(index),
            "genome": None if genome is None else [float(x) for x in np.asarray(genome, dtype=float).reshape(-1)],
            "fitness": float(record.fitness),
            "raw": float(record.raw),
            "status": record.status,
            "time": float(record.time),
        }
        for k, v in record.descriptors.items():
            row.setdefault(k, float(v))
        self._write(row)

    def log_batch(self, result: BatchResult, genomes=None, *, generation: int | None = None) -> None:
        """Append every individual of a :class:`BatchResult` as an eval row."""
        genomes_arr = None if genomes is None else np.asarray(genomes, dtype=float)
        for i, rec in enumerate(result.as_records()):
            genome = None if genomes_arr is None else genomes_arr[i]
            self.log(rec, genome, generation=generation, index=i)

    # ------------------------------------------------------------------
    # Generations
    # ------------------------------------------------------------------

    def _is_better(self, a: float, b: float | None) -> bool:
        if b is None:
            return True
        return a > b if self.maximising else a < b

    def end_generation(
        self,
        generation: int,
        result: BatchResult | Sequence[EvalRecord],
        genomes=None,
    ) -> GenerationSummary:
        """Close a generation: summarise it, checkpoint its best, log one row.

        ``result`` is the batch you evaluated this generation and ``genomes``
        the matching genome array (``N x n_genes``).  Returns the
        :class:`GenerationSummary` and appends it to :attr:`history`.
        """
        records = result.as_records() if isinstance(result, BatchResult) else list(result)
        n = len(records)
        if n == 0:
            raise ValueError("end_generation needs at least one evaluation record")
        genomes_arr = None if genomes is None else np.asarray(genomes, dtype=float)
        if genomes_arr is not None and genomes_arr.shape[0] != n:
            raise ValueError(f"got {genomes_arr.shape[0]} genomes for {n} records")

        fitness = np.array([r.fitness for r in records], dtype=float)
        finite = _finite(fitness)
        # ``argmax`` on the sign-flipped array gives the argmin for minimisation.
        directed = fitness if self.maximising else -fitness
        best_index = -1
        if finite.size:
            masked = np.where(np.isfinite(directed), directed, -np.inf)
            best_index = int(np.argmax(masked))
        best_fit = float(fitness[best_index]) if best_index >= 0 else float("nan")
        best_raw = float(records[best_index].raw) if best_index >= 0 else float("nan")
        best_genome = (
            None
            if (genomes_arr is None or best_index < 0)
            else np.array(genomes_arr[best_index], dtype=float, copy=True)
        )

        n_ok = sum(1 for r in records if r.status == "ok")
        improved = bool(finite.size) and self._is_better(best_fit, self.best_fitness)
        now = time.perf_counter()
        gen_wall = 0.0 if self._last_end is None else now - self._last_end
        self._last_end = now
        self.evals_so_far += n

        summary = GenerationSummary(
            generation=int(generation),
            n_individuals=n,
            best_fitness=best_fit,
            best_index=best_index,
            mean_fitness=float(finite.mean()) if finite.size else float("nan"),
            std_fitness=float(finite.std()) if finite.size else float("nan"),
            worst_fitness=float(finite.min() if self.maximising else finite.max()) if finite.size else float("nan"),
            median_fitness=float(np.median(finite)) if finite.size else float("nan"),
            best_raw=best_raw,
            n_ok=int(n_ok),
            n_failed=int(n - n_ok),
            n_cache_hits=int(getattr(result, "n_cache_hits", 0) or 0),
            best_genome=best_genome,
            global_best_fitness=best_fit,
            global_best_generation=int(generation),
            wall_time=now - self._t0,
            gen_wall_time=gen_wall,
            evals_so_far=self.evals_so_far,
            improved=improved,
        )

        if improved:
            self.best_fitness = best_fit
            self.best_genome = best_genome
            self.best_generation = int(generation)
        summary.global_best_fitness = float(self.best_fitness) if self.best_fitness is not None else best_fit
        summary.global_best_generation = (
            int(self.best_generation) if self.best_generation is not None else int(generation)
        )

        # Checkpoints.
        if best_genome is not None and self.space is not None:
            meta = {
                "generation": int(generation),
                "best_index": int(best_index),
                "fitness": best_fit,
                "mean_fitness": summary.mean_fitness,
                "std_fitness": summary.std_fitness,
                "n_individuals": n,
                "maximising": self.maximising,
            }
            if improved and self.best_path is not None:
                summary.best_path = str(
                    save_best(best_genome, self.space, self.best_path, fitness=best_fit, extra=meta)
                )
            if self.keep_every_best and self._gen_stem is not None:
                gen_path = self._gen_stem.with_name(f"{self._gen_stem.name}.best_gen_{int(generation):04d}.json")
                summary.generation_best_path = str(
                    save_best(best_genome, self.space, gen_path, fitness=best_fit, extra=meta)
                )

        row: dict[str, Any] = summary.as_dict()
        row["best_genome"] = None if best_genome is None else [float(v) for v in best_genome]
        self._write(row)
        self.history.append(summary)
        self._write_summary()
        return summary

    def _write_summary(self) -> None:
        if self.summary_path is None:
            return
        payload: dict[str, Any] = {
            "generations": [h.as_dict() for h in self.history],
            "best": {
                "generation": self.best_generation,
                "fitness": self.best_fitness,
                "genome": None if self.best_genome is None else [float(v) for v in self.best_genome],
                "maximising": self.maximising,
                "evals_so_far": self.evals_so_far,
                "wall_time": time.perf_counter() - self._t0,
            },
        }
        tmp = self.summary_path.with_suffix(self.summary_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, allow_nan=True))
        tmp.replace(self.summary_path)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(generations, best, mean, global_best)`` arrays for plotting."""
        gens = np.array([h.generation for h in self.history], dtype=int)
        best = np.array([h.best_fitness for h in self.history], dtype=float)
        mean = np.array([h.mean_fitness for h in self.history], dtype=float)
        gbest = np.array([h.global_best_fitness for h in self.history], dtype=float)
        return gens, best, mean, gbest

    def close(self) -> None:
        """Flush and close the record file; write the summary one last time."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
        if self.history:
            self._write_summary()

    def __len__(self) -> int:
        return len(self.history)


def load_generation_history(path: str | Path) -> list[dict[str, Any]]:
    """Read the ``kind="generation"`` rows of a generation log (JSONL or CSV).

    Returns plain dicts (genome lists are decoded from JSON in CSV).  Useful for
    plotting a finished run without re-running it.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"generation log not found: {p}")
    text = p.read_text()
    rows: list[dict[str, Any]] = []
    if p.suffix.lower() == ".csv":
        for row in csv.DictReader(text.splitlines()):
            if row.get("kind") != "generation":
                continue
            out: dict[str, Any] = {}
            for k, v in row.items():
                if v in (None, ""):
                    out[k] = None
                elif k == "best_genome":
                    out[k] = json.loads(v)
                else:
                    try:
                        out[k] = int(v)
                    except ValueError:
                        try:
                            out[k] = float(v)
                        except ValueError:
                            out[k] = v
            rows.append(out)
    else:
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "generation":
                rows.append(row)
    return rows
