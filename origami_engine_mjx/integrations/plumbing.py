from __future__ import annotations

"""Pipeline plumbing helpers (no optimization algorithms).

Small, dependency-light utilities that every optimization/RL pipeline needs but
that are not interesting on their own: reproducible seeding, genome
save/load in the existing JSON crease format, a best-so-far checkpoint, an
append-only run logger (JSONL/CSV) and a throughput benchmark.
"""

import csv
import json
import random
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..serialization import load_creases_from_json, save_creases_to_json
from .evaluator import BatchResult, EvalRecord, OrigamiEvaluator
from .search_space import SearchSpace

__all__ = [
    "RunLogger",
    "benchmark",
    "load_best",
    "load_genome",
    "make_rng",
    "save_best",
    "save_genome",
    "set_global_seed",
]


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Return a seeded :class:`numpy.random.Generator` (``None`` → entropy)."""
    return np.random.default_rng(seed)


def set_global_seed(seed: int) -> None:
    """Seed the global Python ``random`` and NumPy RNGs.

    JAX has no global RNG, so JAX code must take an explicit
    ``jax.random.key(seed)``; this function does not (and cannot) seed it.

    The physics itself is deterministic, so this only matters for pipelines
    that sample genomes or reset environments.  Returns nothing.
    """
    random.seed(seed)
    np.random.seed(seed)
    # JAX has no global RNG: JAX code takes an explicit ``jax.random.key(seed)``.


# ----------------------------------------------------------------------
# Genome / checkpoint I/O
# ----------------------------------------------------------------------


def save_genome(
    genome,
    space: SearchSpace,
    path,
    angle_unit: str = "degree",
) -> Path:
    """Save one genome as a crease-pattern JSON file.

    The genome is decoded with :meth:`SearchSpace.decode`, so the file is a
    standard crease pattern readable by
    :func:`origami_engine_mjx.serialization.load_creases_from_json`.
    """
    return save_creases_to_json(space.decode(genome), path, angle_unit=angle_unit)


def load_genome(path, space: SearchSpace) -> np.ndarray:
    """Load a crease-pattern JSON file back into a genome vector.

    Raises
    ------
    ValueError
        If the file's crease topology does not match ``space``.
    """
    creases = load_creases_from_json(path)
    return space.encode(creases)


def save_best(
    genome,
    space: SearchSpace,
    path,
    fitness: float | None = None,
    extra: Mapping[str, Any] | None = None,
    angle_unit: str = "degree",
) -> Path:
    """Save a best-so-far checkpoint: creases plus optional metadata.

    Writes the standard crease JSON with two extra top-level keys (``fitness``
    and ``meta``) that the crease reader ignores, so the file stays loadable by
    :func:`~origami_engine_mjx.serialization.load_creases_from_json`.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(save_creases_to_json(space.decode(genome), out, angle_unit=angle_unit).read_text())
    payload["fitness"] = None if fitness is None else float(fitness)
    payload["meta"] = dict(extra or {})
    out.write_text(json.dumps(payload, indent=2))
    return out


def load_best(path, space: SearchSpace) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a checkpoint written by :func:`save_best`.

    Returns ``(genome, metadata)`` where metadata has ``fitness`` and ``meta``.
    """
    payload = json.loads(Path(path).read_text())
    creases = load_creases_from_json(path)
    genome = space.encode(creases)
    meta = {"fitness": payload.get("fitness"), "meta": payload.get("meta", {})}
    return genome, meta


# ----------------------------------------------------------------------
# Run logging
# ----------------------------------------------------------------------


class RunLogger:
    """Append evaluation records to a JSONL or CSV file.

    Parameters
    ----------
    path : str | Path
        Destination file.  Its suffix selects the format unless ``format`` is
        given explicitly.
    format : {"jsonl", "csv"} | None
        ``None`` infers from the suffix (default JSONL for other suffixes).
    descriptor_names : sequence[str] | None
        Descriptors to include as columns/keys.  Defaults to the keys present
        in the first record.

    Examples
    --------
    >>> with RunLogger("run.jsonl") as log:      # doctest: +SKIP
    ...     log.log_batch(evaluator.evaluate_batch(genomes), genomes)
    """

    def __init__(self, path, format: str | None = None, descriptor_names: Iterable[str] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if format is None:
            format = "csv" if self.path.suffix.lower() == ".csv" else "jsonl"
        if format not in {"jsonl", "csv"}:
            raise ValueError("format must be 'jsonl' or 'csv'")
        self.format = format
        self._descriptor_names = list(descriptor_names) if descriptor_names is not None else None
        self._fh: Any = None
        self._writer: Any = None

    def __enter__(self) -> RunLogger:  # noqa: PYI034 - py3.10 has no typing.Self
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._fh is not None:
            return
        self._fh = self.path.open("w", newline="")
        if self.format == "csv":
            fieldnames = ["genome", "fitness", "raw", "status", "time"]
            if self._descriptor_names:
                fieldnames += list(self._descriptor_names)
            self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames, extrasaction="ignore")
            self._writer.writeheader()

    def log(self, record: EvalRecord, genome=None) -> None:
        """Append a single :class:`~origami_engine_mjx.integrations.evaluator.EvalRecord`."""
        # Infer the descriptor columns from the first record *before* opening the
        # file, so the CSV header includes them (JSONL does not need a header).
        if self._descriptor_names is None:
            self._descriptor_names = [k for k in record.descriptors]
        self._ensure_open()
        row = {
            "genome": None if genome is None else [float(x) for x in np.asarray(genome, dtype=float).reshape(-1)],
            "fitness": float(record.fitness),
            "raw": float(record.raw),
            "status": record.status,
            "time": float(record.time),
        }
        for k, v in record.descriptors.items():
            row[k] = float(v)
        if self.format == "jsonl":
            self._fh.write(json.dumps(row) + "\n")
        else:
            row["genome"] = "" if row["genome"] is None else json.dumps(row["genome"])
            self._writer.writerow(row)
        self._fh.flush()

    def log_batch(self, result: BatchResult, genomes=None) -> None:
        """Append a whole :class:`~origami_engine_mjx.integrations.evaluator.BatchResult`."""
        genomes_arr = None if genomes is None else np.asarray(genomes, dtype=float)
        for i, rec in enumerate(result.as_records()):
            genome = None if genomes_arr is None else genomes_arr[i]
            self.log(rec, genome)

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None


# ----------------------------------------------------------------------
# Throughput benchmark
# ----------------------------------------------------------------------


def benchmark(
    evaluator: OrigamiEvaluator,
    n: int = 32,
    batch_size: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure evaluator throughput on ``n`` random genomes.

    Returns a dict with ``n``, ``wall_time`` (seconds), ``evals_per_sec`` and
    ``sec_per_eval``.  The first genome is discarded from timing as a warm-up
    (JIT/compile costs are not representative of steady state).

    Note: this measures *evaluations per second* for the configured
    ``duration``; it does not measure raw physics steps per second.
    """
    space = evaluator.space
    rng = make_rng(seed)
    genomes = space.sample(rng, size=int(n))
    if batch_size is None:
        batch_size = evaluator.batch_size or n
    batch_size = max(1, int(batch_size))

    # Warm-up (first call may trigger JAX compilation).
    evaluator.evaluate(genomes[0])

    t0 = time.perf_counter()
    done = 0
    for start in range(0, n, batch_size):
        chunk = genomes[start : start + batch_size]
        evaluator.evaluate_batch(chunk, return_details=False)
        done += len(chunk)
    wall = time.perf_counter() - t0
    return {
        "n": int(n),
        "wall_time": float(wall),
        "evals_per_sec": float(done / wall) if wall > 0 else float("inf"),
        "sec_per_eval": float(wall / done) if done else float("inf"),
        "batch_size": batch_size,
    }
