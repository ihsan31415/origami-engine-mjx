from __future__ import annotations

"""Optimization / reinforcement-learning integration interfaces.

This subpackage is the **environment and evaluator substrate** for external
search and learning libraries.  It deliberately ships **no** optimization or
learning algorithm: every algorithm lives in a third-party package
(``pymoo``, ``nevergrad``, ``cma``, ``evosax``, Stable-Baselines3, …) that
consumes the small, stable interfaces defined here.

Modules
-------
* :mod:`~origami_engine_mjx.integrations.search_space` — :class:`SearchSpace`,
  the fixed-topology fold-angle genome spec (encode/decode, bounds, sampling).
* :mod:`~origami_engine_mjx.integrations.evaluator` — :class:`OrigamiEvaluator`,
  a stateless, picklable objective plus behavior descriptors for QD.
* :mod:`~origami_engine_mjx.integrations.adapters` — thin, lazily-imported
  adapters to ``pymoo`` / ``nevergrad`` / ``cma`` / ``evosax`` APIs.
* :mod:`~origami_engine_mjx.integrations.env` — Gymnasium ``OrigamiEnv`` and a
  vectorized ``OrigamiVectorEnv``.
* :mod:`~origami_engine_mjx.integrations.jax_env` — pure JAX ``reset`` / ``step``
  for ``jit`` / ``vmap`` / ``scan``.
* :mod:`~origami_engine_mjx.integrations.plumbing` — seeding, genome/checkpoint
  I/O, run logging and a throughput benchmark.

Everything here is additive: it imports the existing public API and never
changes it.  See ``docs/INTEGRATIONS.md`` for the full guide.

Examples
--------
>>> from origami_engine_mjx.integrations import SearchSpace, OrigamiEvaluator
>>> space = SearchSpace.from_default_pattern()          # 15-d genome
>>> evaluator = OrigamiEvaluator(space, backend="jax")  # doctest: +SKIP
>>> genome = space.sample(0)                            # doctest: +SKIP
>>> score = evaluator(genome)                           # doctest: +SKIP
"""

from .adapters import to_evosax_fitness, to_nevergrad_function, to_pymoo_problem
from .evaluator import DESCRIPTOR_NAMES, BatchResult, EvalRecord, OrigamiEvaluator
from .plumbing import (
    RunLogger,
    benchmark,
    load_best,
    load_genome,
    make_rng,
    save_best,
    save_genome,
    set_global_seed,
)
from .search_space import SearchSpace

# ``env``/``jax_env`` pull in optional dependencies (gymnasium, JAX).  They are
# safe to import without them (the imports are guarded), but we re-export the
# symbols lazily so that a NumPy-only user never pays for them.
__all__ = [
    "DESCRIPTOR_NAMES",
    "BatchResult",
    "EvalRecord",
    "OrigamiEvaluator",
    "RunLogger",
    "SearchSpace",
    "benchmark",
    "load_best",
    "load_genome",
    "make_rng",
    "save_best",
    "save_genome",
    "set_global_seed",
    "to_evosax_fitness",
    "to_nevergrad_function",
    "to_pymoo_problem",
]


def __getattr__(name: str):
    """Lazily expose the optional JAX/Gymnasium environment symbols."""
    if name in {"OrigamiEnv", "OrigamiVectorEnv"}:
        from . import env as _env

        return getattr(_env, name)
    if name in {"EnvParams", "EnvState", "build_jax_model", "make_params", "observe"}:
        from . import jax_env as _jax_env

        return getattr(_jax_env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
