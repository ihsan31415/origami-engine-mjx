from __future__ import annotations

"""Thin adapters to popular optimizer APIs.

Each adapter only reshapes the :class:`~origami_engine_mjx.integrations.evaluator.OrigamiEvaluator`
interface into what a library expects — it never implements a search.  The
third-party library is imported lazily, so importing this module never pulls in
an optional dependency.

* :func:`to_pymoo_problem` — a ``pymoo`` ``Problem`` (minimizes ``F``).
* :func:`to_nevergrad_function` — a scalar minimizer callable (nevergrad, cma).
* :func:`to_evosax_fitness` — a batched maximizer callable (evosax, JAX).

Install the relevant extra, e.g. ``pip install 'origami-engine-mjx[opt]'``.
"""

from collections.abc import Callable

import numpy as np

from .evaluator import OrigamiEvaluator

__all__ = [
    "space_spec",
    "to_evosax_fitness",
    "to_nevergrad_function",
    "to_pymoo_problem",
]

_OPT_HINT = "Install the optional optimizer extras with: pip install 'origami-engine-mjx[opt]'"


def space_spec(space_or_evaluator) -> dict:
    """Return the optimizer-facing spec of a space or evaluator.

    Exposes the dimensionality, per-variable bounds, parameter names and the
    indices of any presence (crease-count) variables, so an external optimizer
    can build its problem without importing the space classes.  Works for both
    :class:`~origami_engine_mjx.integrations.search_space.SearchSpace` and
    :class:`~origami_engine_mjx.integrations.design_space.DesignSpace` (an
    evaluator is accepted and its ``space`` is used).
    """
    space = getattr(space_or_evaluator, "space", space_or_evaluator)
    kinds = list(getattr(space, "variable_kinds", []))
    presence = [i for i, k in enumerate(kinds) if k == "presence"]
    return {
        "dim": int(space.dim),
        "lower": [float(x) for x in space.lower],
        "upper": [float(x) for x in space.upper],
        "names": list(getattr(space, "parameter_names", [])),
        "groups": list(getattr(space, "groups", [])),
        "presence": presence,
        "n_creases": int(getattr(space, "n_creases", len(getattr(space, "topology", [])))),
    }


def to_nevergrad_function(evaluator: OrigamiEvaluator) -> Callable[[np.ndarray], float]:
    """Return a scalar ``f(x) -> float`` that minimizers (nevergrad, cma) accept.

    The value is ``-fitness`` because these libraries minimize; the evaluator's
    ``maximize`` flag still decides which direction is "better".
    """
    return evaluator.to_minimizer()


def to_evosax_fitness(evaluator: OrigamiEvaluator) -> Callable[[np.ndarray], np.ndarray]:
    """Return a batched ``f(X) -> (N,)`` that evosax-style maximizers accept.

    The value is the (higher-is-better) fitness, so a maximizer such as evosax
    drives it up directly.
    """

    def fitness(x) -> np.ndarray:
        return evaluator.evaluate_batch(np.asarray(x, dtype=float)).fitness

    return fitness


def to_pymoo_problem(evaluator: OrigamiEvaluator, name: str = "origami"):
    """Wrap the evaluator as a ``pymoo.core.problem.Problem`` (single objective).

    ``pymoo`` minimizes ``out["F"]``, so the adapter reports ``-fitness``.

    Raises
    ------
    RuntimeError
        If ``pymoo`` is not installed.
    """
    try:
        from pymoo.core.problem import Problem
    except Exception as exc:
        raise RuntimeError(f"pymoo is required for to_pymoo_problem(). {_OPT_HINT}") from exc

    space = evaluator.space

    class _OrigamiProblem(Problem):
        def __init__(self) -> None:
            super().__init__(
                n_var=space.dim,
                n_obj=1,
                n_constr=0,
                xl=space.lower,
                xu=space.upper,
                vtype=float,
            )
            self.name = name

        def _evaluate(self, X, out, *args, **kwargs) -> None:
            out["F"] = (-evaluator.evaluate_batch(np.asarray(X, dtype=float)).fitness).reshape(-1, 1)

    return _OrigamiProblem()
