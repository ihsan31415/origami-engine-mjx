from __future__ import annotations

"""Backend factory and registry.

Resolves a backend name (``"jax"`` or ``"warp"``) to a concrete
:class:`~origami_engine_mjx.backends.base.SimulationBackend` instance.
Missing optional dependencies produce clear, actionable errors.
"""

from .base import SimulationBackend

_REGISTRY: dict[str, type[SimulationBackend]] = {}


def register_backend(name: str, cls: type[SimulationBackend]) -> None:
    _REGISTRY[name] = cls


def available_backends() -> list[str]:
    """Return names of backends whose dependencies are installed."""
    result = []
    for name, cls in _REGISTRY.items():
        try:
            if cls().is_available():
                result.append(name)
        except Exception:  # noqa: BLE001, S110 - availability probe must never raise
            pass
    return result


def get_backend(name: str) -> SimulationBackend:
    """Instantiate the named backend.

    Parameters
    ----------
    name : {"jax", "warp"}
        Backend identifier.  ``"jax"`` selects MJX-JAX (CPU/GPU/TPU via JAX).
        ``"warp"`` selects MJX-Warp (NVIDIA CUDA GPU).

    Raises
    ------
    ValueError
        If the name is unknown.
    RuntimeError
        If the backend's optional dependencies are not installed.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; expected one of {sorted(_REGISTRY)}")
    cls = _REGISTRY[name]
    instance = cls()
    if not instance.is_available():
        raise RuntimeError(
            f"backend {name!r} is not available. "
            f"Install its optional dependencies, e.g.  pip install 'origami-engine-mjx[{name}]'"
        )
    return instance


# Register built-in backends.  Each backend module guards its optional
# imports, so a missing JAX/Warp install does not break the registry; the
# availability probe in available_backends()/get_backend() reports it.
def _register_builtin() -> None:
    from .jax_backend import JaxBackend
    from .warp_backend import WarpBackend

    register_backend("jax", JaxBackend)
    register_backend("warp", WarpBackend)


_register_builtin()


__all__ = ["available_backends", "get_backend", "register_backend"]
