from __future__ import annotations

"""Backend import guards.

Keeping optional imports in one place lets geometry and I/O modules be
imported (and unit-tested) without JAX or Warp installed.
"""

import importlib

# --- classic MuJoCo (used for XML compilation + rendering) ---
MUJOCO_IMPORT_ERROR: BaseException | None = None
try:
    import mujoco
except Exception as exc:  # noqa: BLE001
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc


def require_mujoco(message: str = "MuJoCo is required for this operation."):
    """Return the imported ``mujoco`` module or raise a helpful RuntimeError."""
    if mujoco is None:
        raise RuntimeError(message) from MUJOCO_IMPORT_ERROR
    return mujoco


# --- MJX-JAX ---
MJX_JAX_IMPORT_ERROR: BaseException | None = None
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    from mujoco import mjx as mjx_jax
except Exception as exc:  # noqa: BLE001
    mjx_jax = None
    MJX_JAX_IMPORT_ERROR = exc


def require_mjx_jax(message: str = "MJX-JAX backend requires 'mujoco-mjx' and 'jax'."):
    if mjx_jax is None:
        raise RuntimeError(message + "  Install with: pip install 'origami-engine-mjx[jax]'") from MJX_JAX_IMPORT_ERROR
    return mjx_jax


# --- MJX-Warp ---
MJX_WARP_IMPORT_ERROR: BaseException | None = None


def _import_mjx_warp():
    """Return the MuJoCo Warp module, preferring the standalone package.

    ``mujoco-warp`` >= 4.0 installs as ``mujoco_warp``; older releases shipped
    inside MuJoCo as ``mujoco.warp``.
    """
    try:
        import mujoco_warp

        return mujoco_warp
    except ImportError:  # fall back to the legacy location
        pass
    try:
        # importlib because ``import mujoco.warp`` binds the parent ``mujoco``.
        return importlib.import_module("mujoco.warp")
    except ImportError as exc:  # neither location is installed
        raise ImportError("neither 'mujoco_warp' nor 'mujoco.warp' is importable") from exc


try:
    mjx_warp = _import_mjx_warp()
except Exception as exc:  # noqa: BLE001
    mjx_warp = None
    MJX_WARP_IMPORT_ERROR = exc


def require_mjx_warp(message: str = "MJX-Warp backend requires 'mujoco-warp' and an NVIDIA GPU."):
    if mjx_warp is None:
        raise RuntimeError(
            message + "  Install with: pip install 'origami-engine-mjx[warp]'"
        ) from MJX_WARP_IMPORT_ERROR
    return mjx_warp


__all__ = [
    "MJX_JAX_IMPORT_ERROR",
    "MJX_WARP_IMPORT_ERROR",
    "MUJOCO_IMPORT_ERROR",
    "mjx_jax",
    "mjx_warp",
    "mujoco",
    "require_mjx_jax",
    "require_mjx_warp",
    "require_mujoco",
]
