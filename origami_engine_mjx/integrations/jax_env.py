from __future__ import annotations

"""JAX-functional environment API (pure ``reset`` / ``step``).

These functions expose the fold simulation as pure JAX functions so JAX-native
pipelines can use ``jax.jit``, ``jax.vmap`` and ``jax.lax.scan`` over many
environments.  They are backend-specific to **MJX-JAX** (the Warp backend is
stateful and not exposed here).

What is and is not jittable
---------------------------
* ``reset`` and ``step`` are pure functions of their arguments; ``model`` and
  ``params`` are pytrees and the state is a :class:`EnvState` pytree, so all of
  ``jit`` / ``vmap`` / ``scan`` work.
* The *physics* inside ``step`` (``mjx.step``) is **not differentiable** in any
  useful sense for this model: mesh–mesh contacts and the discrete face graph
  are non-smooth, so gradients through ``step`` are unreliable.  Use the
  evaluator's black-box scores for gradient-free optimizers instead.
* JIT-compiling ``mjx.step`` for this contact model is expensive (the first
  trace can take tens of seconds on CPU; see ``docs/SIMULATION.md``).  Wrap
  ``step`` in ``jax.jit`` once, outside your training loop, and reuse it.

Requires the ``[jax]`` extra.
"""

from typing import Any, NamedTuple

import numpy as np

from ..constants import VIB_AMP_X, VIB_AMP_Z, VIB_FREQ
from ..geometry import build_pattern
from ..model import generate_xml
from ..serialization import normalize_creases
from ..types import Crease
from .search_space import SearchSpace

#: Nonzero probe angle used to recover the per-hinge sign from ``generate_xml``
#: (``target = sign * angle``; the sign is a pure function of the geometry).
_SIGN_PROBE = 0.1

__all__ = [
    "EnvParams",
    "EnvState",
    "build_jax_model",
    "make_params",
    "observe",
    "reset",
    "step",
]

try:  # pragma: no cover - exercised only when JAX is present
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
except Exception as exc:  # noqa: BLE001
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    mjx = None  # type: ignore[assignment]
    _IMPORT_ERROR: BaseException | None = exc
else:
    _IMPORT_ERROR = None


def _require_jax() -> None:
    if jax is None or mjx is None:
        raise RuntimeError(
            "The JAX-functional environment API requires 'mujoco-mjx' and 'jax'. "
            "Install with: pip install 'origami-engine-mjx[jax]'"
        ) from _IMPORT_ERROR


class EnvState(NamedTuple):
    """Pure environment state (a JAX pytree)."""

    data: Any
    """Backend-native MJX ``Data`` pytree."""

    targets: Any
    """Current target fold angles ``(dim,)`` in radians."""

    step_count: Any
    """Integer step counter (scalar)."""

    prev_forward: Any
    """Forward (+x) displacement recorded at the end of the previous step."""

    initial_centroid: Any
    """Centroid ``(2,)`` of the unfolded paper, used as the displacement origin."""


class EnvParams(NamedTuple):
    """Static, vmap-friendly environment parameters (a JAX pytree).

    Build one with :func:`make_params`.  All fields are JAX arrays so the whole
    object is a pytree and can be passed through ``jit``/``vmap``.

    ``hinge_adr`` is ordered to match the search space's genome (one entry per
    ``space.active_indices``); ``ctrl_perm`` reindexes a genome-ordered angle
    vector into the model's actuator order (the model emits hinges in a
    depth-first traversal, which is generally *not* sorted crease order), and
    ``ctrl_sign`` carries the per-hinge sign that :func:`build_jax_model`
    recovers, so a crease angle maps to the same joint target as
    :class:`~origami_engine_mjx.engine.OrigamiEngine`.
    """

    hinge_adr: Any
    ctrl_perm: Any
    ctrl_sign: Any
    table_x_qadr: Any
    table_x_dadr: Any
    table_z_qadr: Any
    table_z_dadr: Any
    paper_mask: Any
    lower: Any
    upper: Any
    initial_targets: Any
    frame_skip: Any
    dt: Any
    max_steps: Any
    vibrate: Any
    amp_x: Any
    amp_z: Any
    freq_hz: Any


def build_jax_model(creases, coll_thick: float | None = None):
    """Compile a crease pattern into an MJX model plus lookup metadata.

    Returns
    -------
    tuple
        ``(mjx_model, classic_model, info)``.  ``classic_model`` is the classic
        :class:`mujoco.MjModel` used for name → address lookups.

    The returned ``info`` also carries ``hinge_creases`` (the crease index each
    model hinge drives) and ``hinge_signs`` (the per-hinge sign that
    :func:`origami_engine_mjx.model.generate_xml` folds into the joint target,
    ``target = sign * angle``).  The environments use both to map a crease-angle
    action onto the model's actuators.
    """
    _require_jax()
    from ..backends.jax_backend import JaxBackend

    normalized = normalize_creases(creases)
    pat = build_pattern(normalized)
    if not pat.valid:
        raise ValueError(f"invalid crease pattern: {pat.reason}")
    xml, info = generate_xml(pat, normalized, coll_thick=coll_thick)
    # Record which crease each actuated hinge corresponds to, in the model's
    # hinge (depth-first) order.  ``make_params`` uses this to align the search
    # space's genome order with the model's actuator order.
    info["hinge_creases"] = [int(pat.edges[h["edge"]]["crease"]) for h in info["hinges"]]
    # Recover the per-hinge sign with a nonzero probe: ``target = sign * angle``
    # and the sign is purely geometric, so ``sign(target)`` is the sign for any
    # positive probe.  The model geometry/axes do not depend on the angles, so
    # this second pass is only used for the sign metadata.
    probe = [Crease(c.p1, c.p2, _SIGN_PROBE) for c in normalized]
    _, probe_info = generate_xml(pat, probe, coll_thick=coll_thick)
    info["hinge_signs"] = [int(np.sign(t)) for t in probe_info["targets"]]
    bundle = JaxBackend().build_model(xml, info)
    return bundle["mjx_model"], bundle["classic_model"], info


def make_params(
    space: SearchSpace,
    classic_model,
    info: dict[str, Any],
    frame_skip: int = 1,
    max_steps: int = 100,
    vibrate: bool = False,
    amp_x: float = VIB_AMP_X,
    amp_z: float = VIB_AMP_Z,
    freq_hz: float = VIB_FREQ,
) -> EnvParams:
    """Assemble :class:`EnvParams` from a search space and a compiled model.

    ``space`` must describe the *same* topology used to build ``classic_model``;
    ``frame_skip`` physics steps are run per :func:`step` call.

    ``info`` must be the dict returned by :func:`build_jax_model` (it carries the
    hinge→crease map and per-hinge sign).  The genome order (sorted crease
    indices) is aligned with the model's actuator order here, so action entry
    ``j`` drives the hinge of crease ``space.active_indices[j]`` — and with the
    same sign that :class:`~origami_engine_mjx.engine.OrigamiEngine` uses.

    Raises
    ------
    ValueError
        If the model's actuated hinges do not match the space's active creases
        (e.g. the wrong pattern, or a crease split across several tree hinges).
    """
    _require_jax()
    hinge_adr = np.asarray([classic_model.joint(h["name"]).qposadr[0] for h in info["hinges"]], dtype=np.int32)
    if hinge_adr.size != space.dim:
        raise ValueError(
            f"search space has {space.dim} variables but the model has {hinge_adr.size} actuated hinges; "
            "build the space with active_only=True over the same pattern"
        )
    # Align the genome order (sorted crease indices) with the model's actuator
    # order.  The model emits hinges in a depth-first traversal, so hinge h does
    # not generally drive crease ``space.active_indices[h]``.  We require the
    # hinge→crease map rather than assuming identity, since assuming identity
    # silently sends each action to the wrong hinge.
    hinge_creases = info.get("hinge_creases")
    if hinge_creases is None:
        raise ValueError("info is missing 'hinge_creases'; pass the dict returned by build_jax_model()")
    crease_to_j = {int(c): j for j, c in enumerate(space.active_indices)}
    if {int(c) for c in hinge_creases} != set(crease_to_j):
        raise ValueError(
            "model hinges do not match the search space's active creases; "
            "build the space with active_only=True over the same pattern"
        )
    # ctrl_perm[h] = genome index whose angle drives model hinge h.
    ctrl_perm = np.asarray([crease_to_j[int(c)] for c in hinge_creases], dtype=np.int32)
    # Per-hinge sign (model order): ``generate_xml`` drives each joint with
    # ``sign * angle``, so applying the raw crease angle without it folds the
    # hinge the wrong way relative to OrigamiEngine.
    hinge_signs = info.get("hinge_signs")
    if hinge_signs is None:
        raise ValueError("info is missing 'hinge_signs'; pass the dict returned by build_jax_model()")
    ctrl_sign = np.asarray(hinge_signs, dtype=np.float32)
    if ctrl_sign.shape != hinge_adr.shape:
        raise ValueError("model hinge signs do not match the number of actuated hinges")
    # Reorder hinge addresses to genome order: hinge_adr[j] is the qpos of
    # the hinge driven by genome entry j.
    space_to_model_h = np.empty(hinge_adr.size, dtype=np.int32)
    space_to_model_h[ctrl_perm] = np.arange(hinge_adr.size, dtype=np.int32)
    hinge_adr = hinge_adr[space_to_model_h]
    table_id = int(classic_model.body("table").id)
    geom_bodyid = np.asarray(classic_model.geom_bodyid)
    paper_mask = geom_bodyid != table_id
    dt = float(classic_model.opt.timestep)
    return EnvParams(
        hinge_adr=jnp.asarray(hinge_adr),
        ctrl_perm=jnp.asarray(ctrl_perm),
        ctrl_sign=jnp.asarray(ctrl_sign),
        table_x_qadr=jnp.asarray(int(classic_model.joint("table_x").qposadr[0]), dtype=jnp.int32),
        table_x_dadr=jnp.asarray(int(classic_model.joint("table_x").dofadr[0]), dtype=jnp.int32),
        table_z_qadr=jnp.asarray(int(classic_model.joint("table_z").qposadr[0]), dtype=jnp.int32),
        table_z_dadr=jnp.asarray(int(classic_model.joint("table_z").dofadr[0]), dtype=jnp.int32),
        paper_mask=jnp.asarray(paper_mask),
        lower=jnp.asarray(space.lower, dtype=jnp.float32),
        upper=jnp.asarray(space.upper, dtype=jnp.float32),
        initial_targets=jnp.zeros((space.dim,), dtype=jnp.float32),
        frame_skip=jnp.asarray(int(frame_skip), dtype=jnp.int32),
        dt=jnp.asarray(dt, dtype=jnp.float32),
        max_steps=jnp.asarray(int(max_steps), dtype=jnp.int32),
        vibrate=jnp.asarray(bool(vibrate)),
        amp_x=jnp.asarray(float(amp_x), dtype=jnp.float32),
        amp_z=jnp.asarray(float(amp_z), dtype=jnp.float32),
        freq_hz=jnp.asarray(float(freq_hz), dtype=jnp.float32),
    )


def _centroid(data, paper_mask):
    return jnp.mean(data.geom_xpos[paper_mask, :2], axis=0)


def _action_to_angles(action, params):
    """Map a normalized ``[-1, 1]`` action to angle-space targets."""
    z = jnp.clip(action, -1.0, 1.0)
    return params.lower + (z + 1.0) * 0.5 * (params.upper - params.lower)


def _apply_vibration(data, params, t):
    w = 2.0 * jnp.pi * params.freq_hz
    qpos = data.qpos.at[params.table_x_qadr].set(params.amp_x * jnp.sin(w * t))
    qvel = data.qvel.at[params.table_x_dadr].set(params.amp_x * w * jnp.cos(w * t))
    qpos = qpos.at[params.table_z_qadr].set(params.amp_z * jnp.sin(w * t))
    qvel = qvel.at[params.table_z_dadr].set(params.amp_z * w * jnp.cos(w * t))
    return data.replace(qpos=qpos, qvel=qvel)


def reset(key, model, params: EnvParams) -> EnvState:
    """Reset the environment to the unfolded sheet.

    ``key`` is accepted for API compatibility; reset is deterministic.
    """
    _require_jax()
    del key  # deterministic reset
    data = mjx.make_data(model)
    data = data.replace(qpos=data.qpos.at[params.hinge_adr].set(0.0))
    # Same genome -> model actuator mapping as ``step`` (a no-op while the
    # initial targets are zero, but keeps an overridden value consistent).
    data = data.replace(ctrl=params.ctrl_sign * params.initial_targets[params.ctrl_perm])
    initial_centroid = _centroid(data, params.paper_mask)
    return EnvState(
        data=data,
        targets=params.initial_targets,
        step_count=jnp.asarray(0, dtype=jnp.int32),
        prev_forward=jnp.asarray(0.0, dtype=jnp.float32),
        initial_centroid=initial_centroid,
    )


def observe(state: EnvState, params: EnvParams):
    """Observation = normalized targets ‖ centroid xy ‖ step fraction."""
    span = params.upper - params.lower
    norm_targets = 2.0 * (state.targets - params.lower) / span - 1.0
    centroid = _centroid(state.data, params.paper_mask)
    progress = state.step_count / jnp.maximum(params.max_steps, 1)
    return jnp.concatenate([norm_targets, centroid, progress[None]])


def step(model, state: EnvState, params: EnvParams, action):
    """Advance one environment step.

    Returns ``(state, obs, reward, done)`` where ``reward`` is the change in
    forward (+x) displacement produced by this step and ``done`` is ``True``
    once ``max_steps`` is reached.
    """
    _require_jax()
    angles = _action_to_angles(action, params)
    ctrl = params.ctrl_sign * angles[params.ctrl_perm]  # genome order -> model actuator order + sign

    def body(k, data):
        t = (state.step_count * params.frame_skip + k).astype(jnp.float32) * params.dt
        data = data.replace(ctrl=ctrl)
        data = jax.lax.cond(params.vibrate, lambda d: _apply_vibration(d, params, t), lambda d: d, data)
        return mjx.step(model, data)

    data = jax.lax.fori_loop(0, params.frame_skip, body, state.data)
    centroid = _centroid(data, params.paper_mask)
    forward = centroid[0] - state.initial_centroid[0]
    reward = forward - state.prev_forward

    step_count = state.step_count + 1
    new_state = EnvState(
        data=data,
        targets=angles,
        step_count=step_count,
        prev_forward=forward,
        initial_centroid=state.initial_centroid,
    )
    obs = observe(new_state, params)
    done = step_count >= params.max_steps
    return new_state, obs, reward, done
