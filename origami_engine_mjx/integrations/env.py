from __future__ import annotations

"""Gymnasium-compatible RL environments.

:class:`OrigamiEnv` exposes the fold simulation as a single Gymnasium
environment; :class:`OrigamiVectorEnv` steps many copies in parallel with
``jax.vmap`` on the MJX-JAX backend.  Neither implements a learning algorithm
— they are the environment an RL library trains against.

Requires the ``[rl]`` extra (Gymnasium) and the ``[jax]`` extra (MJX-JAX).
"""

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from ..constants import MAX_FOLD, VIB_AMP_X, VIB_AMP_Z, VIB_FREQ
from .jax_env import _IMPORT_ERROR as _JAX_IMPORT_ERROR
from .jax_env import _centroid, build_jax_model, make_params
from .search_space import SearchSpace

__all__ = ["OrigamiEnv", "OrigamiVectorEnv"]

try:  # pragma: no cover - gymnasium is optional
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # noqa: BLE001
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _GYM_IMPORT_ERROR: BaseException | None = exc
else:
    _GYM_IMPORT_ERROR = None

try:  # pragma: no cover - JAX is optional
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
except Exception:  # noqa: BLE001
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    mjx = None  # type: ignore[assignment]


def _require_gym() -> None:
    if gym is None:
        raise RuntimeError(
            "OrigamiEnv requires 'gymnasium'. Install with: pip install 'origami-engine-mjx[rl]'"
        ) from _GYM_IMPORT_ERROR


def _require_jax() -> None:
    if jax is None or mjx is None:
        raise RuntimeError(
            "OrigamiEnv requires 'mujoco-mjx' and 'jax'. Install with: pip install 'origami-engine-mjx[jax]'"
        ) from _JAX_IMPORT_ERROR


class OrigamiEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Single-environment origami folding task.

    Observation (``Box``, float32, shape ``(dim + 3,)``)
        ``[normalized_target_angles (dim), centroid_x, centroid_y, progress]``
        where ``progress = step_count / max_steps`` in ``[0, 1]``.

    Action (``Box``, float32, shape ``(dim,)``)
        With ``normalized_action=True`` (default) values are in ``[-1, 1]`` and
        map linearly onto the fold-angle bounds.  ``action_mode="absolute"``
        sets the target angles; ``action_mode="delta"`` adds
        ``action * max_delta`` to the current targets.  With
        ``normalized_action=False`` actions are already fold angles in
        ``[lower, upper]``.

    Reward
        By default the change in forward (+x) centroid displacement produced by
        the step, so the episode return equals the total displacement.  Pass
        ``reward_fn(prev_state, action, state, info)`` to shape it.

    Termination
        ``terminated`` is ``True`` only if the simulation becomes non-finite
        (``info["status"] == "unstable"``).  ``truncated`` is ``True`` once
        ``step_count >= max_steps``.

    Seeding
        ``reset(seed=...)`` seeds the per-episode NumPy generator, which is
        only used when ``options={"randomize_start": True}``.  The physics is
        deterministic, so identical actions give identical rollouts.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 1000}

    def __init__(
        self,
        space: SearchSpace | None = None,
        creases=None,
        backend: str = "jax",
        frame_skip: int = 1,
        max_steps: int = 100,
        action_mode: str = "absolute",
        normalized_action: bool = True,
        max_delta: float = MAX_FOLD,
        vibrate: bool = False,
        amp_x: float = VIB_AMP_X,
        amp_z: float = VIB_AMP_Z,
        freq_hz: float = VIB_FREQ,
        reward_fn: Callable[[Any, np.ndarray, Any, dict], float] | None = None,
        render_mode: str | None = None,
        coll_thick: float | None = None,
    ) -> None:
        _require_gym()
        _require_jax()
        if backend != "jax":
            raise ValueError(
                "OrigamiEnv currently supports backend='jax' only; the Warp backend is stateful and not "
                "wrapped here (see docs/RL.md)."
            )
        if action_mode not in {"absolute", "delta"}:
            raise ValueError("action_mode must be 'absolute' or 'delta'")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")

        if space is None:
            if creases is None:
                creases = SearchSpace.from_default_pattern().decode(np.zeros(SearchSpace.from_default_pattern().dim))
            space = SearchSpace.from_creases(creases, active_only=True)
        self.space = space
        self.backend = backend
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.action_mode = action_mode
        self.normalized_action = bool(normalized_action)
        self.max_delta = float(max_delta)
        self.vibrate = bool(vibrate)
        self.amp_x, self.amp_z, self.freq_hz = float(amp_x), float(amp_z), float(freq_hz)
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.coll_thick = coll_thick

        dim = space.dim
        if self.normalized_action:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(
                low=space.lower.astype(np.float32), high=space.upper.astype(np.float32), shape=(dim,), dtype=np.float32
            )
        obs_low = np.array([-1.0] * dim + [-5.0, -5.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0] * dim + [5.0, 5.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(dim + 3,), dtype=np.float32)

        self._model: Any = None
        self._classic: Any = None
        self._params: Any = None
        self._data: Any = None
        self._targets = np.zeros(dim, dtype=float)
        self._initial_centroid = np.zeros(2, dtype=float)
        self._prev_forward = 0.0
        self._step_count = 0
        self._np_random = np.random.default_rng()
        self._renderer: Any = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        creases = self.space.decode(np.zeros(self.space.dim))
        self._model, self._classic, info = build_jax_model(creases, coll_thick=self.coll_thick)
        self._params = make_params(
            self.space,
            self._classic,
            info,
            frame_skip=self.frame_skip,
            max_steps=self.max_steps,
            vibrate=self.vibrate,
            amp_x=self.amp_x,
            amp_z=self.amp_z,
            freq_hz=self.freq_hz,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _angles_from_action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=float).reshape(-1)
        if self.normalized_action:
            a = np.clip(a, -1.0, 1.0)
            if self.action_mode == "delta":
                targets = self._targets + a * self.max_delta
            else:
                targets = self.space.from_normalized(a)
        else:
            if self.action_mode == "delta":
                targets = self._targets + a
            else:
                targets = a
        return np.clip(targets, self.space.lower, self.space.upper)

    def _obs(self) -> np.ndarray:
        span = self.space.upper - self.space.lower
        norm_targets = 2.0 * (self._targets - self.space.lower) / span - 1.0
        centroid = np.asarray(_centroid(self._data, self._params.paper_mask), dtype=float)
        progress = self._step_count / max(self.max_steps, 1)
        return np.concatenate([norm_targets, centroid, [progress]]).astype(np.float32)

    def _centroid_now(self) -> np.ndarray:
        return np.asarray(_centroid(self._data, self._params.paper_mask), dtype=float)

    def _run_physics(self, targets: np.ndarray) -> None:
        angles = jnp.asarray(targets, dtype=jnp.float32)
        data = self._data

        def body(k, d):
            t = jnp.asarray((self._step_count * self.frame_skip + k) * float(self._classic.opt.timestep))
            d = d.replace(ctrl=angles)
            if self.vibrate:
                w = 2.0 * jnp.pi * self.freq_hz
                qpos = d.qpos.at[int(self._params.table_x_qadr)].set(self.amp_x * jnp.sin(w * t))
                qvel = d.qvel.at[int(self._params.table_x_dadr)].set(self.amp_x * w * jnp.cos(w * t))
                qpos = qpos.at[int(self._params.table_z_qadr)].set(self.amp_z * jnp.sin(w * t))
                qvel = qvel.at[int(self._params.table_z_dadr)].set(self.amp_z * w * jnp.cos(w * t))
                d = d.replace(qpos=qpos, qvel=qvel)
            return mjx.step(self._model, d)

        for k in range(self.frame_skip):
            data = body(k, data)
        self._data = data

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset to the unfolded sheet; returns ``(obs, info)``."""
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self._ensure_model()

        options = options or {}
        if options.get("randomize_start"):
            self._targets = self._np_random.uniform(self.space.lower, self.space.upper)
        else:
            self._targets = np.zeros(self.space.dim, dtype=float)

        data = mjx.make_data(self._model)
        data = data.replace(qpos=data.qpos.at[self._params.hinge_adr].set(0.0))
        data = data.replace(ctrl=jnp.asarray(self._targets, dtype=jnp.float32))
        self._data = data
        self._initial_centroid = self._centroid_now()
        self._prev_forward = 0.0
        self._step_count = 0
        info = {"step_count": 0, "status": "ok", "forward_displacement": 0.0, "centroid": self._initial_centroid.copy()}
        return self._obs(), info

    def step(self, action):
        """Advance one step; returns ``(obs, reward, terminated, truncated, info)``."""
        if self._data is None:
            raise RuntimeError("call reset() before step()")
        prev_state = (self._targets.copy(), self._prev_forward)
        self._targets = self._angles_from_action(action)
        self._run_physics(self._targets)
        self._step_count += 1

        centroid = self._centroid_now()
        if not np.isfinite(np.asarray(self._data.qpos)).all() or not np.isfinite(centroid).all():
            info = {
                "step_count": self._step_count,
                "status": "unstable",
                "forward_displacement": float(self._prev_forward),
            }
            return self._obs(), 0.0, True, False, info

        forward = float(centroid[0] - self._initial_centroid[0])
        reward = forward - self._prev_forward
        self._prev_forward = forward
        info = {
            "step_count": self._step_count,
            "status": "ok",
            "forward_displacement": forward,
            "centroid": centroid.copy(),
            "targets": self._targets.copy(),
        }
        if self.reward_fn is not None:
            reward = float(self.reward_fn(prev_state, action, (self._targets.copy(), forward), info))
        truncated = self._step_count >= self.max_steps
        return self._obs(), float(reward), False, bool(truncated), info

    def render(self):
        """Render the current state as an RGB array (``render_mode="rgb_array"``).

        Uses the classic MuJoCo offscreen renderer, which needs a working GL
        context (EGL/OSMesa).  Raises a clear ``RuntimeError`` if rendering is
        unavailable on this machine.
        """
        if self.render_mode != "rgb_array":
            return None
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._classic, height=480, width=640)
        data = mujoco.MjData(self._classic)
        data.qpos[:] = np.asarray(self._data.qpos, dtype=float)
        data.qvel[:] = np.asarray(self._data.qvel, dtype=float)
        mujoco.mj_forward(self._classic, data)
        self._renderer.update_scene(data)
        return self._renderer.render()

    def close(self) -> None:
        """Release the renderer if one was created."""
        self._renderer = None


class OrigamiVectorEnv:
    """Vectorized origami environment (many envs stepped in parallel on MJX-JAX).

    Uses ``jax.vmap`` over :func:`origami_engine_mjx.integrations.jax_env.step`
    so ``N`` environments share one compiled model and step together.  The API
    follows the common vector-env convention used by
    ``gymnasium.vector``: ``reset`` returns ``(obs, info)`` and ``step``
    returns ``(obs, rewards, terminated, truncated, info)``, with a leading
    ``N`` axis on every array.

    All environments share the same fixed crease topology (and therefore the
    same model); only their states differ.  Requires the JAX backend.
    """

    def __init__(
        self,
        space: SearchSpace,
        n_envs: int = 8,
        frame_skip: int = 1,
        max_steps: int = 100,
        vibrate: bool = False,
        amp_x: float = VIB_AMP_X,
        amp_z: float = VIB_AMP_Z,
        freq_hz: float = VIB_FREQ,
        coll_thick: float | None = None,
    ) -> None:
        _require_jax()
        from .jax_env import reset as jax_reset
        from .jax_env import step as jax_step

        self.space = space
        self.n_envs = int(n_envs)
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self._jax_reset = jax_reset
        self._jax_step = jax_step

        creases = space.decode(np.zeros(space.dim))
        self._model, self._classic, info = build_jax_model(creases, coll_thick=coll_thick)
        self._params = make_params(
            space,
            self._classic,
            info,
            frame_skip=frame_skip,
            max_steps=max_steps,
            vibrate=vibrate,
            amp_x=amp_x,
            amp_z=amp_z,
            freq_hz=freq_hz,
        )
        self._states: Any = None

    @property
    def obs_dim(self) -> int:
        """Observation dimensionality (``space.dim + 3``)."""
        return self.space.dim + 3

    def reset(self, seed: int | None = None):
        """Reset all environments; returns ``(obs (N, obs_dim), info)``."""
        del seed  # deterministic reset
        keys = jax.random.split(jax.random.key(0), self.n_envs)
        reset_fn = jax.vmap(lambda k: self._jax_reset(k, self._model, self._params))
        self._states = reset_fn(keys)
        obs = jax.vmap(lambda s: self._obs_of(s))(self._states)
        return np.asarray(obs), {"step_count": np.zeros(self.n_envs, dtype=int)}

    def _obs_of(self, state):
        from .jax_env import observe

        return observe(state, self._params)

    def step(self, actions):
        """Step all environments; returns ``(obs, reward, terminated, truncated, info)``."""
        if self._states is None:
            raise RuntimeError("call reset() before step()")
        actions = jnp.asarray(np.asarray(actions, dtype=np.float32))
        step_fn = jax.vmap(lambda s, a: self._jax_step(self._model, s, self._params, a))
        states, obs, reward, done = step_fn(self._states, actions)
        self._states = states
        zeros = np.zeros(self.n_envs, dtype=bool)
        info = {
            "step_count": np.asarray(states.step_count),
            "forward_displacement": np.asarray(states.prev_forward),
        }
        return (
            np.asarray(obs),
            np.asarray(reward),
            zeros,
            np.asarray(done),
            info,
        )
