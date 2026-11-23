"""Reinforcement-learning environments for origami folding.

Two ways to drive the simulation from RL code:

1. :class:`OrigamiEnv` — a standard Gymnasium environment (``reset``/``step``,
   ``Box`` spaces).  Pass ``check_env`` and hand it to any Gymnasium-compatible
   trainer (Stable-Baselines3, CleanRL, ...).
2. :func:`origami_engine_mjx.integrations.jax_env.reset` / ``step`` — pure JAX
   functions you can wrap in ``jax.jit`` / ``jax.vmap`` / ``jax.lax.scan``.

Neither implements a learning algorithm; this library is the environment.

Run:
    python examples/05_rl_env.py

The tiny 2-face pattern is used for CPU demo speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from origami_engine_mjx import Crease
from origami_engine_mjx.integrations import SearchSpace


def main():
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        print("gymnasium is not installed. Install with: pip install 'origami-engine-mjx[rl]'")
        return

    from origami_engine_mjx.integrations.env import OrigamiEnv

    creases = [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]
    space = SearchSpace.from_creases(creases)
    env = OrigamiEnv(space=space, max_steps=3, frame_skip=1, vibrate=False)
    print(f"observation space: {env.observation_space}")
    print(f"action space:      {env.action_space}")

    obs, info = env.reset(seed=0)
    print(f"reset -> obs {obs.shape}, info status={info['status']}")

    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(env.max_steps):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        print(f"  step {info['step_count']}: reward={reward:+.4g}  forward={info['forward_displacement']:+.4g} m")
        if terminated or truncated:
            break
    print(f"episode return (total forward displacement): {total:+.4g} m")

    # The pure-JAX API (jittable / vmappable) is available too:
    from origami_engine_mjx.integrations import jax_env

    model, classic, hin = jax_env.build_jax_model(space.decode(np.zeros(space.dim)))
    params = jax_env.make_params(space, classic, hin, frame_skip=1, max_steps=3)
    state = jax_env.reset(None, model, params)
    state, obs, reward, done = jax_env.step(model, state, params, np.zeros(space.dim, dtype=np.float32))
    print(f"jax_env.step -> obs {np.asarray(obs).shape}, reward={float(reward):+.4g}, done={bool(done)}")


if __name__ == "__main__":
    main()
