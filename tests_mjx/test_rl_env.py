"""Compliance tests for the Gymnasium / JAX-functional RL environments.

Requires the ``[rl]`` (gymnasium) and ``[jax]`` extras.  Simulation-backed tests
are marked ``slow`` (MJX-JAX on CPU is seconds per tiny step).
"""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from origami_engine_mjx import Crease
from origami_engine_mjx.integrations import SearchSpace
from origami_engine_mjx.integrations.env import OrigamiEnv, OrigamiVectorEnv


@pytest.fixture(scope="module")
def tiny_space():
    creases = [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]
    return SearchSpace.from_creases(creases)


def test_env_spaces(tiny_space):
    env = OrigamiEnv(space=tiny_space)
    assert env.observation_space.shape == (tiny_space.dim + 3,)
    assert env.action_space.shape == (tiny_space.dim,)
    assert env.action_space.dtype == np.float32


def test_env_rejects_unknown_render_mode(tiny_space):
    with pytest.raises(ValueError):
        OrigamiEnv(space=tiny_space, render_mode="human")


def test_env_rejects_bad_action_mode(tiny_space):
    with pytest.raises(ValueError):
        OrigamiEnv(space=tiny_space, action_mode="relative")


def test_env_rejects_non_jax_backend(tiny_space):
    with pytest.raises(ValueError):
        OrigamiEnv(space=tiny_space, backend="warp")


def test_env_step_before_reset_raises(tiny_space):
    env = OrigamiEnv(space=tiny_space)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(tiny_space.dim, dtype=np.float32))


@pytest.mark.slow
def test_env_reset_and_step(tiny_space):
    env = OrigamiEnv(space=tiny_space, max_steps=2, frame_skip=1)
    obs, info = env.reset(seed=0)
    assert obs.shape == (tiny_space.dim + 3,)
    assert info["status"] == "ok"
    obs, reward, terminated, truncated, info = env.step(np.zeros(tiny_space.dim, dtype=np.float32))
    assert obs.shape == (tiny_space.dim + 3,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


@pytest.mark.slow
def test_env_passes_gymnasium_check_env(tiny_space):
    from gymnasium.utils.env_checker import check_env

    env = OrigamiEnv(space=tiny_space, max_steps=2, frame_skip=1)
    check_env(env, skip_render_check=True)


@pytest.mark.slow
def test_vector_env_reset_and_step(tiny_space):
    venv = OrigamiVectorEnv(tiny_space, n_envs=2, max_steps=2, frame_skip=1)
    obs, info = venv.reset(seed=0)
    assert obs.shape == (2, tiny_space.dim + 3)
    obs, reward, terminated, truncated, info = venv.step(np.zeros((2, tiny_space.dim), dtype=np.float32))
    assert obs.shape == (2, tiny_space.dim + 3)
    assert reward.shape == (2,)
    assert terminated.shape == (2,)
    assert truncated.shape == (2,)
    assert "step_count" in info
