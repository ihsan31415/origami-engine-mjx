"""CPU-only tests for the MJX-Warp import guard and backend wiring.

``mujoco-warp`` >= 4.0 is the standalone ``mujoco_warp`` package; 3.x shipped
as ``mujoco.warp`` inside MuJoCo.  These tests fake both modules (and a minimal
``warp`` module for the CUDA probe), so they run without a GPU or Warp install.
"""

import importlib
import sys
import types
from unittest import mock

import numpy as np
import pytest

import origami_engine_mjx.backend as backend_module
from origami_engine_mjx.backends import available_backends, warp_backend

_OPEN_MJCF = """
<mujoco>
  <worldbody>
    <body name="link">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _fake_warp(*, is_cuda):
    """Minimal stand-in for ``warp`` exposing the CUDA probe."""
    device = types.SimpleNamespace(is_cuda=is_cuda)
    return _module("warp", get_device=mock.Mock(return_value=device))


def _fake_mujoco_warp():
    """Stand-in for the ``mujoco-warp`` >= 4.0 API surface."""
    return _module(
        "mujoco_warp",
        put_model=mock.Mock(name="put_model", return_value="warp-model"),
        make_data=mock.Mock(name="make_data", return_value="warp-data"),
        step=mock.Mock(name="step"),
    )


@pytest.fixture
def install_fakes(monkeypatch):
    """Install fake Warp modules and re-import both import guards."""

    def _install(*, is_cuda: bool = True, layout: str = "standalone"):
        # layout: "standalone" -> mujoco_warp, "legacy" -> mujoco.warp, "none".
        if layout == "standalone":
            monkeypatch.setitem(sys.modules, "mujoco_warp", _fake_mujoco_warp())
            monkeypatch.setitem(sys.modules, "mujoco.warp", None)
        elif layout == "legacy":
            monkeypatch.setitem(sys.modules, "mujoco_warp", None)
            monkeypatch.setitem(sys.modules, "mujoco.warp", _fake_warp(is_cuda=is_cuda))
        else:
            monkeypatch.setitem(sys.modules, "mujoco_warp", None)
            monkeypatch.setitem(sys.modules, "mujoco.warp", None)
        monkeypatch.setitem(sys.modules, "warp", _fake_warp(is_cuda=is_cuda))
        wb = importlib.reload(warp_backend)
        be = importlib.reload(backend_module)
        return wb, be

    yield _install
    for name in ("mujoco_warp", "mujoco.warp", "warp"):
        sys.modules.pop(name, None)
    importlib.reload(warp_backend)
    importlib.reload(backend_module)


def test_mujoco_warp_package_is_preferred(install_fakes):
    wb, be = install_fakes(is_cuda=True)
    assert wb._IMPORT_ERROR is None
    assert wb.mjw is sys.modules["mujoco_warp"]
    assert be.MJX_WARP_IMPORT_ERROR is None
    assert be.mjx_warp is sys.modules["mujoco_warp"]
    assert be.require_mjx_warp() is sys.modules["mujoco_warp"]


def test_legacy_mujoco_warp_fallback(install_fakes):
    wb, be = install_fakes(is_cuda=True, layout="legacy")
    assert wb._IMPORT_ERROR is None
    assert wb.mjw is sys.modules["mujoco.warp"]
    assert be.mjx_warp is sys.modules["mujoco.warp"]


def test_missing_warp_records_clear_error(install_fakes):
    wb, be = install_fakes(is_cuda=False, layout="none")
    assert wb.mjw is None
    assert isinstance(wb._IMPORT_ERROR, ImportError)
    assert "mujoco_warp" in str(wb._IMPORT_ERROR)
    assert be.mjx_warp is None
    assert isinstance(be.MJX_WARP_IMPORT_ERROR, ImportError)
    with pytest.raises(RuntimeError, match="requires 'mujoco-warp'"):
        be.require_mjx_warp()
    with pytest.raises(RuntimeError, match="requires 'mujoco-warp'"):
        wb.WarpBackend().build_model(_OPEN_MJCF, {})


def test_available_backends_lists_warp_only_with_cuda(install_fakes):
    install_fakes(is_cuda=True)
    assert "warp" in available_backends()
    install_fakes(is_cuda=False)
    assert "warp" not in available_backends()


def test_warp_backend_reports_device(install_fakes):
    install_fakes(is_cuda=True)
    backend = warp_backend.WarpBackend()
    assert backend.is_available() is True
    assert "cuda=True" in backend.device_info()


def test_build_model_and_make_state_use_mujoco_warp(install_fakes):
    wb, _ = install_fakes(is_cuda=True)
    mjw = wb.mjw
    backend = wb.WarpBackend()

    bundle = backend.build_model(_OPEN_MJCF, {"hinges": []})

    mjw.put_model.assert_called_once_with(bundle["classic_model"])
    assert bundle["warp_model"] == "warp-model"
    assert bundle["info"] == {"hinges": []}
    # 4.0's make_data takes the host model, not the device model.
    backend.make_state(bundle)
    mjw.make_data.assert_called_once_with(bundle["classic_model"])


def test_build_model_falls_back_to_make_model(install_fakes):
    """``mujoco-warp`` 3.x named the device-model constructor ``make_model``."""
    wb, _ = install_fakes(is_cuda=True)
    legacy = _module(
        "mujoco_warp",
        make_model=mock.Mock(name="make_model", return_value="legacy-model"),
        make_data=mock.Mock(name="make_data", return_value="legacy-data"),
        step=mock.Mock(name="step"),
    )
    wb.mjw = legacy
    bundle = wb.WarpBackend().build_model(_OPEN_MJCF, {})
    legacy.make_model.assert_called_once_with(bundle["classic_model"])
    assert bundle["warp_model"] == "legacy-model"


def test_build_model_without_any_constructor_raises(install_fakes):
    wb, _ = install_fakes(is_cuda=True)
    wb.mjw = _module("mujoco_warp")
    with pytest.raises(RuntimeError, match="put_model"):
        wb.WarpBackend().build_model(_OPEN_MJCF, {})


def test_warp_backend_is_unavailable_without_mujoco_warp(install_fakes):
    wb, _ = install_fakes(is_cuda=True, layout="none")
    backend = wb.WarpBackend()
    assert backend.is_available() is False
    assert backend.device_info() == "mujoco-warp not installed"


# ---------------------------------------------------------------------------
# Strict ``wp.array`` stubs
#
# Newer ``warp-lang`` raises RuntimeError("Item indexing is not supported on
# wp.array objects") when a device array is passed to np.asarray() or indexed
# directly, so every read/write must go through ``.numpy()`` / ``.assign()``.
# ---------------------------------------------------------------------------


class StrictWpArray:
    """``wp.array`` stand-in that rejects NumPy/sequence access."""

    _ERROR = "Item indexing is not supported on wp.array objects"

    def __init__(self, data):
        self._data = np.array(data, dtype=float)
        self.numpy_calls = 0

    def numpy(self):
        self.numpy_calls += 1
        return self._data.copy()

    def assign(self, values):
        self._data = np.array(values, dtype=float)

    def __getitem__(self, key):
        raise RuntimeError(self._ERROR)

    def __setitem__(self, key, values):
        raise RuntimeError(self._ERROR)

    def __array__(self, *args, **kwargs):
        raise RuntimeError(self._ERROR)

    def __iter__(self):
        raise RuntimeError(self._ERROR)


class StrictWpData:
    """Minimal ``mujoco_warp`` data stub whose fields are all strict arrays."""

    def __init__(self, mjm):
        self.qpos = StrictWpArray(np.zeros((1, mjm.nq)))
        self.qvel = StrictWpArray(np.zeros((1, mjm.nv)))
        self.ctrl = StrictWpArray(np.zeros((1, mjm.nu)))
        self.geom_xpos = StrictWpArray(np.full((1, mjm.ngeom, 3), 0.25))


class StrictWpModel:
    def __init__(self, mjm):
        self.geom_bodyid = StrictWpArray(mjm.geom_bodyid)


def _strict_step(model, data):
    data.qpos.assign(data.qpos._data + 0.001)


@pytest.fixture
def strict_warp(monkeypatch):
    """Run ``WarpBackend`` against strict ``wp.array`` stubs (no GPU)."""
    fake = _fake_mujoco_warp()
    fake.put_model = mock.Mock(name="put_model", side_effect=StrictWpModel)
    fake.make_data = mock.Mock(name="make_data", side_effect=StrictWpData)
    fake.step = mock.Mock(name="step", side_effect=_strict_step)
    monkeypatch.setitem(sys.modules, "mujoco_warp", fake)
    monkeypatch.setitem(sys.modules, "mujoco.warp", None)
    monkeypatch.setitem(sys.modules, "warp", _fake_warp(is_cuda=True))
    wb = importlib.reload(warp_backend)
    return wb, fake


@pytest.fixture
def tiny_creases():
    from origami_engine_mjx import Crease

    return [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]


def test_to_numpy_uses_warp_numpy():
    arr = StrictWpArray(np.arange(6).reshape(2, 3))
    out = warp_backend._to_numpy(arr)
    assert np.array_equal(out, np.arange(6).reshape(2, 3))
    assert arr.numpy_calls == 1


def test_to_numpy_falls_back_to_asarray():
    assert np.array_equal(warp_backend._to_numpy([1, 2, 3]), np.array([1, 2, 3]))


def test_set_item_assigns_into_warp_array():
    arr = StrictWpArray(np.zeros((1, 4)))
    warp_backend._set_item(arr, (0, 1), 7.0)
    assert arr._data[0, 1] == 7.0
    warp_backend._set_item(arr, (0, slice(0, 2)), np.array([1.0, 2.0]))
    assert np.array_equal(arr._data[0, :2], [1.0, 2.0])


def test_run_accepts_strict_warp_arrays(strict_warp, tiny_creases):
    wb, _ = strict_warp
    result = wb.WarpBackend().run(tiny_creases, duration=0.005, fold_duration=0.002, vibrate=True, record=True)

    assert result.backend == "warp"
    assert np.isfinite(result.final_centroid).all()
    # Table geoms are excluded: the stub places every paper geom at (0.25, 0.25).
    assert np.allclose(result.initial_centroid, [0.25, 0.25])
    traj = result.extra["trajectory"]
    assert traj["qpos"].shape[1] == result.data.qpos._data.shape[1]
    assert traj["centroid"].shape[1] == 2


def test_run_without_vibration_reads_strict_arrays(strict_warp, tiny_creases):
    wb, _ = strict_warp
    result = wb.WarpBackend().run(tiny_creases, duration=0.003, fold_duration=0.002, vibrate=False)
    assert np.isfinite(result.info["hinge_qpos"]).all()


def test_run_batch_with_strict_arrays(strict_warp, tiny_creases):
    wb, _ = strict_warp
    scores = wb.WarpBackend().run_batch(
        [tiny_creases, tiny_creases], duration=0.003, fold_duration=0.002, vibrate=False
    )
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
