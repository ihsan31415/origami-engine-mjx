"""MJX-Warp simulation backend.

Uses ``mujoco_warp`` (MuJoCo Warp, NVIDIA CUDA) to step the same MJCF model on
GPU.  This backend **requires an NVIDIA CUDA GPU** and the ``mujoco-warp`` +
``warp-lang`` packages (``pip install 'origami-engine-mjx[warp]'``).

Two execution paths are offered:

* :meth:`WarpBackend.run` -- one pattern, returned as a full
  :class:`~origami_engine_mjx.types.SimulationResult`.
* :meth:`WarpBackend.run_batch` -- many patterns in one Warp model, one world
  per pattern, stepped inside a single CUDA graph.  This is the fast path for
  population-based optimisers: cost per step is nearly flat in the number of
  worlds, so a whole generation is scored in about the time of one step loop.
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Any

import numpy as np

from ..constants import (
    VIB_AMP_X,
    VIB_AMP_Z,
    VIB_FREQ,
)
from ..geometry import build_pattern
from ..model import generate_xml, resolve_actuated
from ..serialization import normalize_creases
from ..types import SimulationResult
from .base import SimulationBackend

mujoco: Any = None
MjModel: Any = None
mjw: Any = None
_IMPORT_ERROR: BaseException | None = None
try:  # pragma: no cover - exercised on GPU machines only
    import mujoco as _mujoco
    from mujoco import MjModel as _MjModel

    mujoco, MjModel = _mujoco, _MjModel
except Exception as exc:  # noqa: BLE001 - MuJoCo is optional
    _IMPORT_ERROR = exc

try:  # pragma: no cover - exercised on GPU machines only
    try:
        import mujoco_warp as _mjw
    except ImportError:
        # 3.x shipped as ``mujoco.warp``; report both names when neither works.
        try:
            from mujoco import warp as _mjw
        except Exception as legacy_exc:
            raise ImportError(f"neither 'mujoco_warp' nor 'mujoco.warp' is importable ({legacy_exc})") from legacy_exc
    mjw = _mjw
except Exception as exc:  # noqa: BLE001 - Warp is optional
    if _IMPORT_ERROR is None:
        _IMPORT_ERROR = exc


# ``mujoco_warp.put_model`` takes a batch size per field.  Fields that describe
# the *model* (bodies, joints, geoms, constraints, actuators) become per world.
_BATCHED_FIELDS = (
    # Fields read by ``set_world`` that mujoco-warp 4.0.1 marks as batched
    # (``(*, ...)`` array spec).  Passing any other field to ``put_model``
    # raises "Model field ... is not a batched array field".
    "qpos0",
    "body_pos",
    "body_quat",
    "body_ipos",
    "body_iquat",
    "body_mass",
    "body_inertia",
    "jnt_solref",
    "jnt_solimp",
    "jnt_pos",
    "jnt_axis",
    "jnt_stiffness",
    "jnt_range",
    "jnt_margin",
    "geom_solref",
    "geom_solimp",
    "geom_size",
    "geom_pos",
    "geom_quat",
    "geom_friction",
    "geom_margin",
    "geom_gap",
    "geom_rgba",
    "geom_dataid",
    "eq_solref",
    "eq_solimp",
    "eq_data",
    "site_pos",
    "site_quat",
    "actuator_gainprm",
    "actuator_biasprm",
)

# Constraint / contact buffer floors for one world.  MuJoCo Warp's defaults are
# sized for a handful of constraints; a creased sheet with loop closures needs
# far more, and an undersized ``njmax`` silently drops rows (printing a
# "nefc overflow" warning every step).
_MIN_NJMAX = 512
_MIN_NCONMAX = 256

# Fraction of free device memory a suggested batch may use, plus a fixed
# per-world allowance for solver scratch space.
_HEADROOM = 0.7

# How many times the fold drive is recomputed over one run (see ``_step_batch``).
_CTRL_STAGES = 400
_WORLD_OVERHEAD_BYTES = 1 << 20


# ----------------------------------------------------------------------
# Small array helpers (Warp arrays have no item assignment)
# ----------------------------------------------------------------------


def _to_numpy(array: Any) -> np.ndarray:
    """Read a Warp array back to host (``numpy()`` is a copy)."""
    if array is None:
        return np.zeros((0,), dtype=float)
    getter = getattr(array, "numpy", None)
    if callable(getter):
        return np.asarray(getter())
    return np.asarray(array)


def _assign(target: Any, values: Any) -> None:
    """Overwrite a whole Warp array in place.

    ``wp.array.assign(host_array)`` is Warp's documented whole-array write
    (``wp.copy`` is the kernel-facing equivalent); a plain object lands in a
    numpy-aware stub.
    """
    values = np.asarray(values)
    assigner = getattr(target, "assign", None)
    if callable(assigner):
        assigner(values)
        return
    target[:] = values


def _fill(target: Any, value: float) -> None:
    shape = tuple(getattr(target, "shape", ()) or ())
    _assign(target, np.full(shape, value, dtype=np.float32))


def _set_item(target: Any, index: Any, values: Any) -> None:
    """``target[index] = values`` for a Warp array (read, update, assign)."""
    current = _to_numpy(target)
    current[index] = values
    _assign(target, current)


def _item(array: Any, index: Any) -> np.ndarray:
    """``array[index]`` as a host copy."""
    return np.array(_to_numpy(array)[index], copy=True)


# ----------------------------------------------------------------------
# Buffer sizing
# ----------------------------------------------------------------------


def _contact_floor(classic: Any) -> int:
    """``nconmax`` large enough for one world's contacts."""
    ncon = int(getattr(classic, "ncon", 0) or 0)
    ngeom = int(getattr(classic, "ngeom", 0) or 0)
    return max(_MIN_NCONMAX, 2 * ncon, 8 * ngeom)


def _constraint_floor(classic: Any) -> int:
    """``njmax`` large enough for one world's constraints."""
    nefc = int(getattr(classic, "nefc", 0) or 0)
    neq = int(getattr(classic, "neq", 0) or 0)
    nv = int(getattr(classic, "nv", 0) or 0)
    return max(_MIN_NJMAX, 2 * nefc, 8 * neq, 4 * nv)


def _model_margin_bytes(classic: Any) -> int:
    """Rough device bytes one extra world needs (model fields + solver scratch)."""
    total = 0
    for name in _BATCHED_FIELDS:
        field = getattr(classic, name, None)
        if field is None:
            continue
        total += int(np.asarray(field).size) * 4
    total += _constraint_floor(classic) * 64
    total += _contact_floor(classic) * 64
    total += 32 * int(getattr(classic, "nv", 0) or 0) * 4
    return total + _WORLD_OVERHEAD_BYTES


def _first_int(info: dict, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in info:
            try:
                return int(info[key])
            except (TypeError, ValueError):
                return None
    return None


def _free_device_memory() -> int | None:
    """Free device bytes, or ``None`` when Warp cannot report them."""
    try:
        import warp as wp

        info = wp.get_device().context.device_info
    except Exception:  # noqa: BLE001 - diagnostics only
        return None
    free = _first_int(info, ("freeMemory", "free_memory"))
    if free is None:
        free = _first_int(info, ("totalGlobalMem", "total_memory"))
    return free


def _power_of_two_below(value: float) -> int:
    """Largest power of two ``<= value`` (at least 1)."""
    if value < 1:
        return 1
    return int(2.0 ** np.floor(np.log2(value)))


def suggest_batch_size(classic: Any, minimum: int = 1) -> int:
    """Largest power-of-two batch that fits comfortably in free GPU memory."""
    free = _free_device_memory()
    per_world = _model_margin_bytes(classic)
    if not free or per_world <= 0:
        return max(1, int(minimum))
    return max(int(minimum), _power_of_two_below(_HEADROOM * free / per_world))


def _batch_sizes(classic: Any, nworld: int) -> dict[str, int]:
    return {name: nworld for name in _BATCHED_FIELDS if getattr(classic, name, None) is not None}


def _batchable(template: dict[str, Any]) -> bool:
    """Whether a model can be tiled into per-world copies (meshes + graph)."""
    classic = template["classic_model"]
    if int(getattr(classic, "nmesh", 0)) <= 0:
        return False
    try:
        import warp as wp
    except Exception:  # noqa: BLE001
        return False
    return all(hasattr(wp, name) for name in ("ScopedCapture", "capture_launch", "synchronize_device"))


def _release(batch: _Batch) -> None:
    """Drop references to a batch's device arrays so they can be collected."""
    try:
        import warp as wp

        wp.synchronize_device()
    except Exception as exc:  # noqa: BLE001 - best effort
        warnings.warn(f"could not synchronise Warp device: {exc}", RuntimeWarning, stacklevel=2)
    batch.graph = None
    batch.data = None  # type: ignore[assignment]
    batch.warp_model = None  # type: ignore[assignment]


# ----------------------------------------------------------------------
# Batched device model
# ----------------------------------------------------------------------


class _Batch:
    """One Warp model stepped over ``nworld`` independent simulations.

    ``classic`` is the host model with its meshes duplicated ``nworld`` times:
    ``mesh_vert`` / ``mesh_face`` are shared by every world in ``mujoco-warp``,
    so world ``w`` points its geoms at its own copy through ``geom_dataid``.
    The device model, its derived constants and the captured step graph are
    built once per batch size, so every generation of an optimiser reuses them.
    """

    def __init__(
        self,
        backend: WarpBackend,
        template: dict[str, Any],
        nworld: int,
        worlds: list[dict[str, Any]] | None = None,
    ) -> None:
        if mjw is None:
            raise RuntimeError("mujoco-warp is not available")
        self.nworld = int(nworld)
        self.info = template["info"]
        self.template = template["classic_model"]
        self.worlds = list(worlds) if worlds is not None else None
        if self.worlds is not None and len(self.worlds) > self.nworld:
            raise ValueError("more worlds than the batch can hold")
        self.classic = _replicate_meshes(template["classic_model"], self.nworld, template.get("xml"))
        mesh_vert = getattr(template["warp_model"], "mesh_vert", None)
        if mesh_vert is None:
            mesh_vert = template["classic_model"].mesh_vert
        self.mesh_vert = np.array(_to_numpy(mesh_vert), copy=True)
        batch_sizes = _batch_sizes(self.classic, self.nworld) if self.nworld > 1 else {}
        backend._put_model(self.classic, batch_sizes=batch_sizes or None)
        self.warp_model = backend._last_put_model
        if nworld > 1 and self.mesh_vert.shape[0] == int(self.classic.nmesh) and hasattr(self.warp_model, "mesh_vert"):
            # ``put_model`` broadcast the single-world meshes over every world;
            # install the tiled host copy instead.
            _assign(self.warp_model.mesh_vert, self.mesh_vert)
        # ``set_world`` needs the device data (``set_const`` is applied later).
        t0 = time.perf_counter()
        self.data = backend._make_data(self.classic, nworld=self.nworld)
        backend.last_timings["make_data"] = time.perf_counter() - t0
        backend.last_timings["nworld"] = float(self.nworld)
        if self.worlds is not None:
            # Every candidate's per-world model fields (targets are carried by
            # actuator / equality data, offsets by body and joint fields).
            for world, bundle in enumerate(self.worlds):
                self.set_world(world, bundle["classic_model"])
        self.nmesh = int(self.classic.nmesh) // self.nworld if self.nworld else 0
        self.paper_mask = np.asarray(template["paper_mask"])
        self.hinge_adr = _hinge_addresses(self.classic, self.info)
        self.table_x_qadr = int(self.classic.joint("table_x").qposadr[0])
        self.table_x_dadr = int(self.classic.joint("table_x").dofadr[0])
        self.table_z_qadr = int(self.classic.joint("table_z").qposadr[0])
        self.table_z_dadr = int(self.classic.joint("table_z").dofadr[0])
        self.mesh_geom = np.nonzero(np.asarray(self.classic.geom_type) == mujoco.mjtGeom.mjGEOM_MESH)[0]
        self.mesh_dataid = np.asarray(template["mesh_dataid"])[np.asarray(self.classic.geom_dataid)[self.mesh_geom]]
        self.graph = None
        self._capture()

    # -- capture ----------------------------------------------------------

    def _capture(self) -> None:
        import warp as wp

        # One warm-up step initialises derived state outside the graph.
        mjw.step(self.warp_model, self.data)
        wp.synchronize_device()
        with wp.ScopedCapture() as capture:
            mjw.step(self.warp_model, self.data)
        self.graph = capture.graph
        self._wp = wp

    # -- per-world model data ---------------------------------------------

    def set_world(self, world: int, classic: Any) -> None:
        """Copy every per-world field of a validated host model into ``world``."""
        if not 0 <= world < self.nworld:
            raise IndexError(f"world {world} outside a batch of {self.nworld}")
        sets = getattr(getattr(self.warp_model, "fields", None), "get", None)
        for name in _BATCHED_FIELDS:
            if sets is not None and sets(name) is None:
                continue
            src = getattr(classic, name, None)
            if src is not None:
                _set_item(getattr(self.warp_model, name), (world,), src)
        if self.mesh_geom.size:
            per_world = self.mesh_dataid + world * self.nmesh
            target = self.warp_model.geom_dataid
            if world == 0:
                _assign(target, np.asarray(self.classic.geom_dataid, dtype=np.int32))
            else:
                _set_item(target, (world, self.mesh_geom), per_world)

    def finish(self) -> None:
        """Recompute derived constants after the ``set_world`` calls."""
        if hasattr(mjw, "set_const"):
            mjw.set_const(self.warp_model, self.data)

    def set_meshes(self, mesh_vert: np.ndarray) -> None:
        """Overwrite the shared mesh vertices in place (constant topology only)."""
        if mesh_vert.shape != self.mesh_vert.shape:
            raise ValueError(f"mesh vertices changed shape {mesh_vert.shape} -> {self.mesh_vert.shape}")
        self.mesh_vert = np.array(mesh_vert, copy=True)
        if hasattr(self.warp_model, "mesh_vert"):
            _assign(self.warp_model.mesh_vert, self.mesh_vert)

    # -- stepping ---------------------------------------------------------

    def reset(self) -> None:
        """Reset every world to a flat sheet (``qpos0``, zero velocity)."""
        qpos0 = np.asarray(self.classic.qpos0, dtype=np.float32)
        _assign(self.data.qpos, np.tile(qpos0, (self.nworld, 1)))
        if getattr(self.data, "qvel", None) is not None:
            _fill(self.data.qvel, 0.0)

    def step(self) -> None:
        self._wp.capture_launch(self.graph)

    def forward_centroids(self) -> np.ndarray:
        xpos = _to_numpy(self.data.geom_xpos)
        return np.asarray(xpos[:, self.paper_mask, :2], dtype=float).mean(axis=1)

    def hinge_qpos(self) -> np.ndarray:
        qpos = _to_numpy(self.data.qpos)
        if not self.hinge_adr.size:
            return np.zeros((self.nworld, 0), dtype=float)
        return np.asarray(qpos[:, self.hinge_adr], dtype=float)


def _hinge_addresses(classic: Any, info: dict[str, Any]) -> np.ndarray:
    return np.array([classic.joint(h["name"]).qposadr[0] for h in info.get("hinges", ())], dtype=int)


# ----------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------


class WarpBackend(SimulationBackend):
    """MJX-Warp backend: NVIDIA CUDA GPU, batched ``mjw.step`` via CUDA graph."""

    name = "warp"

    # Compiled models are shared by every ``WarpBackend`` instance: the
    # evaluator builds a fresh engine (and backend) per genome, so a per-instance
    # cache would recompile the same XML thousands of times.  Keyed by XML, so
    # it is correct for any pattern and can never return a stale model.
    _MODEL_CACHE: dict[str, dict[str, Any]] = {}  # noqa: RUF012 - released via clear_model_cache
    _MODEL_CACHE_LIMIT = 8
    _LOCK = threading.Lock()

    def __init__(self) -> None:
        # Host-side timings of the last compile / data build, in seconds (a
        # profiling run reads these to split setup cost from simulation cost),
        # and the device model of the most recent ``put_model``.
        self.last_timings: dict[str, float] = {}
        self._last_put_model: Any = None
        # One batched device model + captured graph per (XML, world count),
        # reused across generations; building it costs ~2 s on a T4.
        self._batch_cache: dict[tuple[str, int], _Batch] = {}

    def is_available(self) -> bool:
        if mujoco is None or mjw is None:
            return False
        try:
            import warp as wp

            return wp.get_device().is_cuda
        except Exception:  # noqa: BLE001 - no CUDA device means backend unavailable
            return False

    def device_info(self) -> str:
        if mujoco is None or mjw is None:
            return "mujoco-warp not installed"
        try:
            import warp as wp

            dev = wp.get_device()
            return f"warp device={dev}, cuda={dev.is_cuda}"
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            return f"warp not available: {exc}"

    @classmethod
    def clear_model_cache(cls) -> None:
        """Drop every cached device model and batch (frees GPU memory)."""
        with cls._LOCK:
            cls._MODEL_CACHE.clear()

    # ------------------------------------------------------------------
    # Model / state construction
    # ------------------------------------------------------------------

    def build_model(self, xml: str, info: dict[str, Any]) -> Any:
        """Compile MJCF XML into a Warp model (cached by XML text).

        Returns a dict with ``warp_model``, ``classic_model``, ``info``,
        ``mesh_vert``, ``paper_mask`` and ``hinge_qposadr``.  Repeated calls
        with the same XML -- every candidate of one design space -- reuse the
        compiled model, the most expensive host-side step of a one-off run.
        """
        if mjw is None:
            raise RuntimeError(
                "MJX-Warp backend requires 'mujoco-warp' and 'warp-lang' plus an NVIDIA GPU. "
                "Install with: pip install 'origami-engine-mjx[warp]'"
            ) from _IMPORT_ERROR
        with self._LOCK:
            cached = self._MODEL_CACHE.get(xml)
            if cached is None:
                cached = self._compile(xml, info)
                if len(self._MODEL_CACHE) >= self._MODEL_CACHE_LIMIT:
                    self._MODEL_CACHE.pop(next(iter(self._MODEL_CACHE)))
                self._MODEL_CACHE[xml] = cached
        # ``_compile`` already installed the device model; a cache hit has to as
        # well, so ``_make_data`` and the batch path see this XML's model.
        self._last_put_model = cached["warp_model"]
        # ``mesh_vert`` is shared by every batched world; a batched run leaves
        # the per-world copies in it, so restore the model's own mesh first.
        single = np.array(cached["mesh_vert"], copy=True)
        current = getattr(cached["warp_model"], "mesh_vert", None)
        if current is not None and not np.array_equal(_to_numpy(current), single):
            _assign(current, single)
        classic = cached["classic_model"]
        bundle = dict(cached)
        info = dict(info)
        info.setdefault("xml", xml)
        bundle["info"] = info
        bundle["hinge_qposadr"] = _hinge_addresses(classic, info)
        return bundle

    def _compile(self, xml: str, info: dict[str, Any]) -> dict[str, Any]:
        # Compile the host model and its device twin once per design space.
        timings = self.last_timings
        t0 = time.perf_counter()
        classic = MjModel.from_xml_string(xml)
        timings["mjcf_compile"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._put_model(classic)
        timings["put_model"] = time.perf_counter() - t0
        warp_model = self._last_put_model
        mesh_vert = getattr(warp_model, "mesh_vert", None)
        if mesh_vert is None:
            mesh_vert = classic.mesh_vert
        mesh_vert = np.array(_to_numpy(mesh_vert), copy=True)
        if mesh_vert.shape != np.asarray(classic.mesh_vert).shape:
            mesh_vert = np.array(classic.mesh_vert, copy=True)
        mesh_dataid = getattr(classic, "mesh_dataid", None)
        if mesh_dataid is None:
            mesh_dataid = np.arange(int(classic.nmesh))
        try:
            table_id = int(classic.body("table").id)
        except (KeyError, ValueError):  # stub / table-less model: no paper geoms to exclude
            table_id = -1
        return {
            "warp_model": warp_model,
            "classic_model": classic,
            "info": info,
            "mesh_vert": mesh_vert,
            "mesh_dataid": np.asarray(mesh_dataid),
            "table_id": table_id,
            "paper_mask": np.asarray(classic.geom_bodyid) != table_id,
            "xml": xml,
        }

    def _put_model(self, classic: Any, batch_sizes: dict[str, int] | None = None) -> None:
        # ``mujoco-warp`` >= 4.0 names the device-model constructor ``put_model``;
        # 3.x (``mujoco.warp``) called it ``make_model``.
        put_model = getattr(mjw, "put_model", None) or getattr(mjw, "make_model", None)
        if put_model is None:
            raise RuntimeError(
                "installed mujoco-warp exposes neither 'put_model' nor 'make_model'; "
                "install mujoco-warp>=4.0 and warp-lang>=1.15"
            ) from _IMPORT_ERROR
        if batch_sizes:
            self._last_put_model = put_model(classic, batch_sizes=batch_sizes)
        else:
            self._last_put_model = put_model(classic)

    def make_state(self, model_bundle: dict[str, Any]) -> Any:
        # ``mjw.make_data`` takes the *host* ``mujoco.MjModel`` in 4.0 (3.x took
        # the Warp model), so it is given the classic model kept in the bundle.
        classic = model_bundle.get("classic_model")
        if classic is None:
            return mjw.make_data(model_bundle["warp_model"])
        return self._make_data(classic)

    def _make_data(self, classic: Any, **kwargs: Any) -> Any:
        """``mjw.make_data`` with constraint / contact buffers big enough to run.

        ``mujoco-warp``'s defaults (~64 constraint rows) silently drop
        constraints for articulated models like a creased sheet, which changes
        the physics.  Explicit ``nconmax`` / ``njmax`` keep the caller's choice.
        """
        kwargs.setdefault("nconmax", _contact_floor(classic))
        kwargs.setdefault("njmax", _constraint_floor(classic))
        # ``make_data`` sizes every device array from the batch it is given;
        # a batched model needs one world per batched field.
        nworld = kwargs.pop("nworld", None)
        if nworld is not None:
            kwargs["nworld"] = int(nworld)
        try:
            return mjw.make_data(classic, **kwargs)
        except TypeError:  # 3.x ``make_data(warp_model)``
            warp_model = getattr(self, "_last_put_model", None)
            if warp_model is None or warp_model is classic:
                raise
            return mjw.make_data(warp_model)
        except AttributeError as exc:  # ``njmax=None`` on pre-4.0 builds
            if "njmax" not in str(exc) and "nconmax" not in str(exc):
                raise
            return mjw.make_data(classic, **kwargs)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(creases, coll_thick: float | None, actuated) -> tuple[Any, str, dict[str, Any]]:
        normalized = normalize_creases(creases)
        pat = build_pattern(normalized)
        if not pat.valid:
            raise ValueError(f"invalid crease pattern: {pat.reason}")
        xml, info = generate_xml(pat, normalized, coll_thick=coll_thick, actuated=actuated)
        return pat, xml, info

    # ------------------------------------------------------------------
    # Single simulation
    # ------------------------------------------------------------------

    def run(
        self,
        creases,
        duration: float = 8.0,
        fold_duration: float = 3.0,
        vibrate: bool = True,
        amp_x: float = VIB_AMP_X,
        amp_z: float = VIB_AMP_Z,
        freq_hz: float = VIB_FREQ,
        coll_thick: float | None = None,
        record: bool = False,
        record_every: int = 1,
        *,
        actuated=None,
        actuation_schedule=None,
        **kwargs: Any,  # accept future backend kwargs
    ) -> SimulationResult:
        """Simulate one crease pattern (see :meth:`JaxBackend.run`).

        ``record=True`` stores the same per-frame trajectory dict in
        ``result.extra["trajectory"]``; recording copies state to host memory
        every ``record_every`` steps.  ``actuated`` and ``actuation_schedule``
        behave exactly as in the JAX backend.
        """
        if duration <= 0:
            raise ValueError("duration must be positive.")

        normalized = normalize_creases(creases)
        pat = build_pattern(normalized)
        if not pat.valid:
            raise ValueError(f"invalid crease pattern: {pat.reason}")

        xml, info = generate_xml(pat, normalized, coll_thick=coll_thick, actuated=actuated)
        bundle = self.build_model(xml, info)
        batch = _Batch(self, bundle, 1)
        return self._simulate(
            batch,
            bundle,
            pat,
            duration=duration,
            fold_duration=fold_duration,
            vibrate=vibrate,
            amp_x=amp_x,
            amp_z=amp_z,
            freq_hz=freq_hz,
            record=record,
            record_every=record_every,
            actuation_schedule=actuation_schedule,
        )

    def _simulate(
        self,
        batch: _Batch,
        bundle: dict[str, Any],
        pat: Any,
        *,
        duration: float,
        fold_duration: float,
        vibrate: bool,
        amp_x: float,
        amp_z: float,
        freq_hz: float,
        record: bool = False,
        record_every: int = 1,
        actuation_schedule=None,
        world: int = 0,
        actuation=None,
    ) -> SimulationResult:
        """Step one world of ``batch`` and package a :class:`SimulationResult`."""
        data = batch.data
        classic = batch.classic
        info = dict(bundle["info"])
        world_id = bundle.get("world")
        idx = (0,) if world_id is None else (int(world_id),)

        hinge_adr = batch.hinge_adr
        table_x_qadr, table_x_dadr = batch.table_x_qadr, batch.table_x_dadr
        table_z_qadr, table_z_dadr = batch.table_z_qadr, batch.table_z_dadr
        info["table_qposadr"] = (table_x_qadr, table_z_qadr)
        info["hinge_qposadr"] = hinge_adr

        targets = np.asarray(info["targets"], dtype=np.float32)
        actuator_hinges = np.asarray(info["actuator_hinges"], dtype=int)
        actuator_adr = hinge_adr[actuator_hinges] if actuator_hinges.size else np.zeros(0, dtype=int)
        model_mask = np.asarray(info["actuated"], dtype=bool)
        schedule_static = actuation_schedule is None
        current_mask = model_mask.copy()
        if actuation is not None:
            current_mask = resolve_actuated(actuation, len(hinge_adr))

        # Zero hinge qpos.
        if len(hinge_adr):
            _set_item(data.qpos, idx + (hinge_adr,), 0.0)

        paper_mask = batch.paper_mask
        initial_centroid = self._centroid_from_data(data, paper_mask, int(world_id or 0))

        timestep = float(classic.opt.timestep)
        n_steps = round(duration / timestep)
        record_every = max(1, int(record_every))
        times: list[float] = []
        qpos_hist: list[np.ndarray] = []
        qvel_hist: list[np.ndarray] = []
        centroid_hist: list[np.ndarray] = []
        mask_hist: list[np.ndarray] = []

        def capture(t: float) -> None:
            times.append(float(t))
            qpos_hist.append(_item(data.qpos, idx))
            qvel_hist.append(_item(data.qvel, idx))
            centroid_hist.append(self._centroid_from_data(data, paper_mask, int(world_id or 0)))
            if not schedule_static:
                mask_hist.append(current_mask.copy())

        if record:
            capture(0.0)

        for step_i in range(n_steps):
            t = step_i * timestep
            frac = min(t / fold_duration, 1.0) if fold_duration > 0 else 1.0
            s = frac * frac * (3.0 - 2.0 * frac)
            if not schedule_static:
                current_mask = resolve_actuated(actuation_schedule(t), len(hinge_adr))
            active = current_mask[model_mask]
            if actuator_hinges.size:
                drive = s * targets[actuator_hinges]
                neutral = _item(data.qpos, idx + (actuator_adr,))
                _set_item(data.ctrl, idx, np.where(active, drive, neutral))
            if vibrate:
                w = 2.0 * np.pi * freq_hz
                _set_item(data.qpos, idx + (table_x_qadr,), amp_x * np.sin(w * t))
                _set_item(data.qvel, idx + (table_x_dadr,), amp_x * w * np.cos(w * t))
                _set_item(data.qpos, idx + (table_z_qadr,), amp_z * np.sin(w * t))
                _set_item(data.qvel, idx + (table_z_dadr,), amp_z * w * np.cos(w * t))
            # A recorded run keeps real ``mjw.step`` calls (per-frame readback);
            # otherwise the captured graph replays the same step.
            if record:
                mjw.step(batch.warp_model, data)
            else:
                batch.step()
            if record and ((step_i + 1) % record_every == 0 or step_i + 1 == n_steps):
                capture((step_i + 1) * timestep)

        final_qpos = _item(data.qpos, idx)
        if not np.isfinite(final_qpos).all():
            raise FloatingPointError("unstable simulation: non-finite joint coordinates produced.")

        final_centroid = self._centroid_from_data(data, paper_mask, int(world_id or 0))
        info["actuated"] = current_mask
        info["hinge_qpos"] = final_qpos[hinge_adr] if len(hinge_adr) else np.zeros(0, dtype=float)

        extra: dict[str, Any] = {}
        if record:
            if not times:
                capture(n_steps * timestep)
            extra["trajectory"] = {
                "time": np.asarray(times, dtype=float),
                "qpos": np.asarray(qpos_hist),
                "qvel": np.asarray(qvel_hist),
                "centroid": np.asarray(centroid_hist, dtype=float),
                # Hinge qpos columns, so plots never mix in stretch/panel joints.
                "hinge_qposadr": np.asarray(hinge_adr, dtype=int),
            }
            if mask_hist:
                extra["trajectory"]["actuated"] = np.asarray(mask_hist, dtype=bool)

        return SimulationResult(
            model=bundle,
            data=data,
            pattern=pat,
            info=info,
            initial_centroid=initial_centroid,
            final_centroid=final_centroid,
            backend="warp",
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Batch population evaluation
    # ------------------------------------------------------------------

    def make_batch(self, template: dict[str, Any], nworld: int) -> _Batch:
        """Build (or reuse) a batched Warp model over ``nworld`` worlds."""
        key = (template["xml"], int(nworld))
        batch = self._batch_cache.get(key)
        if batch is None:
            batch = _Batch(self, template, nworld)
            self._batch_cache[key] = batch
        return batch

    def run_batch(
        self,
        population: list,
        duration: float = 8.0,
        fold_duration: float = 3.0,
        vibrate: bool = True,
        amp_x: float = VIB_AMP_X,
        amp_z: float = VIB_AMP_Z,
        freq_hz: float = VIB_FREQ,
        batch_size: int | None = None,
        actuated=None,
        actuation_schedule=None,
    ) -> np.ndarray:
        """Score a population in one batched Warp model.

        Every candidate must generate the same XML topology (the ordinary case
        for a single design space: the creases differ, the face layout does
        not).  Then one world per candidate is stepped inside one captured CUDA
        graph, so the whole population costs roughly one single-candidate step
        loop.  Scoring is the +x centroid displacement, as in :class:`JaxBackend`.

        Candidates that cannot be batched fall back to sequential :meth:`run`
        calls, and a single failing candidate yields ``nan`` (the evaluator
        turns that into its penalty score) instead of aborting the generation.
        """
        population = list(population)
        if not population:
            return np.zeros((0,), dtype=float)
        if batch_size is not None and int(batch_size) < len(population):
            chunks = [
                self.run_batch(population[i : i + batch_size], **_batch_kwargs(locals()))
                for i in range(0, len(population), int(batch_size))
            ]
            return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=float)

        prepared: list[tuple[Any, str, dict[str, Any]] | None] = []
        for creases in population:
            try:
                prepared.append(self._prepare(creases, None, actuated))
            except Exception:  # noqa: BLE001 - one bad individual must not kill the batch
                prepared.append(None)

        scores = np.full(len(population), np.nan, dtype=float)
        valid = [(i, item) for i, item in enumerate(prepared) if item is not None]
        if not valid:
            return scores
        # Same XML for every candidate => one batched device model.
        templates = {item[1][1] for item in valid}
        if len(templates) == 1:
            try:
                batched = self._run_batched_worlds(
                    valid,
                    duration=duration,
                    fold_duration=fold_duration,
                    vibrate=vibrate,
                    amp_x=amp_x,
                    amp_z=amp_z,
                    freq_hz=freq_hz,
                )
            except Exception as exc:  # noqa: BLE001 - degrade to sequential scoring
                warnings.warn(
                    f"batched Warp evaluation failed ({exc}); scoring candidates one by one",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                for (index, _), score in zip(valid, batched, strict=True):
                    scores[index] = score
                return scores
        for i, (pat, xml, info) in valid:
            try:
                result = self.run(
                    population[i],
                    duration=duration,
                    fold_duration=fold_duration,
                    vibrate=vibrate,
                    amp_x=amp_x,
                    amp_z=amp_z,
                    freq_hz=freq_hz,
                    actuated=actuated,
                    actuation_schedule=actuation_schedule,
                )
            except Exception as exc:  # noqa: BLE001 - one bad individual must not kill the batch
                warnings.warn(f"warp candidate {i} failed: {exc}", RuntimeWarning, stacklevel=2)
                continue
            delta = result.final_centroid - result.initial_centroid
            scores[i] = float(delta[0])
        return scores

    def _run_batched_worlds(
        self,
        items: list[tuple[int, tuple[Any, str, dict[str, Any]]]],
        *,
        duration: float,
        fold_duration: float,
        vibrate: bool,
        amp_x: float,
        amp_z: float,
        freq_hz: float,
    ) -> np.ndarray:
        """Run one world per ``item`` (same topology) and return +x scores."""
        bundles = [self.build_model(item[1][1], item[1][2]) for item in items]
        nworld = len(items)
        batch = self.make_batch(bundles[0], nworld)
        for world, bundle in enumerate(bundles):
            batch.set_world(world, bundle["classic_model"])
        batch.finish()

        initial_centroids = np.asarray(_to_numpy(batch.data.geom_xpos)[:, batch.paper_mask, :2], dtype=float).mean(
            axis=1
        )
        return self._step_batch(
            batch,
            duration=duration,
            fold_duration=fold_duration,
            vibrate=vibrate,
            amp_x=amp_x,
            amp_z=amp_z,
            freq_hz=freq_hz,
            initial_centroids=initial_centroids,
        )

    def _step_batch(
        self,
        batch: _Batch,
        *,
        duration: float,
        fold_duration: float,
        vibrate: bool,
        amp_x: float,
        amp_z: float,
        freq_hz: float,
        initial_centroids: np.ndarray,
    ) -> np.ndarray:
        """Drive every world of ``batch`` and return +x centroid displacement."""
        data = batch.data
        classic = batch.classic
        info = batch.info
        targets = np.asarray(info["targets"], dtype=np.float32)
        actuator_hinges = np.asarray(info["actuator_hinges"], dtype=int)
        hinge_adr = batch.hinge_adr
        actuator_adr = hinge_adr[actuator_hinges] if actuator_hinges.size else np.zeros(0, dtype=int)
        model_mask = np.asarray(info["actuated"], dtype=bool)
        timestep = float(classic.opt.timestep)
        n_steps = round(duration / timestep)

        # Zero hinge qpos, exactly like ``run``.
        if hinge_adr.size:
            qpos = _to_numpy(data.qpos)
            qpos[:, hinge_adr] = 0.0
            _assign(data.qpos, qpos)
        if getattr(data, "qvel", None) is not None:
            _fill(data.qvel, 0.0)
        if actuator_hinges.size:
            # Zero the drive so every world starts from a flat sheet.
            _fill(data.ctrl, 0.0)

        # The drive advances in a fixed number of stages, so that the square
        # root wave (the fold profile) is re-applied only a few hundred times
        # instead of on every one of the ~8000 steps.  Each stage writes one
        # constant ``ctrl`` row, then launches the captured step in a tight
        # loop; the graph reads ``data.ctrl`` on every launch.
        stages = int(min(n_steps, _CTRL_STAGES))
        nu = int(getattr(classic, "nu", 0) or 0)
        readback = 0.0
        launched = 0.0
        for stage in range(stages):
            t0 = time.perf_counter()
            start_step = (stage * n_steps) // stages
            end_step = ((stage + 1) * n_steps) // stages
            t = start_step * timestep
            frac = min(t / fold_duration, 1.0) if fold_duration > 0 else 1.0
            s = frac * frac * (3.0 - 2.0 * frac)
            if actuator_hinges.size:
                drive = s * targets[actuator_hinges]
                neutral = _to_numpy(data.qpos)[:, actuator_adr] if actuator_adr.size else np.zeros((batch.nworld, 0))
                ctrl = np.where(model_mask[None, :], drive[None, :], neutral).astype(np.float32)
                _assign(data.ctrl, np.broadcast_to(ctrl, (batch.nworld, nu)))
            readback += time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(end_step - start_step):
                batch.step()
            launched += time.perf_counter() - t0

        self.last_timings["launch"] = launched
        self.last_timings["readback"] = readback
        self.last_timings["steps"] = float(n_steps)

        final_qpos = _to_numpy(data.qpos)
        if not np.isfinite(final_qpos).all():
            raise FloatingPointError("unstable simulation: non-finite joint coordinates produced.")
        final_centroids = batch.forward_centroids()
        return (final_centroids - initial_centroids)[:, 0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _centroid_from_data(data, paper_mask: np.ndarray, world: int | None = None) -> np.ndarray:
        xpos = np.array(_to_numpy(data.geom_xpos), copy=True)
        # ``mujoco-warp`` 4.x arrays are (nworld, ngeom, 3); select our world.
        if xpos.ndim == 3:
            xpos = xpos[0 if world is None else world]
        pts = xpos[paper_mask, :2]
        return pts.mean(axis=0)


def _batch_kwargs(frame: dict[str, Any]) -> dict[str, Any]:
    """Forward the shared arguments of a ``run_batch`` chunk."""
    return {
        key: frame[key]
        for key in (
            "duration",
            "fold_duration",
            "vibrate",
            "amp_x",
            "amp_z",
            "freq_hz",
            "actuated",
            "actuation_schedule",
        )
    }


# ----------------------------------------------------------------------
# Mesh replication (one mesh copy per world)
# ----------------------------------------------------------------------


def _mesh_name(name: str, world: int) -> str:
    return name if world == 0 else f"{name}__w{world}"


def _mesh_asset_vertices(spec: Any) -> list[np.ndarray]:
    """``uservert`` triangles of every mesh in an ``MjSpec``, as authored."""
    out = []
    for mesh in spec.meshes:
        verts = np.asarray(list(mesh.uservert), dtype=float).reshape(-1, 3)
        out.append(verts)
    return out


def _replicate_meshes(classic: Any, nworld: int, template_xml: str | None = None) -> Any:
    """Copy ``classic`` with its meshes duplicated once per world.

    Returns ``classic`` itself for a single world.  Mesh / vertex / face counts
    are unchanged and the first copy keeps the original mesh indices, so a
    world's geoms are selected with ``mesh_dataid + world * nmesh``.
    A model without meshes is returned unchanged.

    Copies are created from the *asset* definition (``uservert`` / ``userface``
    as written in the XML), not from ``mesh_vert``: MuJoCo re-processes an asset
    mesh (scaling, inertia frame, recentring) but uses a user mesh verbatim, so
    only the authored vertices reproduce the mesh exactly.
    """
    if nworld <= 1:
        return classic
    if int(getattr(classic, "nmesh", 0)) == 0:
        # No meshes to share, so every world already sees the same geometry.
        return classic
    if template_xml is None:
        raise RuntimeError("cannot replicate meshes without the model's source XML")
    spec = mujoco.MjSpec.from_string(template_xml)
    originals = _mesh_asset_vertices(spec)
    if len(originals) != int(classic.nmesh):
        raise RuntimeError(f"mesh table mismatch: XML declares {len(originals)} meshes, model has {int(classic.nmesh)}")
    authored = list(spec.meshes)  # snapshot: ``add_mesh`` grows the spec
    for world in range(1, nworld):
        for mesh in authored:
            copy = spec.add_mesh()
            copy.name = _mesh_name(mesh.name, world)
            copy.uservert = [float(v) for v in mesh.uservert]
            copy_faces = list(mesh.userface)
            if copy_faces:
                copy.userface = [int(f) for f in copy_faces]
    model = spec.compile()
    if int(model.nmesh) != nworld * len(originals):
        raise RuntimeError(f"mesh replication produced {model.nmesh} meshes, expected {nworld * len(originals)}")

    # Every copy must be the original mesh.  A mismatch would silently deform
    # the sheet in some worlds, so refuse to batch instead.
    def _vertices(mesh_id: int) -> np.ndarray:
        adr, num = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        return np.asarray(model.mesh_vert[adr : adr + num], dtype=np.float64)

    def _faces(mesh_id: int) -> np.ndarray:
        adr, num = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        return np.asarray(model.mesh_face[adr : adr + num])

    count = len(originals)
    refs = [(_vertices(i), _faces(i)) for i in range(count)]
    for copy_id in range(count, model.nmesh):
        verts, faces = _vertices(copy_id), _faces(copy_id)
        ref_verts, ref_faces = refs[copy_id % count]
        if not np.allclose(verts, ref_verts, atol=1e-6) or not np.array_equal(faces, ref_faces):
            raise RuntimeError(
                f"mesh copy {copy_id} does not match mesh {copy_id % count}; "
                "candidate batching requires identical mesh geometry"
            )
    return model


__all__ = ["WarpBackend", "suggest_batch_size"]
