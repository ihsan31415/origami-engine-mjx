from __future__ import annotations

"""MJX-Warp simulation backend.

Uses ``mujoco.warp`` (MuJoCo Warp, NVIDIA CUDA) to step the same MJCF model on
GPU.  This backend **requires an NVIDIA CUDA GPU** and the ``mujoco-warp`` +
``warp-lang`` packages.  Population evaluation currently loops per individual;
native batched Warp execution is not implemented yet.

All Warp imports are lazy — importing this module does not initialise CUDA.

``mujoco-warp`` 3.x shipped as ``mujoco.warp`` inside MuJoCo, while 4.x is the
standalone ``mujoco_warp`` package, so :func:`_import_mjwarp` tries the
standalone module first and falls back to the legacy import.
"""

import importlib
from typing import Any

import numpy as np

from ..constants import VIB_AMP_X, VIB_AMP_Z, VIB_FREQ
from ..geometry import build_pattern
from ..model import generate_xml, resolve_actuated
from ..serialization import normalize_creases
from ..types import SimulationResult
from .base import SimulationBackend

# Lazy imports.
_IMPORT_ERROR: BaseException | None = None


def _import_mjwarp():
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
    import mujoco
    from mujoco import MjModel

    mjw = _import_mjwarp()
except Exception as exc:  # noqa: BLE001
    mujoco = None  # type: ignore[assignment]
    mjw = None  # type: ignore[assignment]
    MjModel = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


class WarpBackend(SimulationBackend):
    """MJX-Warp backend: NVIDIA CUDA GPU, ``mjw.step`` per individual."""

    name = "warp"

    def is_available(self) -> bool:
        if mujoco is None or mjw is None:
            return False
        # Check for CUDA availability.
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

    # ------------------------------------------------------------------
    # Model / state construction
    # ------------------------------------------------------------------

    def build_model(self, xml: str, info: dict[str, Any]) -> Any:
        """Compile MJCF XML into a Warp model.

        Returns a dict with keys ``warp_model``, ``classic_model``, ``info``.
        """
        if mjw is None:
            raise RuntimeError(
                "MJX-Warp backend requires 'mujoco-warp' and 'warp-lang' plus an NVIDIA GPU. "
                "Install with: pip install 'origami-engine-mjx[warp]'"
            ) from _IMPORT_ERROR
        # ``mujoco-warp`` >= 4.0 names the device-model constructor ``put_model``;
        # 3.x (``mujoco.warp``) called it ``make_model``.
        put_model = getattr(mjw, "put_model", None) or getattr(mjw, "make_model", None)
        if put_model is None:
            raise RuntimeError(
                "installed mujoco-warp exposes neither 'put_model' nor 'make_model'; "
                "install mujoco-warp>=4.0 and warp-lang>=1.15"
            ) from _IMPORT_ERROR
        classic = MjModel.from_xml_string(xml)
        warp_model = put_model(classic)
        return {"warp_model": warp_model, "classic_model": classic, "info": info}

    def make_state(self, model_bundle: dict[str, Any]) -> Any:
        # ``mjw.make_data`` takes the *host* ``mujoco.MjModel`` in 4.0 (3.x took
        # the Warp model), so it is given the classic model kept in the bundle.
        classic = model_bundle.get("classic_model")
        if classic is None:
            return mjw.make_data(model_bundle["warp_model"])
        return mjw.make_data(classic)

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
        warp_model = bundle["warp_model"]
        classic = bundle["classic_model"]

        # ``mujoco-warp`` 4.x arrays are (nworld, ...); this backend simulates
        # one world, so every device array is indexed at world 0.
        world = bundle.get("world")
        idx = (0,) if world is None else (world,)
        data = self.make_state(bundle)

        hinge_adr = np.array([classic.joint(h["name"]).qposadr[0] for h in info["hinges"]], dtype=int)
        table_x_qadr = int(classic.joint("table_x").qposadr[0])
        table_x_dadr = int(classic.joint("table_x").dofadr[0])
        table_z_qadr = int(classic.joint("table_z").qposadr[0])
        table_z_dadr = int(classic.joint("table_z").dofadr[0])

        info = dict(info)
        info["table_qposadr"] = (table_x_qadr, table_z_qadr)
        info["hinge_qposadr"] = hinge_adr

        targets = np.asarray(info["targets"], dtype=np.float32)
        actuator_hinges = np.asarray(info["actuator_hinges"], dtype=int)
        actuator_adr = hinge_adr[actuator_hinges] if actuator_hinges.size else np.zeros(0, dtype=int)
        model_mask = np.asarray(info["actuated"], dtype=bool)
        schedule_static = actuation_schedule is None
        current_mask = model_mask.copy()

        # Zero hinge qpos.
        if len(hinge_adr):
            data.qpos.numpy()[idx + (hinge_adr,)] = 0.0

        table_id = int(classic.body("table").id)
        geom_bodyid = getattr(warp_model, "geom_bodyid", None)
        if geom_bodyid is None:
            # Fall back to the host model (device array copied to the CPU by ``np.asarray``).
            geom_bodyid = np.asarray(classic.geom_bodyid)
        else:
            geom_bodyid = np.asarray(geom_bodyid)
        paper_mask = geom_bodyid != table_id
        initial_centroid = self._centroid_from_data(data, paper_mask, world)

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
            qpos_hist.append(np.array(data.qpos.numpy()[idx], copy=True))
            qvel_hist.append(np.array(data.qvel.numpy()[idx], copy=True))
            centroid_hist.append(self._centroid_from_data(data, paper_mask, world))
            if not schedule_static:
                mask_hist.append(current_mask.copy())

        if record:
            capture(0.0)

        for step_i in range(n_steps):
            t = step_i * timestep
            frac = min(t / fold_duration, 1.0)
            s = frac * frac * (3.0 - 2.0 * frac)
            if not schedule_static:
                current_mask = resolve_actuated(actuation_schedule(t), len(hinge_adr))
            active = current_mask[model_mask]
            if actuator_hinges.size:
                drive = s * targets[actuator_hinges]
                neutral = data.qpos.numpy()[idx + (actuator_adr,)]
                data.ctrl.numpy()[idx] = np.where(active, drive, neutral)
            if vibrate:
                w = 2.0 * np.pi * freq_hz
                data.qpos.numpy()[idx + (table_x_qadr,)] = amp_x * np.sin(w * t)
                data.qvel.numpy()[idx + (table_x_dadr,)] = amp_x * w * np.cos(w * t)
                data.qpos.numpy()[idx + (table_z_qadr,)] = amp_z * np.sin(w * t)
                data.qvel.numpy()[idx + (table_z_dadr,)] = amp_z * w * np.cos(w * t)
            mjw.step(warp_model, data)
            if record and ((step_i + 1) % record_every == 0 or step_i + 1 == n_steps):
                capture((step_i + 1) * timestep)

        final_qpos = data.qpos.numpy()[idx]
        if not np.isfinite(final_qpos).all():
            raise FloatingPointError("unstable simulation: non-finite joint coordinates produced.")

        final_centroid = self._centroid_from_data(data, paper_mask, world)
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

    def run_batch(
        self,
        population: list,
        duration: float = 8.0,
        fold_duration: float = 3.0,
        vibrate: bool = True,
        amp_x: float = VIB_AMP_X,
        amp_z: float = VIB_AMP_Z,
        freq_hz: float = 75.0,
        batch_size: int | None = None,
        actuated=None,
        actuation_schedule=None,
    ) -> np.ndarray:
        """Evaluate a population sequentially, one individual per ``run``.

        Native batched Warp execution (``mjw.step`` over batched state) is not
        implemented yet, so ``batch_size`` is ignored.  Scoring is always the
        +x centroid displacement.  ``actuated`` / ``actuation_schedule`` are
        forwarded to :meth:`run`.
        """
        scores = []
        for creases in population:
            try:
                result = self.run(
                    creases,
                    duration=duration,
                    fold_duration=fold_duration,
                    vibrate=vibrate,
                    amp_x=amp_x,
                    amp_z=amp_z,
                    freq_hz=freq_hz,
                    actuated=actuated,
                    actuation_schedule=actuation_schedule,
                )
                delta = result.final_centroid - result.initial_centroid
                scores.append(float(delta[0]))
            except Exception:  # noqa: BLE001 - one bad individual must not kill the batch
                scores.append(float("nan"))
        return np.asarray(scores, dtype=float)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _centroid_from_data(data, paper_mask: np.ndarray, world: int | None = None) -> np.ndarray:
        xpos = data.geom_xpos.numpy()
        if world is not None:
            # ``mujoco-warp`` 4.x arrays are (nworld, ngeom, 3); index our world.
            xpos = xpos[world]
        pts = np.asarray(xpos)[paper_mask, :2]
        return pts.mean(axis=0)


__all__ = ["WarpBackend"]
