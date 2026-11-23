from __future__ import annotations

"""MJX-Warp simulation backend.

Uses ``mujoco.warp`` (MuJoCo Warp, NVIDIA CUDA) for high-throughput batched
simulation on GPU.  This backend **requires an NVIDIA CUDA GPU** and the
``mujoco-warp`` + ``warp-lang`` packages.

All Warp imports are lazy — importing this module does not initialise CUDA.
"""

from typing import Any

import numpy as np

from ..constants import VIB_AMP_X, VIB_AMP_Z, VIB_FREQ
from ..geometry import build_pattern
from ..model import generate_xml
from ..serialization import normalize_creases
from ..types import SimulationResult
from .base import SimulationBackend

# Lazy imports.
_IMPORT_ERROR: BaseException | None = None

try:
    import mujoco
    import mujoco.warp as mjw
    from mujoco import MjModel
except Exception as exc:  # noqa: BLE001
    mujoco = None  # type: ignore[assignment]
    mjw = None  # type: ignore[assignment]
    MjModel = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


class WarpBackend(SimulationBackend):
    """MJX-Warp backend: NVIDIA CUDA GPU, batched high-throughput simulation."""

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
        classic = MjModel.from_xml_string(xml)
        warp_model = mjw.make_model(classic)
        return {"warp_model": warp_model, "classic_model": classic, "info": info}

    def make_state(self, model_bundle: dict[str, Any]) -> Any:
        return mjw.make_data(model_bundle["warp_model"])

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
    ) -> SimulationResult:
        if duration <= 0:
            raise ValueError("duration must be positive.")

        normalized = normalize_creases(creases)
        pat = build_pattern(normalized)
        if not pat.valid:
            raise ValueError(f"invalid crease pattern: {pat.reason}")

        xml, info = generate_xml(pat, normalized, coll_thick=coll_thick)
        bundle = self.build_model(xml, info)
        warp_model = bundle["warp_model"]
        classic = bundle["classic_model"]

        data = self.make_state(bundle)

        hinge_adr = np.array([classic.joint(h["name"]).qposadr[0] for h in info["hinges"]], dtype=int)
        table_x_qadr = int(classic.joint("table_x").qposadr[0])
        table_x_dadr = int(classic.joint("table_x").dofadr[0])
        table_z_qadr = int(classic.joint("table_z").qposadr[0])
        table_z_dadr = int(classic.joint("table_z").dofadr[0])

        targets = np.asarray(info["targets"], dtype=np.float32)

        # Zero hinge qpos.
        if len(hinge_adr):
            data.qpos.numpy()[hinge_adr] = 0.0

        table_id = int(classic.body("table").id)
        geom_bodyid = np.asarray(warp_model.geom_bodyid)
        paper_mask = geom_bodyid != table_id
        initial_centroid = self._centroid_from_data(data, paper_mask)

        timestep = float(classic.opt.timestep)
        n_steps = round(duration / timestep)

        for step_i in range(n_steps):
            t = step_i * timestep
            frac = min(t / fold_duration, 1.0)
            s = frac * frac * (3.0 - 2.0 * frac)
            data.ctrl.numpy()[:] = s * targets
            if vibrate:
                w = 2.0 * np.pi * freq_hz
                data.qpos.numpy()[table_x_qadr] = amp_x * np.sin(w * t)
                data.qvel.numpy()[table_x_dadr] = amp_x * w * np.cos(w * t)
                data.qpos.numpy()[table_z_qadr] = amp_z * np.sin(w * t)
                data.qvel.numpy()[table_z_dadr] = amp_z * w * np.cos(w * t)
            mjw.step(warp_model, data)

        final_qpos = data.qpos.numpy()
        if not np.isfinite(final_qpos).all():
            raise FloatingPointError("unstable simulation: non-finite joint coordinates produced.")

        final_centroid = self._centroid_from_data(data, paper_mask)

        return SimulationResult(
            model=bundle,
            data=data,
            pattern=pat,
            info=info,
            initial_centroid=initial_centroid,
            final_centroid=final_centroid,
            backend="warp",
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
    ) -> np.ndarray:
        """Evaluate a population.  Warp backend loops per-individual for now;
        true batched Warp execution (``mjw.step`` over batched state) is a
        Colab-GPU integration target.
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
    def _centroid_from_data(data, paper_mask: np.ndarray) -> np.ndarray:
        xpos = data.geom_xpos.numpy()
        pts = xpos[paper_mask, :2]
        return pts.mean(axis=0)


__all__ = ["WarpBackend"]
