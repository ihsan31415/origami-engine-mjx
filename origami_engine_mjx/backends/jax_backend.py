from __future__ import annotations

"""MJX-JAX simulation backend.

Uses ``mujoco.mjx`` (the JAX implementation of MuJoCo).  Runs on CPU locally
and on GPU/TPU through JAX backends.  State is an immutable JAX pytree, so
callers can apply ``jax.jit``/``jax.vmap``/``jax.lax.scan`` themselves; this
module steps eagerly by default because JIT-compiling ``mjx.step`` on
mesh-contact models is prohibitively slow.
"""

from typing import Any

import numpy as np

from ..constants import VIB_AMP_X, VIB_AMP_Z, VIB_FREQ
from ..geometry import build_pattern
from ..model import generate_xml
from ..serialization import normalize_creases
from ..types import SimulationResult
from .base import SimulationBackend

# Lazy imports — JAX/MJX are optional dependencies.
_IMPORT_ERROR: BaseException | None = None

try:
    import jax
    import jax.numpy as jnp
    from mujoco import MjModel, mjx
except Exception as exc:  # noqa: BLE001
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    mjx = None  # type: ignore[assignment]
    MjModel = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


class JaxBackend(SimulationBackend):
    """MJX-JAX backend: JAX pytree state, eager ``mjx.step`` by default."""

    name = "jax"

    def is_available(self) -> bool:
        return jax is not None and mjx is not None

    def device_info(self) -> str:
        if jax is None:
            return "jax not installed"
        devices = jax.devices()
        backend = jax.default_backend()
        return f"jax backend={backend}, devices={[str(d) for d in devices]}"

    # ------------------------------------------------------------------
    # Model / state construction
    # ------------------------------------------------------------------

    def build_model(self, xml: str, info: dict[str, Any]) -> Any:
        """Compile MJCF XML into an MJX model plus metadata.

        Returns a dict with keys ``mjx_model``, ``classic_model`` (for name
        lookups), ``info``.
        """
        if mjx is None:
            raise RuntimeError(
                "MJX-JAX backend requires 'mujoco-mjx' and 'jax'. Install with: pip install 'origami-engine-mjx[jax]'"
            ) from _IMPORT_ERROR
        classic = MjModel.from_xml_string(xml)
        mjx_model = mjx.put_model(classic)
        return {"mjx_model": mjx_model, "classic_model": classic, "info": info}

    def make_state(self, model_bundle: dict[str, Any]) -> Any:
        """Create initial MJX Data from a model bundle."""
        return mjx.make_data(model_bundle["mjx_model"])

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
        jit: bool = False,
        record: bool = False,
        record_every: int = 1,
        **kwargs: Any,  # accept future backend kwargs
    ) -> SimulationResult:
        """Simulate one crease pattern.

        Set ``record=True`` to store a per-frame trajectory in
        ``result.extra["trajectory"]`` (see
        :class:`~origami_engine_mjx.rendering.Trajectory`): a dict with
        ``time``, ``qpos``, ``qvel`` and ``centroid`` arrays sampled every
        ``record_every`` steps (always including the first and last step).
        Recording costs one host copy per sampled step and does not change the
        physics.
        """
        if duration <= 0:
            raise ValueError("duration must be positive.")

        normalized = normalize_creases(creases)
        pat = build_pattern(normalized)
        if not pat.valid:
            raise ValueError(f"invalid crease pattern: {pat.reason}")

        xml, info = generate_xml(pat, normalized, coll_thick=coll_thick)
        bundle = self.build_model(xml, info)
        mjx_model = bundle["mjx_model"]
        classic = bundle["classic_model"]

        data = self.make_state(bundle)

        # Resolve joint addresses from the classic model for bookkeeping.
        hinge_adr = np.array([classic.joint(h["name"]).qposadr[0] for h in info["hinges"]], dtype=int)
        table_x_qadr = int(classic.joint("table_x").qposadr[0])
        table_x_dadr = int(classic.joint("table_x").dofadr[0])
        table_z_qadr = int(classic.joint("table_z").qposadr[0])
        table_z_dadr = int(classic.joint("table_z").dofadr[0])

        info = dict(info)
        info["table_qposadr"] = (table_x_qadr, table_z_qadr)

        targets = jnp.asarray(info["targets"], dtype=jnp.float32)

        # Zero hinge qpos initially.
        if len(hinge_adr):
            data = data.replace(qpos=data.qpos.at[hinge_adr].set(0.0))

        # Compute initial centroid from geom_xpos (exclude table body).
        table_id = int(classic.body("table").id)
        geom_bodyid = np.asarray(mjx_model.geom_bodyid)
        paper_mask = geom_bodyid != table_id
        initial_centroid = self._centroid_from_data(data, paper_mask)

        # Run the simulation loop.  By default we use a plain Python loop with
        # eager mjx.step (JIT compilation of mjx.step on mesh-contact models is
        # prohibitively slow).  Set jit=True to JIT-compile the step function
        # (useful for long steady-state runs after warm-up, or on GPU).
        timestep = float(classic.opt.timestep)
        n_steps = round(duration / timestep)

        def step_fn(d, t_scalar):
            frac = jnp.minimum(t_scalar / fold_duration, 1.0)
            s = frac * frac * (3.0 - 2.0 * frac)
            d = d.replace(ctrl=s * targets)
            if vibrate:
                w = 2.0 * jnp.pi * freq_hz
                qpos = d.qpos.at[table_x_qadr].set(amp_x * jnp.sin(w * t_scalar))
                qvel = d.qvel.at[table_x_dadr].set(amp_x * w * jnp.cos(w * t_scalar))
                qpos = qpos.at[table_z_qadr].set(amp_z * jnp.sin(w * t_scalar))
                qvel = qvel.at[table_z_dadr].set(amp_z * w * jnp.cos(w * t_scalar))
                d = d.replace(qpos=qpos, qvel=qvel)
            return mjx.step(mjx_model, d)

        step = jax.jit(step_fn) if jit else step_fn
        record_every = max(1, int(record_every))
        times: list[float] = []
        qpos_hist: list[np.ndarray] = []
        qvel_hist: list[np.ndarray] = []
        centroid_hist: list[np.ndarray] = []

        def capture(d, t: float) -> None:
            times.append(float(t))
            qpos_hist.append(np.asarray(d.qpos).copy())
            qvel_hist.append(np.asarray(d.qvel).copy())
            centroid_hist.append(self._centroid_from_data(d, paper_mask))

        if record:
            capture(data, 0.0)
        final_data = data
        for i in range(n_steps):
            t_scalar = jnp.asarray(i * timestep, dtype=jnp.float32)
            final_data = step(final_data, t_scalar)
            if record and ((i + 1) % record_every == 0 or i + 1 == n_steps):
                capture(final_data, (i + 1) * timestep)

        final_qpos = np.asarray(final_data.qpos)
        if not np.isfinite(final_qpos).all():
            raise FloatingPointError("unstable simulation: non-finite joint coordinates produced.")

        final_centroid = self._centroid_from_data(final_data, paper_mask)

        extra: dict[str, Any] = {}
        if record:
            if not times:
                capture(final_data, n_steps * timestep)
            extra["trajectory"] = {
                "time": np.asarray(times, dtype=float),
                "qpos": np.asarray(qpos_hist),
                "qvel": np.asarray(qvel_hist),
                "centroid": np.asarray(centroid_hist, dtype=float),
                # Hinge qpos columns, so plots never mix in stretch/panel joints.
                "hinge_qposadr": np.asarray(hinge_adr, dtype=int),
            }

        return SimulationResult(
            model=bundle,
            data=final_data,
            pattern=pat,
            info=info,
            initial_centroid=initial_centroid,
            final_centroid=final_centroid,
            backend="jax",
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Batch population evaluation (sequential per-individual loop)
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
        """Evaluate a population of crease patterns sequentially.

        For MJX-JAX each individual gets its own model (different topology),
        so this loops over individuals and calls :meth:`run` eagerly.  There
        is no vmap or JIT batching in this path, and ``batch_size`` is
        ignored.  Scoring is always the +x centroid displacement.
        """
        scores = []
        for i, creases in enumerate(population):
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
                scores.append(float(delta[0]))  # forward along +x
            except Exception:  # noqa: BLE001 - one bad individual must not kill the batch
                scores.append(float("nan"))
        return np.asarray(scores, dtype=float)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _centroid_from_data(data, paper_mask: np.ndarray) -> np.ndarray:
        """Mean XY of paper geoms (those whose body is not the table)."""
        xpos = np.asarray(data.geom_xpos)
        pts = xpos[paper_mask, :2]
        return pts.mean(axis=0)


__all__ = ["JaxBackend"]
