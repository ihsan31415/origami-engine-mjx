from __future__ import annotations

"""Per-frame simulation trajectories: capture, replay and plotting.

A :class:`Trajectory` wraps the plain dict that the backends store in
``SimulationResult.extra["trajectory"]`` (``time``, ``qpos``, ``qvel``,
``centroid``) with a few conveniences:

* :meth:`Trajectory.from_result` pulls it out of a result (and raises a clear
  error when the run was not recorded);
* :meth:`Trajectory.center_of_mass` gives the paper centre of mass per frame
  (hinge angles are not a meaningful movement measure);
* :meth:`Trajectory.rate` / :meth:`Trajectory.velocity` give first differences;
* :meth:`Trajectory.subsample` thins frames to a target count.

Nothing here imports MuJoCo or JAX, so trajectories can be inspected and
plotted in environments without a simulator or GL context.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["Trajectory"]


class Trajectory:
    """Recorded frames of one simulation (all arrays are host NumPy).

    Attributes
    ----------
    time : np.ndarray
        Frame times in seconds, shape ``(T,)``.
    qpos, qvel : np.ndarray
        Full joint state per frame, shape ``(T, nq)`` / ``(T, nv)``.
    centroid : np.ndarray
        Mean XY of the paper geoms per frame, shape ``(T, 2)``.  This is the
        quantity the engine scores (``centroid[-1] - centroid[0]``).
    """

    def __init__(
        self,
        time,
        qpos,
        qvel,
        centroid,
        *,
        table_offset: tuple[float, float] | None = None,
        hinge_qposadr=None,
    ) -> None:
        self.time: np.ndarray = np.asarray(time, dtype=float).reshape(-1)
        self.qpos: np.ndarray = np.asarray(qpos)
        self.qvel: np.ndarray = np.asarray(qvel)
        self.centroid: np.ndarray = np.asarray(centroid, dtype=float).reshape(-1, 2)
        n = self.time.size
        if self.qpos.ndim != 2 or self.qvel.ndim != 2 or self.centroid.shape[0] != n:
            raise ValueError("trajectory arrays must be (T, nq), (T, nv) and (T, 2)")
        #: Index of the ``table_x`` / ``table_z`` hinge positions in ``qpos``.
        self.table_offset = table_offset
        #: ``qpos`` columns of the fold hinges (``None`` if not recorded).
        self.hinge_qposadr: np.ndarray | None = (
            None if hinge_qposadr is None else np.asarray(hinge_qposadr, dtype=int).reshape(-1)
        )

    # -- construction ---------------------------------------------------

    @classmethod
    def from_result(cls, result: Any) -> Trajectory:
        """Build a trajectory from a recorded ``SimulationResult``.

        Raises
        ------
        ValueError
            If the simulation was not run with ``record=True``.
        """
        extra = getattr(result, "extra", None) or {}
        raw = extra.get("trajectory")
        if raw is None:
            raise ValueError("no trajectory recorded; run with record=True (engine.run or evaluate)")
        table_offset = None
        info = getattr(result, "info", None) or {}
        qadr = info.get("table_qposadr")
        if qadr is not None:
            table_offset = (int(qadr[0]), int(qadr[1]))
        return cls(
            raw["time"],
            raw["qpos"],
            raw["qvel"],
            raw["centroid"],
            table_offset=table_offset,
            hinge_qposadr=raw.get("hinge_qposadr"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Trajectory:
        """Build a trajectory from a plain ``{"time", "qpos", "qvel", "centroid"}`` dict."""
        return cls(data["time"], data["qpos"], data["qvel"], data["centroid"], hinge_qposadr=data.get("hinge_qposadr"))

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the plain dict form stored in ``result.extra["trajectory"]``."""
        out = {"time": self.time, "qpos": self.qpos, "qvel": self.qvel, "centroid": self.centroid}
        if self.hinge_qposadr is not None:
            out["hinge_qposadr"] = self.hinge_qposadr
        return out

    # -- convenience ----------------------------------------------------

    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def duration(self) -> float:
        """Simulated span in seconds (last frame time - first frame time)."""
        return float(self.time[-1] - self.time[0]) if self.time.size > 1 else 0.0

    @property
    def displacement(self) -> np.ndarray:
        """Net XY displacement of the centroid: ``centroid[-1] - centroid[0]``."""
        return self.centroid[-1] - self.centroid[0]

    def center_of_mass(self, *, relative: bool = False) -> np.ndarray:
        """Paper centre per frame, minus the moving table offset if known.

        ``relative=True`` also subtracts the first frame so the trace starts at
        the origin (useful for movement plots).
        """
        com = self.centroid.copy()
        if self.table_offset is not None:
            tx = self.qpos[:, int(self.table_offset[0])]
            tz = self.qpos[:, int(self.table_offset[1])]
            com[:, 0] -= tx
            com[:, 1] -= tz
        if relative:
            com -= com[0]
        return com

    def rate(self, values: np.ndarray | None = None) -> np.ndarray:
        """Per-second rate of change of ``values`` (default: centroid)."""
        vals = self.centroid if values is None else np.asarray(values, dtype=float)
        return self.velocity(vals)

    def velocity(self, values: np.ndarray | None = None) -> np.ndarray:
        """``d(values)/dt`` using centred finite differences."""
        vals = self.centroid if values is None else np.asarray(values, dtype=float)
        if self.time.size < 2:
            return np.zeros_like(vals)
        grad: np.ndarray = np.gradient(vals, self.time, axis=0)
        return grad

    def hinge_angles(self) -> np.ndarray:
        """Fold-hinge ``qpos`` columns (not the table, stretch or panel joints)."""
        if self.hinge_qposadr is not None:
            return self.qpos[:, self.hinge_qposadr]
        if self.table_offset is None:
            return self.qpos
        drop = set(self.table_offset)
        keep = [i for i in range(self.qpos.shape[1]) if i not in drop]
        return self.qpos[:, keep]

    def subsample(self, n_frames: int) -> np.ndarray:
        """Return ``n_frames`` evenly spaced frame indices (first/last included)."""
        n_frames = max(1, int(n_frames))
        if n_frames >= len(self):
            return np.arange(len(self))
        return np.unique(np.linspace(0, len(self) - 1, n_frames).astype(int))

    def subset(self, indices) -> Trajectory:
        """Return a new trajectory containing only ``indices``."""
        idx = np.asarray(indices, dtype=int)
        return Trajectory(
            self.time[idx],
            self.qpos[idx],
            self.qvel[idx],
            self.centroid[idx],
            table_offset=self.table_offset,
            hinge_qposadr=self.hinge_qposadr,
        )
