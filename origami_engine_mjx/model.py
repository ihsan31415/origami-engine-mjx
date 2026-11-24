from __future__ import annotations

"""MJCF XML generation from a validated crease pattern.

This module is **backend-independent**: it produces a standard MJCF XML string
that can be consumed by classic MuJoCo (``MjModel.from_xml_string``), MJX-JAX
(``mjx.put_model``), or MJX-Warp.  No MuJoCo runtime is required here.
"""

from typing import Any

import numpy as np
from shapely.geometry import Point

from .constants import (
    DAMPING,
    EQ_SOLIMP,
    EQ_SOLREF,
    FORCE_MAX,
    KP,
    KV,
    MAX_FOLD,
    MU_PAPER,
    MU_TABLE,
    PAPER_MASS,
    STRETCH_DAMP,
    STRETCH_K,
    STRETCH_MAX,
    THICK,
    Z0,
    S,
)
from .geometry import _triangles


def _prism(tri, O, thick=None):
    """Triangular prism vertex list."""
    t = THICK if thick is None else thick
    v = []
    for x, y in np.asarray(tri)[:, :2]:
        v += [x - O[0], y - O[1], -t, x - O[0], y - O[1], t]
    return " ".join(f"{a:.6f}" for a in v)


def resolve_actuated(actuated, n_hinges: int) -> np.ndarray:
    """Turn the ``actuated`` argument into a boolean mask of length ``n_hinges``.

    ``None`` actuates every hinge (the historical behaviour), ``True`` /
    ``False`` select all / none, and any sequence is validated and copied in
    ``info["hinges"]`` order.  Invalid input raises ``ValueError`` so a typo in
    a mask never silently changes the physics.
    """
    if actuated is None:
        return np.ones(n_hinges, dtype=bool)
    if isinstance(actuated, (bool, np.bool_)):
        return np.full(n_hinges, bool(actuated), dtype=bool)
    mask = np.asarray(actuated, dtype=bool).reshape(-1)
    if mask.size != n_hinges:
        raise ValueError(f"actuated has {mask.size} entries, expected one per hinge ({n_hinges}).")
    return mask.copy()


def generate_xml(pat, creases, name="origami_free", coll_thick=None, actuated=None):
    """Build a MuJoCo XML string for a pattern.

    Parameters
    ----------
    pat : Pattern
        Validated pattern produced by :func:`origami_engine_mjx.geometry.build_pattern`.
    creases : list[Crease]
        Normalized creases, used to look up target fold angles.
    coll_thick : float | None
        ``None`` keeps the baseline single-geom behavior. A value enables thin
        collision mode with separate visual and collision geoms.
    actuated : bool | sequence[bool] | None
        Which hinges get a ``<position>`` actuator. ``None`` (default) keeps the
        historical behaviour: every hinge is actuated. ``True`` / ``False``
        select all / no hinges. A sequence must have one entry per hinge, in
        ``info["hinges"]`` order (i.e. crease-tree order). A non-actuated hinge
        still has a target angle and is still simulated — it is simply never
        driven and folds only through the forces its neighbours transmit.
    """
    faces, edges = pat.faces, pat.edges
    n = len(faces)
    density = PAPER_MASS / (S * S * 2 * THICK)
    mats = ["paper_A", "paper_B"]

    children = {i: [] for i in range(n)}
    for v, u in pat.parent.items():
        if u is not None:
            children[u].append(v)

    origin = {}
    for v in range(n):
        if v == pat.root:
            c = faces[v].centroid
            origin[v] = (c.x, c.y)
        else:
            origin[v] = edges[pat.tree_edge[v]]["p"]

    mesh_lines, body_lines, hinges = [], [], []

    def emit(i, depth):
        ind = "    " * (depth + 2)
        if i == pat.root:
            pos = (origin[i][0] - S / 2, origin[i][1] - S / 2, Z0)
        else:
            po = origin[pat.parent[i]]
            pos = (origin[i][0] - po[0], origin[i][1] - po[1], 0.0)
        body_lines.append(f'{ind}<body name="face_{i}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}">')

        if i == pat.root:
            body_lines.append(f'{ind}    <freejoint name="root"/>')
        else:
            e = edges[pat.tree_edge[i]]
            p, q = np.array(e["p"]), np.array(e["q"])
            a = (q - p) / np.linalg.norm(q - p)
            nl = np.array([-a[1], a[0]])
            left = faces[i].intersects(Point(*((p + q) / 2 + 1e-4 * nl)))
            sgn = 1.0 if left else -1.0
            ang = creases[e["crease"]].angle if e["crease"] >= 0 else 0.0
            h = len(hinges)
            hinges.append(
                {
                    "name": f"crease_{h}",
                    "face": i,
                    "edge": pat.tree_edge[i],
                    "target": float(np.clip(sgn * ang, -MAX_FOLD, MAX_FOLD)),
                }
            )
            for tag, ax_ in (("a", a), ("n", nl)):
                body_lines.append(
                    f'{ind}    <joint name="stretch_{tag}_{h}" type="slide" '
                    f'axis="{ax_[0]:.6f} {ax_[1]:.6f} 0" stiffness="{STRETCH_K}" '
                    f'damping="{STRETCH_DAMP}" armature="0.001" '
                    f'range="{-STRETCH_MAX} {STRETCH_MAX}" '
                    f'solreflimit="0.002 1" solimplimit="0.99 0.999 0.0001"/>'
                )
            body_lines.append(
                f'{ind}    <joint name="crease_{h}" type="hinge" pos="0 0 0" '
                f'axis="{a[0]:.6f} {a[1]:.6f} 0" range="{-np.pi:.4f} {np.pi:.4f}" '
                f'damping="{DAMPING}" armature="0.001"/>'
            )

        for j, tri in enumerate(_triangles(faces[i])):
            if coll_thick is None:
                mname = f"m{i}_{j}"
                mesh_lines.append(f'        <mesh name="{mname}" vertex="{_prism(tri, origin[i])}"/>')
                body_lines.append(
                    f'{ind}    <geom type="mesh" mesh="{mname}" '
                    f'material="{mats[pat.depth[i] % 2]}" density="{density:.3f}" '
                    f'contype="1" conaffinity="1" condim="3" '
                    f'friction="{MU_PAPER} 0.005 0.0001"/>'
                )
            else:
                vname = f"mv{i}_{j}"
                mesh_lines.append(f'        <mesh name="{vname}" vertex="{_prism(tri, origin[i])}"/>')
                body_lines.append(
                    f'{ind}    <geom type="mesh" mesh="{vname}" '
                    f'material="{mats[pat.depth[i] % 2]}" density="{density:.3f}" '
                    f'contype="0" conaffinity="0" condim="3"/>'
                )
                cname = f"mc{i}_{j}"
                mesh_lines.append(f'        <mesh name="{cname}" vertex="{_prism(tri, origin[i], coll_thick)}"/>')
                body_lines.append(
                    f'{ind}    <geom type="mesh" mesh="{cname}" rgba="0 0 0 0" '
                    f'density="0" contype="1" conaffinity="1" condim="3" '
                    f'friction="{MU_PAPER} 0.005 0.0001"/>'
                )

        for c in children[i]:
            emit(c, depth + 1)
        body_lines.append(f"{ind}</body>")

    emit(pat.root, 0)

    eq_lines = []
    for k in pat.loop_edges:
        e = edges[k]
        oi = origin[e["i"]]
        for pt in (e["p"], e["q"]):
            eq_lines.append(
                f'        <connect body1="face_{e["i"]}" body2="face_{e["j"]}" '
                f'anchor="{pt[0] - oi[0]:.6f} {pt[1] - oi[1]:.6f} 0" solref="{EQ_SOLREF}" solimp="{EQ_SOLIMP}"/>'
            )
    eq_block = ""
    if eq_lines:
        eq_block = "\n  <equality>\n" + "\n".join(eq_lines) + "\n  </equality>\n"

    act_mask = resolve_actuated(actuated, len(hinges))
    act_lines = [
        f'        <position name="fold_{h}" joint="{hi["name"]}" kp="{KP}" kv="{KV}" '
        f'ctrlrange="{-np.pi:.4f} {np.pi:.4f}" forcerange="-{FORCE_MAX} {FORCE_MAX}"/>'
        for h, hi in enumerate(hinges)
        if act_mask[h]
    ]
    act_block = ""
    if act_lines:
        act_block = "\n  <actuator>\n" + "\n".join(act_lines) + "\n  </actuator>\n"
    else:
        act_block = "\n  <!-- no actuated hinges: every hinge is passive -->\n"

    xml = f'''<mujoco model="{name}">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <default><geom solref="0.004 1" solimp="0.9 0.95 0.001"/></default>

  <asset>
    <material name="paper_A" rgba="0.85 0.90 1.0 1" specular="0.1"/>
    <material name="paper_B" rgba="1.0 0.85 0.85 1" specular="0.1"/>
    <texture name="checker" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.1 0.1 0.1" rgb2="0.8 0.8 0.8"/>
    <material name="floor" texture="checker" texrepeat="8 8" reflectance="0.1" specular="0.2"/>
  {chr(10).join(mesh_lines)}
  </asset>

  <worldbody>
    <body name="table" pos="0 0 -0.05">
      <joint name="table_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="table_z" type="slide" axis="0 0 1" damping="0"/>
      <geom name="ground" type="box" size="5 5 0.05" material="floor"
            contype="1" conaffinity="1" condim="3"
            friction="{MU_TABLE} 0.005 0.0001" mass="1000"/>
    </body>
  {chr(10).join(body_lines)}
  </worldbody>

{eq_block}{act_block}</mujoco>'''
    info = {
        "hinges": hinges,
        "targets": np.array([h["target"] for h in hinges]),
        "n_faces": n,
        "n_hinges": len(hinges),
        "n_loops": len(pat.loop_edges),
        # Per-hinge actuation state, in `hinges` order, plus the actuator
        # indices of the actuated hinges (element i of ctrl drives
        # hinges[actuator_hinges[i]]).
        "actuated": act_mask,
        "actuator_hinges": np.nonzero(act_mask)[0].astype(int),
    }
    return xml, info


def hinge_fold_error(qpos, info, actuated=None) -> np.ndarray:
    """Signed fold error ``achieved - target`` per hinge, in radians.

    Parameters
    ----------
    qpos : array-like
        Either a full MuJoCo ``qpos`` vector, or an array of achieved hinge
        angles in ``info["hinges"]`` order (one value per hinge), which is
        what the backends store in ``result.info["hinge_qpos"]``.
    info : dict
        Metadata returned by :func:`generate_xml` (needs ``targets`` and the
        hinge joint addresses).
    actuated : sequence[bool] | None
        Select which hinges to score.  ``None`` scores every hinge, so
        non-actuated hinges are scored too — their target is still the angle
        you *want* the paper to reach.

    Returns
    -------
    np.ndarray
        Shape ``(n_selected,)``; non-finite values are returned as-is.
    """
    targets = np.asarray(info["targets"], dtype=float).reshape(-1)
    qpos = np.asarray(qpos, dtype=float).reshape(-1)
    if qpos.size == targets.size:
        achieved = qpos  # already per-hinge angles
    else:
        adr = np.asarray(info["hinge_qposadr"], dtype=int).reshape(-1)
        achieved = qpos[adr] if adr.size else np.zeros(0, dtype=float)
    error = achieved - targets
    mask = resolve_actuated(actuated, error.size)
    return error[mask]


def fold_score(
    qpos,
    info,
    actuated=None,
    percent: bool = False,
    zero_tolerance_deg: float = 2.0,
) -> dict[str, Any]:
    """Score how closely a state folds the paper to its target hinge angles.

    This is the *fold* half of the fold fitness (see
    :mod:`origami_engine_mjx.fitness`); the *movement* half is the centroid
    displacement.  Two complementary numbers are returned:

    * ``score`` — fold progress in ``[0, 1]`` (higher is better).  It is
      ``1.0`` when every selected hinge is within ``zero_tolerance_deg`` of its
      target, and ``0.0`` when a hinge is still flat or has been folded past
      its target by more than its own target angle.  When ``percent`` is true
      it is a percentage ``0..100``.
    * ``mean_abs_error_deg`` — the plain mean absolute hinge error, handy for
      reports and logs.

    Each folding hinge moves linearly from ``0`` (still flat) to ``1``
    (within tolerance) as it approaches its target, so a hinge that overshoots
    is never rewarded.

    Returns a dict with ``score``, ``mean_abs_error_deg``, ``max_abs_error_deg``,
    ``rms_error_deg``, ``n_hinges`` (selected) and ``per_hinge_error_deg``.
    """
    error = hinge_fold_error(qpos, info, actuated=actuated)
    target = np.asarray(info["targets"], dtype=float).reshape(-1)
    mask = resolve_actuated(actuated, target.size)
    target = target[mask]
    abs_err = np.abs(error)
    tol = float(np.radians(zero_tolerance_deg))
    abs_target = np.abs(target)
    if abs_err.size:
        progress = np.empty_like(abs_err)
        # Hinges basically meant not to fold: only the tolerance matters.
        flat = abs_target <= tol
        progress[flat] = np.clip(1.0 - abs_err[flat] / tol, 0.0, 1.0)
        # Folding hinges: 0 while flat, 1 once within tolerance.
        fold = ~flat
        span = abs_target[fold] - tol
        progress[fold] = np.clip((abs_target[fold] - abs_err[fold]) / span, 0.0, 1.0)
        mean_err = float(np.mean(abs_err))
        max_err = float(np.max(abs_err))
        rms = float(np.sqrt(np.mean(abs_err**2)))
        value = float(np.mean(progress))
    else:  # a pattern without hinges trivially matches its (empty) target
        mean_err = max_err = rms = 0.0
        value = 1.0
    if percent:
        value *= 100.0
    return {
        "score": value,
        "mean_abs_error_deg": float(np.degrees(mean_err)),
        "max_abs_error_deg": float(np.degrees(max_err)),
        "rms_error_deg": float(np.degrees(rms)),
        "n_hinges": int(abs_err.size),
        "per_hinge_error_deg": np.degrees(error),
    }


__all__ = ["fold_score", "generate_xml", "hinge_fold_error", "resolve_actuated"]
