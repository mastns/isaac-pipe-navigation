# SPDX-License-Identifier: MIT
"""Rover headlights: two spot-shaped sphere lights that follow the chassis every frame.

Like :mod:`pipe_rover.scene_setup` this module needs the running Isaac Sim app; import it after the
app has been launched.
"""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply
from pxr import Gf, UsdGeom, UsdLux

from . import log
from .scene_setup import get_stage

__all__ = ["HEADLIGHTS", "create_headlights", "HeadlightFollower"]

# Prim path and position in the chassis frame (x forward, y left, z up, metres) of each headlight.
HEADLIGHTS = (
    ("/World/HeadlightL", (0.12, 0.025, 0.08)),
    ("/World/HeadlightR", (0.12, -0.025, 0.08)),
)
_LIGHT_RADIUS = 0.02          # m, sphere-light radius
_CONE_SOFTNESS = 0.25
_FOCUS = 3.0


def create_headlights(intensity: float, cone_angle: float, color, robot_init_pos):
    """Create the two headlight prims at the rover's initial pose and return their prim paths.

    The lights are created at the top level of ``/World`` (not under the robot) so that the renderer
    picks them up before ``sim.reset()``; :class:`HeadlightFollower` then moves them with the chassis.
    """
    if intensity <= 0:
        log.info("Headlights disabled (intensity=0).")
        return []
    stage = get_stage()
    if stage is None:
        return []
    rx, ry, rz = robot_init_pos
    created = []
    for path, (ox, oy, oz) in HEADLIGHTS:
        if stage.GetPrimAtPath(path).IsValid():
            log.info(f"Headlight already exists at {path}, skipping.")
            created.append(path)
            continue
        prim = stage.DefinePrim(path, "SphereLight")
        if not prim.IsValid():
            log.error(f"Failed to create headlight prim at {path}")
            continue
        light = UsdLux.SphereLight(prim)
        light.GetIntensityAttr().Set(float(intensity))
        light.GetColorAttr().Set(Gf.Vec3f(*color))
        light.GetRadiusAttr().Set(_LIGHT_RADIUS)
        light.GetNormalizeAttr().Set(True)
        for attr_name, val in (("inputs:intensity", float(intensity)),
                               ("inputs:radius", _LIGHT_RADIUS),
                               ("inputs:normalize", True)):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                attr.Set(val)
        xformable = UsdGeom.Xformable(prim)
        xformable.AddTranslateOp().Set(Gf.Vec3d(rx + ox, ry + oy, rz + oz))
        xformable.AddRotateYOp().Set(-90.0)
        try:
            shaping = UsdLux.ShapingAPI.Apply(prim)
            shaping.GetShapingConeAngleAttr().Set(float(cone_angle))
            shaping.GetShapingConeSoftnessAttr().Set(_CONE_SOFTNESS)
            shaping.GetShapingFocusAttr().Set(_FOCUS)
        except Exception as err:
            log.warn(f"ShapingAPI failed on {path}: {err}")
        created.append(path)
        log.info(f"Headlight created at {path} (cone {cone_angle} deg)")
    return created


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


class HeadlightFollower:
    """Moves the headlight prims to the chassis pose every frame (USD ``xformOp:transform``).

    Parameters
    ----------
    paths:
        Prim paths returned by :func:`create_headlights`.
    device:
        Torch device of the simulation tensors.
    """

    def __init__(self, paths, device):
        offsets = dict(HEADLIGHTS)
        self._offsets = [torch.tensor(offsets[path], device=device) for path in paths]
        # the light shines along its local -Z; rotate -90 deg about Y so that it points along the chassis +X
        self._light_local_quat = torch.tensor(
            [math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0], device=device)
        stage = get_stage()
        self._ops = []
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()
                xform_op = xformable.AddTransformOp()
                xform_op.Set(Gf.Matrix4d(1.0))
                self._ops.append(xform_op)
                log.debug(f"Headlight {path}: switched to xformOp:transform")
            else:
                self._ops.append(None)
                log.warn(f"Headlight prim NOT found: {path}")
        log.info("Headlight per-frame follower ready.")

    def update(self, root_pos: torch.Tensor, root_quat: torch.Tensor) -> None:
        """Place the lights for the chassis at ``root_pos`` (3,) with orientation ``root_quat`` (w, x, y, z)."""
        world_quat = _quat_mul(root_quat, self._light_local_quat)
        qw, qx, qy, qz = world_quat.detach().cpu().tolist()
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        for op, offset in zip(self._ops, self._offsets, strict=True):
            if op is None:
                continue
            wp = (root_pos + quat_apply(root_quat, offset)).detach().cpu().tolist()
            op.Set(Gf.Matrix4d(
                1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0,
                2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0,
                2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0,
                wp[0], wp[1], wp[2], 1,
            ))
