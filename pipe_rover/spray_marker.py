# SPDX-License-Identifier: MIT
"""Spray marking: artificial visual features painted on the pipe wall.

A burst casts ``cluster_size`` PhysX rays from the rover centre inside a cone around the rover's forward
axis. Every ray that hits something other than the rover or the ground plane leaves a small emissive
orange sphere at the hit point. The ray sampling lives in :mod:`pipe_rover.spray`; this module adds the
PhysX queries and the USD prim authoring, so it needs the running Isaac Sim app (import it after
``AppLauncher``).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from isaaclab.utils.math import quat_apply
from pxr import Gf, Sdf, UsdGeom

from . import log
from .scene_setup import bind_material, define_omni_pbr, get_stage
from .spray import dot_center, jitter_radius, sample_cone_direction

__all__ = ["SprayMarker"]

_MAX_RAY_DISTANCE = 3.0      # m
_SKIP_MARGIN = 0.05          # m, step past an ignored hit (rover / ground) before re-casting
_MAX_SKIPS = 6


class SprayMarker:
    """Paints spray dots on whatever the cone of rays hits.

    Parameters
    ----------
    color:
        RGB of the paint (also used as emission colour).
    dot_radius:
        Mean dot radius in metres (each dot gets a +-30 % jitter).
    cluster_size:
        Rays (and at most dots) per burst.
    cone_half_angle_deg:
        Half-angle of the spray cone around the rover's +X axis.
    device:
        Torch device of the simulation tensors.
    ignore_prefixes:
        Prim-path prefixes whose hits are skipped (the rover itself and the ground plane).
    """

    def __init__(self, color, dot_radius: float, cluster_size: int, cone_half_angle_deg: float, device,
                 ignore_prefixes=("/World/envs/env_0/Robot", "/World/defaultGroundPlane"),
                 root_path: str = "/World/SprayDots", material_path: str = "/World/Materials/SprayPaint"):
        self.dot_radius = float(dot_radius)
        self.cluster_size = int(cluster_size)
        self.half_angle_rad = math.radians(cone_half_angle_deg)
        self.device = device
        self.ignore_prefixes = tuple(ignore_prefixes)
        self.dot_count = 0
        self.bursts = 0

        self._stage = get_stage()
        self._stage.DefinePrim(root_path, "Xform")
        self._root_path = root_path
        rgb = Gf.Vec3f(*color)
        self._material = define_omni_pbr(self._stage, material_path, {
            "diffuse_color_constant": (Sdf.ValueTypeNames.Color3f, rgb),
            "reflection_roughness_constant": (Sdf.ValueTypeNames.Float, 0.4),
            "metallic_constant": (Sdf.ValueTypeNames.Float, 0.0),
            "enable_emission": (Sdf.ValueTypeNames.Bool, True),
            "emissive_color": (Sdf.ValueTypeNames.Color3f, rgb),
            "emissive_intensity": (Sdf.ValueTypeNames.Float, 1000.0),
        })

        # PhysX scene queries are optional: without them the marker is created but cannot spray
        self._sqi = None
        try:
            import carb
            from omni.physx import get_physx_scene_query_interface
            self._carb = carb
            self._sqi = get_physx_scene_query_interface()
            log.info("Spray system: PhysX raycast available")
        except Exception as err:
            log.warn(f"Spray system: PhysX raycast unavailable ({err}); spraying is disabled")

    @property
    def available(self) -> bool:
        return self._sqi is not None

    # ------------------------------------------------------------------ #
    # PhysX ray cast
    # ------------------------------------------------------------------ #

    def _raycast(self, origin, direction, max_dist: float = _MAX_RAY_DISTANCE, depth: int = 0):
        """Closest hit that is not on an ignored prim: ``(position, normal)`` or ``(None, None)``.

        Hits on the rover or the ground plane are skipped by re-casting from just behind the hit.
        """
        try:
            hit = self._sqi.raycast_closest(
                self._carb.Float3(*origin), self._carb.Float3(*direction), max_dist, False)
        except Exception as err:
            log.debug(f"[SPRAY-RAY] raycast raised {type(err).__name__}: {err}")
            return None, None
        if not hit or not hit.get("hit", False):
            return None, None
        collision = hit.get("collision", "")
        dist = hit.get("distance", 0.0)
        if any(collision.startswith(prefix) for prefix in self.ignore_prefixes):
            skip = dist + _SKIP_MARGIN
            remaining = max_dist - skip
            if remaining > _SKIP_MARGIN and depth < _MAX_SKIPS:
                new_origin = tuple(origin[i] + direction[i] * skip for i in range(3))
                return self._raycast(new_origin, direction, remaining, depth + 1)
            return None, None
        p = hit.get("position")
        n = hit.get("normal")
        try:
            pos = (p.x, p.y, p.z)
        except AttributeError:
            pos = tuple(p)
        try:
            nrm = (n.x, n.y, n.z)
        except AttributeError:
            nrm = tuple(n)
        return pos, nrm

    # ------------------------------------------------------------------ #
    # Burst
    # ------------------------------------------------------------------ #

    def spray(self, root_pos: torch.Tensor, root_quat: torch.Tensor) -> int:
        """One spray burst from the chassis pose ``(root_pos, root_quat)``; returns the number of dots placed."""
        if not self.available:
            return 0
        origin = tuple(root_pos.detach().cpu().numpy().tolist())
        placed = missed = failed = 0
        for _ in range(self.cluster_size):
            local_dir = sample_cone_direction(self.half_angle_rad)
            world_dir = quat_apply(
                root_quat, torch.tensor(local_dir, dtype=torch.float32, device=self.device)
            ).detach().cpu().numpy()
            world_dir = world_dir / max(np.linalg.norm(world_dir), 1e-8)
            hit_pos, hit_normal = self._raycast(origin, tuple(world_dir.tolist()))
            if hit_pos is None:
                missed += 1
                continue
            radius = jitter_radius(self.dot_radius)
            pos = dot_center(hit_pos, hit_normal, radius)
            idx = self.dot_count
            try:
                dot = self._stage.DefinePrim(f"{self._root_path}/d{idx}", "Sphere")
                if dot.IsValid():
                    UsdGeom.Sphere(dot).GetRadiusAttr().Set(float(radius))
                    UsdGeom.Xformable(dot).AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))
                    bind_material(dot, self._material)
                    self.dot_count += 1
                    placed += 1
                else:
                    failed += 1
            except Exception as err:
                failed += 1
                if failed <= 3:
                    log.warn(f"Spray dot #{idx} failed: {type(err).__name__}: {err}")
        self.bursts += 1
        summary = (f"Spray burst {self.bursts}: {placed} placed, {missed} missed, {failed} failed; "
                   f"{self.dot_count} dots in total")
        if self.bursts == 1:
            log.info(summary)
        else:
            log.debug(summary)
        return placed
