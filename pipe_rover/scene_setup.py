# SPDX-License-Identifier: MIT
"""Scene construction helpers: materials, lights, the pipe and the IMU markers.

This module needs the running Isaac Sim app (``pxr`` and ``omni`` are only importable after
``AppLauncher`` has started it), so import it *after* the app has been launched, as
``pipe_rover_sim.py`` does.
"""

from __future__ import annotations

import os

import isaacsim.core.utils.prims as prim_utils
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from . import log

__all__ = [
    "get_stage", "define_omni_pbr", "bind_material_to_meshes", "force_triangle_mesh_collision",
    "apply_corrosion_material", "apply_mud_ground_material", "setup_outdoor_sky",
    "disable_groundplane_light", "create_sun_light", "create_imu_markers",
    "bind_material", "require_pipe_asset", "spawn_pipe",
]


def get_stage():
    """The current USD stage (``None`` if there is none)."""
    return omni.usd.get_context().get_stage()


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #

def define_omni_pbr(stage, mat_path: str, inputs: dict):
    """Create an ``OmniPBR.mdl`` material at ``mat_path``.

    ``inputs`` maps a shader input name to ``(Sdf.ValueTypeNames.<type>, value)``; the inputs are
    created in dictionary order.
    """
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    for name, (value_type, value) in inputs.items():
        shader.CreateInput(name, value_type).Set(value)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mat


def bind_material(prim, material) -> None:
    """Bind ``material`` to ``prim`` (applying ``MaterialBindingAPI`` first, as USD requires)."""
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(material)


def bind_material_to_meshes(stage, root_path: str, material) -> int:
    """Bind ``material`` to every mesh below ``root_path`` (or to the prim itself if it has none).

    Returns the number of prims bound, or 0 if ``root_path`` does not exist.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    bound = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            bind_material(prim, material)
            bound += 1
    if bound == 0:
        bind_material(root, material)
        bound = 1
    return bound


def apply_corrosion_material(pipe_prim_path: str, texture_path: str, tex_scale: float) -> None:
    """Corroded-metal OmniPBR material, projected in object space, on the pipe meshes."""
    stage = get_stage()
    if stage is None:
        log.warn("No USD stage, cannot apply pipe texture.")
        return
    abs_tex = os.path.abspath(texture_path)
    mat = define_omni_pbr(stage, "/World/Materials/PipeCorrosion", {
        "diffuse_texture": (Sdf.ValueTypeNames.Asset, abs_tex),
        "project_uvw": (Sdf.ValueTypeNames.Bool, True),
        "world_or_object": (Sdf.ValueTypeNames.Bool, False),
        "texture_scale": (Sdf.ValueTypeNames.Float2, Gf.Vec2f(tex_scale, tex_scale)),
        "reflection_roughness_constant": (Sdf.ValueTypeNames.Float, 0.97),
        "metallic_constant": (Sdf.ValueTypeNames.Float, 0.05),
        "reflectionroughness_texture": (Sdf.ValueTypeNames.Asset, abs_tex),
        "reflection_roughness_texture_influence": (Sdf.ValueTypeNames.Float, 0.6),
    })
    bound = bind_material_to_meshes(stage, pipe_prim_path, mat)
    if bound:
        log.info(f"Corrosion material applied to {bound} mesh(es) under {pipe_prim_path}")
    else:
        log.warn(f"Pipe prim not found at {pipe_prim_path}, cannot bind material.")


def apply_mud_ground_material(ground_prim_path: str, texture_path: str, tex_scale: float) -> None:
    """Muddy-ground OmniPBR material, projected in world space, on the default ground plane."""
    stage = get_stage()
    if stage is None:
        return
    abs_tex = os.path.abspath(texture_path)
    mat = define_omni_pbr(stage, "/World/Materials/MudGround", {
        "diffuse_texture": (Sdf.ValueTypeNames.Asset, abs_tex),
        "project_uvw": (Sdf.ValueTypeNames.Bool, True),
        "world_or_object": (Sdf.ValueTypeNames.Bool, True),
        "texture_scale": (Sdf.ValueTypeNames.Float2, Gf.Vec2f(tex_scale, tex_scale)),
        "reflection_roughness_constant": (Sdf.ValueTypeNames.Float, 0.95),
        "metallic_constant": (Sdf.ValueTypeNames.Float, 0.0),
    })
    bound = bind_material_to_meshes(stage, ground_prim_path, mat)
    if bound:
        log.info(f"Mud ground material applied to {bound} mesh(es) under {ground_prim_path}")
    else:
        log.warn(f"Ground prim not found at {ground_prim_path}")


def setup_outdoor_sky(dome_light_path: str, sky_texture_path: str) -> None:
    """Use the overcast sky image as the dome-light texture."""
    stage = get_stage()
    if stage is None:
        return
    abs_tex = os.path.abspath(sky_texture_path)
    prim = stage.GetPrimAtPath(dome_light_path)
    if not (prim and prim.IsValid()):
        log.warn(f"Dome light prim not found at {dome_light_path}")
        return
    try:
        dome = UsdLux.DomeLight(prim)
        if dome:
            dome.GetTextureFileAttr().Set(abs_tex)
            log.info("Sky texture applied to dome light via UsdLux API")
    except Exception:
        tex_attr = prim.GetAttribute("inputs:texture:file")
        if not (tex_attr and tex_attr.IsValid()):
            tex_attr = prim.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset)
        tex_attr.Set(abs_tex)
        log.info("Sky texture applied to dome light via attribute fallback")


# --------------------------------------------------------------------------- #
# Lights
# --------------------------------------------------------------------------- #

def disable_groundplane_light() -> None:
    """Switch off the hidden sphere light that ships with the default ground plane."""
    try:
        stage = get_stage()
        if stage is None:
            return
        gp_light_path = "/World/defaultGroundPlane/SphereLight"
        prim = stage.GetPrimAtPath(gp_light_path)
        if prim and prim.IsValid():
            for attr_name in ("inputs:intensity", "intensity"):
                attr = prim.GetAttribute(attr_name)
                if attr and attr.IsValid():
                    attr.Set(0.0)
            img = UsdGeom.Imageable(prim)
            if img:
                img.MakeInvisible()
            log.info(f"Disabled hidden ground-plane light at {gp_light_path}")
        else:
            log.debug(f"No hidden ground-plane light found at {gp_light_path}")
    except Exception as err:
        log.warn(f"Could not disable ground-plane light: {err}")


def create_sun_light(intensity: float, color, elevation_deg: float, azimuth_deg: float,
                     path: str = "/World/SunLight") -> None:
    """Directional 'sun' light. Must exist before ``sim.reset()`` so that the renderer picks it up."""
    try:
        stage = get_stage()
        if stage is None:
            return
        if stage.GetPrimAtPath(path).IsValid():
            log.info(f"Sun light already exists at {path}")
            return
        sun_prim = stage.DefinePrim(path, "DistantLight")
        sun_light = UsdLux.DistantLight(sun_prim)
        sun_light.GetIntensityAttr().Set(intensity)
        sun_light.GetColorAttr().Set(Gf.Vec3f(*color))
        sun_light.GetAngleAttr().Set(0.53)
        xform = UsdGeom.Xformable(sun_prim)
        xform.AddRotateXOp().Set(elevation_deg)
        xform.AddRotateYOp().Set(azimuth_deg)
        log.info(f"Sun DistantLight created at {path}")
    except Exception as err:
        log.warn(f"Could not create sun light: {err}")


# --------------------------------------------------------------------------- #
# Pipe
# --------------------------------------------------------------------------- #

def force_triangle_mesh_collision(prim_path: str) -> None:
    """Make every mesh below ``prim_path`` collide as its exact triangle mesh (approximation ``none``)."""
    root = prim_utils.get_prim_at_path(prim_path)
    if not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(True)
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_col = UsdPhysics.MeshCollisionAPI(prim)
        approx_attr = None
        for attr_name in ("GetApproximationAttr", "GetMeshCollisionApproximationAttr"):
            fn = getattr(mesh_col, attr_name, None)
            if fn is not None:
                approx_attr = fn()
                break
        if approx_attr is not None:
            approx_attr.Set("none")
        else:
            log.warn(f"Could not find mesh collision approximation attr on {prim.GetPath()}")


def spawn_pipe(prim_path: str, usd_path: str, position, scale: float, orientation_wxyz) -> None:
    """Reference the pipe USD at ``prim_path`` (translation, uniform scale, wxyz orientation)."""
    prim_utils.create_prim(
        prim_path=prim_path,
        prim_type="Xform",
        usd_path=os.path.abspath(usd_path),
        translation=tuple(float(v) for v in position),
        orientation=orientation_wxyz,
        scale=(float(scale),) * 3,
    )
    log.info(f"Pipe spawned at {prim_path}")


def require_pipe_asset(usd_path: str) -> str:
    """Return ``usd_path`` if the pipe USD exists, otherwise stop with a clear error."""
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(
            f"Pipe USD not found: {usd_path}. The model ships as assets/pipe/Real_sim.usd; "
            "pass --pipe_usd to use another file.")
    return usd_path


# --------------------------------------------------------------------------- #
# Rover decoration
# --------------------------------------------------------------------------- #

def create_imu_markers(robot_prim_path: str, imu_positions, imu_angles_deg, marker_radius: float,
                       z_above: float) -> None:
    """Small black spheres above the chassis that show where the IMUs sit on the ring (visual only)."""
    stage = get_stage()
    if stage is None:
        return
    chassis_path = f"{robot_prim_path}/chassis"
    mat = define_omni_pbr(stage, "/World/Materials/ImuMarkerBlack", {
        "diffuse_color_constant": (Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.02, 0.02, 0.02)),
        "reflection_roughness_constant": (Sdf.ValueTypeNames.Float, 0.9),
        "metallic_constant": (Sdf.ValueTypeNames.Float, 0.0),
    })
    for i, (ix, iy, iz) in enumerate(imu_positions):
        dot_path = f"{chassis_path}/ImuMarker{i}"
        prim = stage.DefinePrim(dot_path, "Sphere")
        if not prim.IsValid():
            log.warn(f"Could not create IMU marker at {dot_path}")
            continue
        UsdGeom.Sphere(prim).GetRadiusAttr().Set(float(marker_radius))
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(ix, iy, iz + z_above))
        bind_material(prim, mat)
        log.debug(f"IMU marker {i} at chassis-local ({ix:.4f}, {iy:.4f}, {iz + z_above:.4f}), "
                  f"{imu_angles_deg[i]:.0f} deg")
    log.info(f"{len(imu_positions)} IMU markers created above chassis")
