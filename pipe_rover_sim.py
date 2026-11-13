# SPDX-License-Identifier: MIT
"""Pipe-inspection rover simulation for Isaac Sim 5.1 / Isaac Lab 2.3.2.

A Jetbot-style differential-drive rover drives through a pipe (plain grey wall, optionally corroded) lying in an
abandoned field. It carries a front and a rear stereo camera pair, three IMUs on a small ring, two headlights and a
spray marker that paints small orange dots on the pipe wall (artificial visual features). The stereo images (mono8),
the noisy IMU streams and the ground-truth pose are recorded to a ROS 2 bag.

Run it with Isaac Sim's Python, with ROS 2 Humble sourced so that ``rosbag2_py`` can be imported::

    python3 pipe_rover_sim.py                 # GUI + keyboard teleoperation
    python3 scripts/headless_demo.py          # no window, no keyboard: scripted drive, records and checks a bag

Keyboard (the GUI window needs focus): Up/Down = drive, Z / X = turn, B = start/stop bag recording, P = spray now
(automatic spraying every ``--auto_spray_interval`` seconds), R = put the rover back at its start pose, L = zero the
drive command, Esc = quit.

The tunable defaults are listed in the DEFAULTS section below; most of them can be overridden on the command line
(``python3 pipe_rover_sim.py --help``).
"""

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from pipe_rover import __version__, log
from pipe_rover.imu_noise import ImuNoiseModel
from pipe_rover.recorder import Ros2StereoImuRecorder
from pipe_rover.textures import generate_corrosion_texture, generate_mud_texture, generate_sky_texture

# =========================================================================== #
#                             DEFAULTS / CONFIG                               #
# =========================================================================== #
# Units: metres, seconds, radians (unless a name says _DEG), pixels.

REPO_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = REPO_ROOT / "assets"
TEXTURE_DIR = ASSETS_DIR / "textures"      # procedural textures are generated here on first use

# ---- Simulation ----
PHYSICS_DT          = 0.005     # s    physics step (200 Hz); IMUs and ground truth are recorded every step
HUD_UPDATE_PERIOD   = 0.1       # s    refresh period of the on-screen stereo view
IMAGE_RECORD_PERIOD = 0.033     # s    *minimum* period between recorded image sets; with a 5 ms step
                                #      images are written every 7th step = 35 ms = 28.6 Hz

# ---- Robot geometry ----
ROBOT_INIT_POS      = (0.3, 0.0, 0.15)  # m  initial chassis position (inside the pipe mouth)
WHEEL_RADIUS        = 0.0325    # m
WHEEL_BASE          = 0.112     # m    distance between the wheels
MAX_WHEEL_SPEED     = 20.0      # rad/s

# ---- Teleop ----
TELEOP_LINEAR_SPEED  = 0.25     # m/s  forward speed at full key press
TELEOP_ANGULAR_SPEED = 2.0      # rad/s turn rate at full key press

# ---- IMU placement: three IMUs on a ring around the chassis origin, 120 deg apart ----
IMU_NAMES           = ("imu_0", "imu_1", "imu_2")
IMU_RADIUS          = 0.05      # m
IMU_ANGLES_DEG      = (0, 120, 240)
IMU_MARKER_RADIUS   = 0.008     # m    black spheres drawn above the chassis (visual only)
IMU_MARKER_Z_ABOVE  = 0.12      # m

# ---- IMU noise (see pipe_rover/imu_noise.py): per-sample values at the 200 Hz physics rate ----
IMU_GYRO_NOISE_STD      = 0.005     # rad/s       white noise std per sample
IMU_ACCEL_NOISE_STD     = 0.05      # m/s^2       white noise std per sample
IMU_GYRO_BIAS_STD       = 0.001     # rad/s       bias random-walk increment std per sample
IMU_ACCEL_BIAS_STD      = 0.005     # m/s^2       bias random-walk increment std per sample

# ---- Front stereo cameras ----
STEREO_WIDTH        = 320       # px
STEREO_HEIGHT       = 240       # px
STEREO_BASELINE     = 0.06      # m
STEREO_MOUNT_X      = 0.12      # m    forward of the chassis origin
STEREO_MOUNT_Z      = 0.10      # m    above the chassis origin

# ---- Rear stereo cameras (same baseline and image size, looking backwards) ----
REAR_STEREO_MOUNT_X = -0.10     # m
REAR_STEREO_MOUNT_Z = 0.10      # m

# ---- Follow camera (GUI viewport) ----
FOLLOW_DISTANCE     = 0.35      # m    behind the rover
FOLLOW_HEIGHT       = 0.22      # m    above the rover

# ---- Pipe ----
PIPE_SCALE          = 1.5       # uniform scale applied to the pipe model
PIPE_POS            = (0.1, 0.0, 0.375)  # m
PIPE_USD            = str(ASSETS_DIR / "pipe" / "Real_sim.usd")   # Blender export, see assets/README.md
PIPE_TEXTURE_ON     = False     # procedural corrosion texture on the pipe wall
PIPE_TEXTURE_SEED   = 42
PIPE_TEXTURE_SIZE   = 512       # px
PIPE_TEX_SCALE      = 5.0

# ---- Headlights ----
HEADLIGHT_INTENSITY = 5000.0
HEADLIGHT_CONE_DEG  = 35.0      # deg  half-angle (UsdLux shaping:cone:angle is measured from the axis)
HEADLIGHT_COLOR     = (1.0, 0.97, 0.92)

# ---- Sun / sky / ground ----
DOME_LIGHT_INTENSITY = 250.0
SUN_INTENSITY        = 200.0
SUN_COLOR            = (0.9, 0.88, 0.82)
SUN_ELEVATION_DEG    = -35.0    # deg  rotateX of the DistantLight prim (the sun ends up ~50 deg above the horizon)
SUN_AZIMUTH_DEG      = 20.0     # deg  rotateY of the DistantLight prim
GROUND_TEX_SCALE     = 12.0
GROUND_TEXTURE_SEED  = 99       # seeds and sizes of the generated ground and sky images
GROUND_TEXTURE_SIZE  = 1024     # px
SKY_TEXTURE_SEED     = 88
SKY_TEXTURE_SIZE     = (2048, 1024)     # px (width, height)

# ---- Spray marking ----
SPRAY_DOT_RADIUS     = 0.006    # m    mean dot radius (+-30 % jitter)
SPRAY_CLUSTER_SIZE   = 120      # rays per burst
SPRAY_CONE_HALF_ANGLE = 70.0    # deg  around the rover's forward axis
SPRAY_COLOR          = (1.0, 0.35, 0.0)
AUTO_SPRAY_INTERVAL  = 0.5      # s    automatic burst period (0 = disabled)

# ---- ROS 2 recording ----
ROS2_BAG_DIR         = str(REPO_ROOT / "rosbags")
ROS2_BAG_PREFIX      = "stereo_imus"

# ---- Derived: IMU positions on the circle (chassis frame) ----
IMU_POSITIONS = tuple(
    (IMU_RADIUS * math.cos(math.radians(a)),
     IMU_RADIUS * math.sin(math.radians(a)),
     0.0)
    for a in IMU_ANGLES_DEG
)

# =========================================================================== #
#                            CLI / APP LAUNCHER                               #
# =========================================================================== #

from isaaclab.app import AppLauncher  # noqa: E402  (Isaac Sim modules can only be imported after the app starts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jetbot rover with front+rear stereo, 3 IMUs, headlights and spray marking in a pipe. "
                    "Isaac Lab / AppLauncher options (--headless, --device, --verbose, ...) are accepted too.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = parser.add_argument_group("run")
    g.add_argument("--num_envs", type=int, default=1,
                   help="must be 1 (the pipe, sensors and recorder exist for env 0 only)")
    g.add_argument("--control_mode", choices=["teleop", "auto"], default="teleop",
                   help="teleop: keyboard drive (needs the GUI window); auto: no controller, the rover stays still "
                        "and recording cannot be started")
    g.add_argument("--seed", type=int, default=None,
                   help="seed NumPy and PyTorch so that IMU noise and spray patterns are reproducible")
    g.add_argument("--follow_camera", dest="follow_camera", action="store_true", default=True,
                   help="GUI viewport follows the rover")
    g.add_argument("--no_follow_camera", dest="follow_camera", action="store_false")
    g.add_argument("--follow_distance", type=float, default=FOLLOW_DISTANCE,
                   help="follow camera: metres behind the rover")
    g.add_argument("--follow_height", type=float, default=FOLLOW_HEIGHT, help="follow camera: metres above the rover")
    g.add_argument("--hud", dest="hud", action="store_true", default=True,
                   help="show the 'Stereo View' window with the four camera images (GUI only)")
    g.add_argument("--no_hud", dest="hud", action="store_false")

    g = parser.add_argument_group("rover")
    g.add_argument("--teleop_linear_speed", type=float, default=TELEOP_LINEAR_SPEED, help="m/s at full key press")
    g.add_argument("--teleop_angular_speed", type=float, default=TELEOP_ANGULAR_SPEED, help="rad/s at full key press")
    g.add_argument("--wheel_radius", type=float, default=WHEEL_RADIUS, help="m, used to convert (v, w) to wheel speeds")
    g.add_argument("--wheel_base", type=float, default=WHEEL_BASE, help="m, distance between the wheels")
    g.add_argument("--max_wheel_speed", type=float, default=MAX_WHEEL_SPEED, help="rad/s, wheel command clamp")

    g = parser.add_argument_group("cameras")
    g.add_argument("--stereo_width", type=int, default=STEREO_WIDTH, help="px, all four cameras")
    g.add_argument("--stereo_height", type=int, default=STEREO_HEIGHT, help="px, all four cameras")
    g.add_argument("--stereo_baseline", type=float, default=STEREO_BASELINE, help="m, both pairs")
    g.add_argument("--stereo_mount_x", type=float, default=STEREO_MOUNT_X,
                   help="m, front pair, forward of the chassis origin")
    g.add_argument("--stereo_mount_z", type=float, default=STEREO_MOUNT_Z,
                   help="m, front pair, above the chassis origin")
    g.add_argument("--rear_stereo_mount_x", type=float, default=REAR_STEREO_MOUNT_X,
                   help="m, rear pair (negative = behind)")
    g.add_argument("--rear_stereo_mount_z", type=float, default=REAR_STEREO_MOUNT_Z,
                   help="m, rear pair, above the chassis origin")

    g = parser.add_argument_group("pipe")
    g.add_argument("--pipe_scale", type=float, default=PIPE_SCALE, help="uniform scale of the pipe model")
    g.add_argument("--pipe_x", type=float, default=PIPE_POS[0], help="m, world position of the pipe model")
    g.add_argument("--pipe_y", type=float, default=PIPE_POS[1], help="m")
    g.add_argument("--pipe_z", type=float, default=PIPE_POS[2], help="m")
    g.add_argument("--pipe_usd", type=str, default=PIPE_USD, help="pipe model (USD)")
    g.add_argument("--pipe_texture", dest="pipe_texture", action="store_true", default=PIPE_TEXTURE_ON,
                   help="paint the pipe wall with a procedural corrosion texture (off: plain grey wall)")
    g.add_argument("--no_pipe_texture", dest="pipe_texture", action="store_false")
    g.add_argument("--pipe_texture_seed", type=int, default=PIPE_TEXTURE_SEED, help="corrosion texture seed")
    g.add_argument("--pipe_texture_size", type=int, default=PIPE_TEXTURE_SIZE, help="px, corrosion texture side length")

    g = parser.add_argument_group("headlights")
    g.add_argument("--headlight_intensity", type=float, default=HEADLIGHT_INTENSITY,
                   help="0 switches the headlights off")
    g.add_argument("--headlight_cone_angle", type=float, default=HEADLIGHT_CONE_DEG, help="deg, cone half-angle")
    g.add_argument("--headlight_color_r", type=float, default=HEADLIGHT_COLOR[0])
    g.add_argument("--headlight_color_g", type=float, default=HEADLIGHT_COLOR[1])
    g.add_argument("--headlight_color_b", type=float, default=HEADLIGHT_COLOR[2])

    g = parser.add_argument_group("spray marking")
    g.add_argument("--spray_dot_radius", type=float, default=SPRAY_DOT_RADIUS,
                   help="m, mean dot radius (+-30 %% jitter)")
    g.add_argument("--spray_cluster_size", type=int, default=SPRAY_CLUSTER_SIZE, help="rays per burst")
    g.add_argument("--spray_cone_half_angle", type=float, default=SPRAY_CONE_HALF_ANGLE,
                   help="deg, half-angle of the cone around the rover's forward axis")
    g.add_argument("--spray_color_r", type=float, default=SPRAY_COLOR[0])
    g.add_argument("--spray_color_g", type=float, default=SPRAY_COLOR[1])
    g.add_argument("--spray_color_b", type=float, default=SPRAY_COLOR[2])
    g.add_argument("--auto_spray_interval", type=float, default=AUTO_SPRAY_INTERVAL,
                   help="s of simulation time between automatic bursts (0 = manual only, key P)")

    g = parser.add_argument_group("recording")
    g.add_argument("--ros2_record", dest="ros2_record", action="store_true", default=True,
                   help="enable the in-process ROS 2 bag recorder (start/stop with key B)")
    g.add_argument("--no_ros2_record", dest="ros2_record", action="store_false")
    g.add_argument("--ros2_bag_dir", type=str, default=ROS2_BAG_DIR, help="directory that receives the bags")
    g.add_argument("--ros2_bag_prefix", type=str, default=ROS2_BAG_PREFIX, help="bag folder name prefix")

    AppLauncher.add_app_launcher_args(parser)
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.num_envs != 1:
        parser.error("--num_envs must be 1: the pipe, IMU markers, spray, HUD and recorder exist for env 0 only")
    positive = ("wheel_radius", "wheel_base", "max_wheel_speed", "stereo_width", "stereo_height", "stereo_baseline",
                "pipe_scale", "pipe_texture_size", "spray_dot_radius")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be greater than 0")
    non_negative = ("teleop_linear_speed", "teleop_angular_speed", "headlight_intensity", "spray_cluster_size",
                    "auto_spray_interval")
    for name in non_negative:
        if getattr(args, name) < 0:
            parser.error(f"--{name} must not be negative")
    if not 0.0 < args.spray_cone_half_angle <= 180.0:
        parser.error("--spray_cone_half_angle must be in (0, 180] degrees")
    if not 0.0 < args.headlight_cone_angle <= 90.0:
        parser.error("--headlight_cone_angle must be in (0, 90] degrees")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")


_parser = _build_parser()
args_cli = _parser.parse_args()
_validate(args_cli, _parser)
log.set_verbose(getattr(args_cli, "verbose", False))   # --verbose is an Isaac Lab (AppLauncher) flag
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True      # the stereo cameras need the RTX sensor pipeline

log.info(f"pipe_rover_sim v{__version__} (Isaac Sim 5.1 / Isaac Lab 2.3.2)")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# Imports that need the running app
# --------------------------------------------------------------------------- #

from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg, ImuCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from pipe_rover import scene_setup  # noqa: E402
from pipe_rover.headlights import HeadlightFollower, create_headlights  # noqa: E402
from pipe_rover.spray_marker import SprayMarker  # noqa: E402

# --------------------------------------------------------------------------- #
# Scene configuration
# --------------------------------------------------------------------------- #

IMU_ROT = (1.0, 0.0, 0.0, 0.0)                      # IMU axes = chassis axes
PIPE_ROT_Z_NEG90 = (0.70710678, 0.0, 0.0, -0.70710678)   # wxyz: -90 deg about Z, pipe axis along world +X

JETBOT_CFG = ArticulationCfg(
    prim_path="/World/Jetbot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=ROBOT_INIT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=None, velocity_limit=None, stiffness=None, damping=None,
        ),
    },
)


def _stereo_camera(name: str, x: float, y: float, z: float, rot_wxyz) -> CameraCfg:
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/chassis/{name}",
        offset=CameraCfg.OffsetCfg(pos=(x, y, z), rot=rot_wxyz, convention="world"),
        spawn=sim_utils.PinholeCameraCfg(),
        width=args_cli.stereo_width, height=args_cli.stereo_height, data_types=["rgb"],
    )


@configclass
class RoverSceneCfg(InteractiveSceneCfg):
    """Ground, sky, rover, three IMUs and the front/rear stereo pairs (rear cameras are yawed 180 deg)."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=DOME_LIGHT_INTENSITY, color=(1.0, 1.0, 1.0)),
    )
    robot = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    imu_0 = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/chassis",
                   offset=ImuCfg.OffsetCfg(pos=IMU_POSITIONS[0], rot=IMU_ROT))
    imu_1 = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/chassis",
                   offset=ImuCfg.OffsetCfg(pos=IMU_POSITIONS[1], rot=IMU_ROT))
    imu_2 = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/chassis",
                   offset=ImuCfg.OffsetCfg(pos=IMU_POSITIONS[2], rot=IMU_ROT))
    stereo_left = _stereo_camera("StereoLeft", args_cli.stereo_mount_x, args_cli.stereo_baseline * 0.5,
                                 args_cli.stereo_mount_z, (1.0, 0.0, 0.0, 0.0))
    stereo_right = _stereo_camera("StereoRight", args_cli.stereo_mount_x, -args_cli.stereo_baseline * 0.5,
                                  args_cli.stereo_mount_z, (1.0, 0.0, 0.0, 0.0))
    rear_stereo_left = _stereo_camera("RearStereoLeft", args_cli.rear_stereo_mount_x,
                                      args_cli.stereo_baseline * 0.5, args_cli.rear_stereo_mount_z,
                                      (0.0, 0.0, 0.0, 1.0))
    rear_stereo_right = _stereo_camera("RearStereoRight", args_cli.rear_stereo_mount_x,
                                       -args_cli.stereo_baseline * 0.5, args_cli.rear_stereo_mount_z,
                                       (0.0, 0.0, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def _camera_rgb(camera) -> np.ndarray:
    return camera.data.output["rgb"][0].cpu().numpy()


def reset_rover(robot, scene) -> None:
    """Put the rover back at its initial pose with zero velocity (what the R key advertises).

    ``scene.reset()`` alone only clears actuator and sensor buffers; it does not move the robot.
    """
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    scene.reset()


def _portable(value):
    """Paths below the repository are stored as ``<repo>/...`` so that the sidecar carries no machine paths."""
    root = str(REPO_ROOT)
    if isinstance(value, str) and value.startswith(root):
        return "<repo>" + value[len(root):]
    return value


def _run_info() -> dict:
    """Provenance written next to every recorded bag (sim_config.json)."""
    from importlib.metadata import PackageNotFoundError, version

    def ver(package):
        try:
            return version(package)
        except PackageNotFoundError:
            return None

    pinhole = sim_utils.PinholeCameraCfg()
    fx = args_cli.stereo_width * pinhole.focal_length / pinhole.horizontal_aperture
    return {
        "script": "pipe_rover_sim.py",
        "version": __version__,
        "packages": {"isaacsim": ver("isaacsim"), "isaaclab": ver("isaaclab"), "torch": torch.__version__,
                     "numpy": np.__version__},
        "timing": {"physics_dt_s": PHYSICS_DT, "image_record_period_s": IMAGE_RECORD_PERIOD,
                   "image_every_n_steps": math.ceil(IMAGE_RECORD_PERIOD / PHYSICS_DT - 1e-9)},
        "imu": {"names": list(IMU_NAMES), "positions_chassis_m": [list(p) for p in IMU_POSITIONS],
                "noise_per_sample": {"gyro_std": IMU_GYRO_NOISE_STD, "accel_std": IMU_ACCEL_NOISE_STD,
                                     "gyro_bias_step_std": IMU_GYRO_BIAS_STD,
                                     "accel_bias_step_std": IMU_ACCEL_BIAS_STD}},
        "cameras": {"size_px": [args_cli.stereo_width, args_cli.stereo_height],
                    "baseline_m": args_cli.stereo_baseline,
                    "front_mount_xz_m": [args_cli.stereo_mount_x, args_cli.stereo_mount_z],
                    "rear_mount_xz_m": [args_cli.rear_stereo_mount_x, args_cli.rear_stereo_mount_z],
                    "pinhole": {"focal_length": pinhole.focal_length,
                                "horizontal_aperture": pinhole.horizontal_aperture,
                                "fx_fy_px": fx, "cx_cy_px": [args_cli.stereo_width / 2, args_cli.stereo_height / 2]}},
        "args": {k: _portable(v) for k, v in vars(args_cli).items()
                 if isinstance(v, (str, int, float, bool, type(None)))},
    }


def run_simulator(sim, scene, dof_idx, teleop, hud, recorder, headlight_paths=None, pipe_pos=None):
    sim_dt = sim.get_physics_dt()
    dt_ns = round(sim_dt * 1e9)     # integer nanoseconds per step: stamps are exact multiples of the step
    step = 0                        # simulation time = step * dt, starting at 0 when the loop starts

    scene.reset()

    spray_requested = False
    reset_requested = False

    def request_spray():
        nonlocal spray_requested
        spray_requested = True

    def request_reset():
        nonlocal reset_requested
        reset_requested = True

    if teleop:
        teleop.add_callback("ESCAPE", lambda: setattr(teleop, "_exit", True))
        teleop.add_callback("R", request_reset)
        teleop.add_callback("B", recorder.request_toggle)
        teleop.add_callback("P", request_spray)
        teleop.reset()
        log.info("Teleop active. Up/Down=drive  Z/X=turn  B=record  P=spray (manual)  R=reset rover  "
                 f"Esc=quit  (auto-spray every {args_cli.auto_spray_interval:.1f}s)")
    else:
        log.warn("--control_mode auto: no controller is implemented, the rover stays still and "
                 "recording cannot be started (B needs the keyboard).")

    robot = scene["robot"]
    l_cam, r_cam = scene["stereo_left"], scene["stereo_right"]
    rl_cam, rr_cam = scene["rear_stereo_left"], scene["rear_stereo_right"]
    num_envs = scene.num_envs
    device = robot.device
    hud_accum = 0.0
    follow = args_cli.follow_camera and not getattr(args_cli, "headless", False)

    headlights = HeadlightFollower(headlight_paths, device) if headlight_paths else None
    if hud:
        from pipe_rover.hud import update_hud     # omni.ui is only importable (and needed) with the GUI

    sprayer = None
    try:
        sprayer = SprayMarker(
            color=(args_cli.spray_color_r, args_cli.spray_color_g, args_cli.spray_color_b),
            dot_radius=args_cli.spray_dot_radius, cluster_size=args_cli.spray_cluster_size,
            cone_half_angle_deg=args_cli.spray_cone_half_angle, device=device)
        if sprayer.available:
            log.info(f"Spray marking ready: P=manual, auto every {args_cli.auto_spray_interval:.1f}s "
                     f"({args_cli.spray_cluster_size} rays, cone={args_cli.spray_cone_half_angle:.0f} deg "
                     f"half-angle, r={args_cli.spray_dot_radius * 1000:.1f} mm)")
        else:
            log.warn("Spray marking is unavailable (PhysX scene queries are not usable): no dots will be placed.")
    except Exception as err:
        log.warn(f"Spray system init failed: {err}")

    imu_noise = ImuNoiseModel(IMU_NAMES, IMU_ACCEL_NOISE_STD, IMU_GYRO_NOISE_STD,
                              IMU_ACCEL_BIAS_STD, IMU_GYRO_BIAS_STD)
    auto_spray_accum = 0.0

    def spray_now(trigger: str):
        try:
            sprayer.spray(root_pos, root_quat)
            log.debug(f"[SPRAY] {trigger} at t={step * sim_dt:.2f}s")
        except Exception as err:
            log.error(f"Spray failed ({trigger}): {type(err).__name__}: {err}")
            traceback.print_exc()

    while simulation_app.is_running():
        if teleop and getattr(teleop, "_exit", False):
            break

        if reset_requested:
            reset_requested = False
            reset_rover(robot, scene)
            log.info("Rover reset to its initial pose (an open bag keeps recording: the pose jumps).")

        actions = torch.zeros((num_envs, 2), device=device)
        cmd_vx, cmd_wz = 0.0, 0.0

        if teleop:
            base_cmd = teleop.advance()
            cmd_vx, cmd_wz = float(base_cmd[0]), float(base_cmd[2])
            l_val = (cmd_vx - 0.5 * cmd_wz * args_cli.wheel_base) / args_cli.wheel_radius
            r_val = (cmd_vx + 0.5 * cmd_wz * args_cli.wheel_base) / args_cli.wheel_radius
            actions = torch.tensor([[l_val, r_val]], device=device).repeat(num_envs, 1)
            actions = torch.clamp(actions, -args_cli.max_wheel_speed, args_cli.max_wheel_speed)

        robot.set_joint_velocity_target(actions, joint_ids=dof_idx)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1
        t_ns = step * dt_ns

        recorder.process_toggle_request()

        root_pos, root_quat = robot.data.root_pos_w[0], robot.data.root_quat_w[0]

        # IMU data with noise. Isaac Lab computes the acceleration lazily as a finite difference between two
        # reads of .data, so every IMU must be read exactly once per physics step (also while not recording).
        imu_dict = {}
        for name in IMU_NAMES:
            imu = scene[name]
            accel_noisy, gyro_noisy = imu_noise.apply(
                name, imu.data.lin_acc_b[0].detach().cpu().numpy(), imu.data.ang_vel_b[0].detach().cpu().numpy())
            imu_dict[name] = (accel_noisy.tolist(), gyro_noisy.tolist())

        if headlights is not None:
            headlights.update(root_pos, root_quat)

        # spray: manual (P key) and automatic timer
        if spray_requested and sprayer is not None:
            spray_requested = False
            auto_spray_accum = 0.0          # a manual burst restarts the automatic timer
            spray_now("manual")
        if sprayer is not None and args_cli.auto_spray_interval > 0:
            auto_spray_accum += sim_dt
            if auto_spray_accum >= args_cli.auto_spray_interval:
                auto_spray_accum = 0.0
                spray_now("auto")

        # follow camera (GUI viewport only)
        if follow:
            offset = torch.tensor([-args_cli.follow_distance, 0.0, args_cli.follow_height], device=device)
            eye = (root_pos + quat_apply(root_quat, offset)).detach().cpu().tolist()
            look_offset = torch.tensor([0.25, 0.0, 0.10], device=device)
            target = (root_pos + quat_apply(root_quat, look_offset)).detach().cpu().tolist()
            set_camera_view(eye, target)

        # HUD
        if hud:
            hud_accum += sim_dt
            if hud_accum >= HUD_UPDATE_PERIOD:
                hud_accum = 0.0
                update_hud(
                    hud, cmd_vx, cmd_wz, actions[0, 0].item(), actions[0, 1].item(),
                    (_camera_rgb(l_cam), _camera_rgb(r_cam), _camera_rgb(rl_cam), _camera_rgb(rr_cam)),
                    is_recording=recorder.recording, robot_pos=root_pos.detach().cpu().tolist(),
                    pipe_pos=pipe_pos)

        # ROS 2 recording
        if recorder.recording:
            recorder.publish_imus(imu_dict, t_ns)
            recorder.publish_ground_truth(root_pos.detach().cpu().tolist(), root_quat.detach().cpu().tolist(), t_ns)
            if recorder.image_due(sim_dt, IMAGE_RECORD_PERIOD):
                recorder.publish_images(_camera_rgb(l_cam), _camera_rgb(r_cam), t_ns,
                                        rear_left_rgb=_camera_rgb(rl_cam), rear_right_rgb=_camera_rgb(rr_cam))


def main():
    pipe_usd = scene_setup.require_pipe_asset(args_cli.pipe_usd)

    # PhysX scene queries (the spray ray casts) are off by default in Isaac Lab and are only switched on
    # automatically when the GUI is running, so they must be requested explicitly for --headless runs.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=PHYSICS_DT, device=args_cli.device, enable_scene_query_support=True))
    try:
        _build_and_run(sim, pipe_usd)
    finally:
        # Drop the simulation context so that Isaac Lab's "keep rendering after Stop" handler cannot
        # keep simulation_app.close() from returning.
        sim_utils.SimulationContext.clear_instance()


def _build_and_run(sim, pipe_usd):
    scene = InteractiveScene(RoverSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=True))

    # lights that must exist before sim.reset()
    scene_setup.disable_groundplane_light()
    scene_setup.create_sun_light(SUN_INTENSITY, SUN_COLOR, SUN_ELEVATION_DEG, SUN_AZIMUTH_DEG)
    headlight_paths = []
    try:
        headlight_paths = create_headlights(
            intensity=args_cli.headlight_intensity, cone_angle=args_cli.headlight_cone_angle,
            color=(args_cli.headlight_color_r, args_cli.headlight_color_g, args_cli.headlight_color_b),
            robot_init_pos=ROBOT_INIT_POS)
    except Exception as err:
        log.warn(f"Failed to create headlights: {err}")

    sim.reset()
    scene.reset()
    log.info(f"Scene initialized. Pipe asset: {pipe_usd}")

    # The pipe is essential: a failure here must stop the run (the other decorations only warn).
    # It is authored after sim.reset() as a static triangle-mesh collider; PhysX picks it up at the next step.
    pipe_prim_path = "/World/envs/env_0/Pipe"
    scene_setup.spawn_pipe(pipe_prim_path, pipe_usd, (args_cli.pipe_x, args_cli.pipe_y, args_cli.pipe_z),
                           args_cli.pipe_scale, PIPE_ROT_Z_NEG90)
    scene_setup.force_triangle_mesh_collision(pipe_prim_path)
    if args_cli.pipe_texture:
        try:
            tex = TEXTURE_DIR / f"pipe_corrosion_{args_cli.pipe_texture_size}_{args_cli.pipe_texture_seed}.png"
            if not tex.exists():
                TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
                generate_corrosion_texture(str(tex), size=args_cli.pipe_texture_size,
                                           seed=args_cli.pipe_texture_seed)
            scene_setup.apply_corrosion_material(pipe_prim_path, str(tex), PIPE_TEX_SCALE)
        except Exception as err:
            log.warn(f"Failed to apply the pipe texture: {err}")

    try:
        scene_setup.create_imu_markers("/World/envs/env_0/Robot", IMU_POSITIONS, IMU_ANGLES_DEG,
                                       IMU_MARKER_RADIUS, IMU_MARKER_Z_ABOVE)
    except Exception as err:
        log.warn(f"Failed to create IMU markers: {err}")

    try:
        TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
        mud_tex = TEXTURE_DIR / f"ground_mud_{GROUND_TEXTURE_SIZE}_{GROUND_TEXTURE_SEED}.png"
        if not mud_tex.exists():
            generate_mud_texture(str(mud_tex), size=GROUND_TEXTURE_SIZE, seed=GROUND_TEXTURE_SEED)
        scene_setup.apply_mud_ground_material("/World/defaultGroundPlane", str(mud_tex), GROUND_TEX_SCALE)
        sky_tex = TEXTURE_DIR / f"sky_overcast_{SKY_TEXTURE_SIZE[0]}x{SKY_TEXTURE_SIZE[1]}_{SKY_TEXTURE_SEED}.png"
        if not sky_tex.exists():
            generate_sky_texture(str(sky_tex), width=SKY_TEXTURE_SIZE[0], height=SKY_TEXTURE_SIZE[1],
                                 seed=SKY_TEXTURE_SEED)
        scene_setup.setup_outdoor_sky("/World/DomeLight", str(sky_tex))
    except Exception as err:
        log.warn(f"Failed to set up outdoor environment: {err}")

    recorder = Ros2StereoImuRecorder(enabled=args_cli.ros2_record, bag_dir=args_cli.ros2_bag_dir,
                                     bag_prefix=args_cli.ros2_bag_prefix)
    recorder.run_info = _run_info()
    hud = None
    if args_cli.hud and not getattr(args_cli, "headless", False):
        from pipe_rover.hud import create_hud
        hud = create_hud(args_cli.stereo_width, args_cli.stereo_height)

    robot = scene["robot"]
    dof_idx, _ = robot.find_joints(["left_wheel_joint", "right_wheel_joint"], preserve_order=True)
    dof_idx = torch.tensor(dof_idx, device=robot.device, dtype=torch.long)

    teleop = None
    if args_cli.control_mode == "teleop":
        teleop = Se2Keyboard(Se2KeyboardCfg(
            v_x_sensitivity=args_cli.teleop_linear_speed,
            omega_z_sensitivity=args_cli.teleop_angular_speed,
            sim_device=str(robot.device),
        ))

    if args_cli.seed is not None:
        np.random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)
        log.info(f"Random seed {args_cli.seed}")

    try:
        run_simulator(sim, scene, dof_idx, teleop, hud, recorder, headlight_paths,
                      pipe_pos=(args_cli.pipe_x, args_cli.pipe_y, args_cli.pipe_z))
    finally:
        recorder.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # simulation_app.close() ends the process (with status 0) before Python could print the error,
        # so report it first and leave with a non-zero status.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
