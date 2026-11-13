# SPDX-License-Identifier: MIT
"""The numbers that define the simulated experiment and the command-line checks of pipe_rover_sim.py.

The script cannot be imported without Isaac Sim, so its constants are read with ``ast`` and its argument parser /
validator are compiled on their own (with a stand-in for Isaac Lab's ``AppLauncher``)."""

import argparse
import ast
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "pipe_rover_sim.py"

# Values of the original script that this simulator derives from; a change here must be deliberate (and documented).
ORIGINAL = {
    "PHYSICS_DT": 0.005, "IMAGE_RECORD_PERIOD": 0.033,
    "ROBOT_INIT_POS": (0.3, 0.0, 0.15), "WHEEL_RADIUS": 0.0325, "WHEEL_BASE": 0.112,
    "TELEOP_LINEAR_SPEED": 0.25, "TELEOP_ANGULAR_SPEED": 2.0,
    "IMU_NAMES": ("imu_0", "imu_1", "imu_2"), "IMU_RADIUS": 0.05, "IMU_ANGLES_DEG": (0, 120, 240),
    "IMU_GYRO_NOISE_STD": 0.005, "IMU_ACCEL_NOISE_STD": 0.05, "IMU_GYRO_BIAS_STD": 0.001, "IMU_ACCEL_BIAS_STD": 0.005,
    "STEREO_WIDTH": 320, "STEREO_HEIGHT": 240, "STEREO_BASELINE": 0.06, "STEREO_MOUNT_X": 0.12, "STEREO_MOUNT_Z": 0.10,
    "REAR_STEREO_MOUNT_X": -0.10, "REAR_STEREO_MOUNT_Z": 0.10,
    "PIPE_SCALE": 1.5, "PIPE_POS": (0.1, 0.0, 0.375), "PIPE_TEXTURE_ON": False,
    "PIPE_TEXTURE_SEED": 42, "PIPE_TEXTURE_SIZE": 512,
    "GROUND_TEXTURE_SEED": 99, "GROUND_TEXTURE_SIZE": 1024, "SKY_TEXTURE_SEED": 88, "SKY_TEXTURE_SIZE": (2048, 1024),
    "HEADLIGHT_INTENSITY": 5000.0, "HEADLIGHT_CONE_DEG": 35.0,
    "SPRAY_DOT_RADIUS": 0.006, "SPRAY_CLUSTER_SIZE": 120, "SPRAY_CONE_HALF_ANGLE": 70.0, "AUTO_SPRAY_INTERVAL": 0.5,
    "SPRAY_COLOR": (1.0, 0.35, 0.0),
}


def _module_nodes():
    return ast.parse(SCRIPT.read_text(encoding="utf-8")).body


def _constants():
    out = {}
    for node in _module_nodes():
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def test_simulator_defaults_are_the_ones_of_the_original_script():
    found = _constants()
    assert {k: found.get(k) for k in ORIGINAL} == ORIGINAL


def test_golden_texture_images_are_made_with_the_simulator_defaults():
    from test_textures import GOLDEN
    c = _constants()
    kwargs = {name: kw for name, kw, _ in GOLDEN}
    assert kwargs["generate_mud_texture"] == {"size": c["GROUND_TEXTURE_SIZE"], "seed": c["GROUND_TEXTURE_SEED"]}
    assert kwargs["generate_corrosion_texture"] == {"size": c["PIPE_TEXTURE_SIZE"],
                                                    "seed": c["PIPE_TEXTURE_SEED"]}
    assert kwargs["generate_sky_texture"] == {"width": c["SKY_TEXTURE_SIZE"][0], "height": c["SKY_TEXTURE_SIZE"][1],
                                              "seed": c["SKY_TEXTURE_SEED"]}


def test_image_period_gives_the_documented_28_57_hz():
    c = _constants()
    steps = math.ceil(c["IMAGE_RECORD_PERIOD"] / c["PHYSICS_DT"] - 1e-9)
    assert steps == 7 and 1.0 / (steps * c["PHYSICS_DT"]) == pytest.approx(28.571, abs=1e-3)


class _FakeAppLauncher:
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")


@pytest.fixture(scope="module")
def cli():
    nodes = _module_nodes()
    stop = next(i for i, n in enumerate(nodes) if isinstance(n, ast.ImportFrom) and n.module == "isaaclab.app")
    keep = [n for n in nodes[:stop] if isinstance(n, ast.Assign)]
    keep += [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in {"_build_parser", "_validate"}]
    namespace = {"argparse": argparse, "math": math, "Path": Path, "__file__": str(SCRIPT),
                 "AppLauncher": _FakeAppLauncher}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


def test_parser_defaults_match_the_documented_defaults(cli):
    args = cli["_build_parser"]().parse_args([])
    assert (args.num_envs, args.seed, args.control_mode) == (1, None, "teleop")
    assert (args.auto_spray_interval, args.spray_cluster_size, args.spray_cone_half_angle) == (0.5, 120, 70.0)
    assert (args.stereo_width, args.stereo_height, args.stereo_baseline) == (320, 240, 0.06)
    assert args.pipe_texture is False and args.hud is True and args.ros2_record is True


@pytest.mark.parametrize("argv", [
    ["--num_envs", "2"], ["--num_envs", "0"], ["--wheel_radius", "0"], ["--wheel_base", "-1"],
    ["--max_wheel_speed", "0"],
    ["--stereo_width", "0"], ["--stereo_height", "-5"], ["--pipe_scale", "0"], ["--pipe_texture_size", "0"],
    ["--spray_cluster_size", "-1"], ["--auto_spray_interval", "-0.1"],
    ["--spray_cone_half_angle", "0"], ["--spray_cone_half_angle", "180.5"],
])
def test_invalid_options_are_rejected(cli, argv):
    parser = cli["_build_parser"]()
    with pytest.raises(SystemExit):
        cli["_validate"](parser.parse_args(argv), parser)


@pytest.mark.parametrize("argv", [
    [], ["--spray_cluster_size", "0"], ["--auto_spray_interval", "0"], ["--spray_cone_half_angle", "180"],
    ["--seed", "7"],
])
def test_boundary_options_are_accepted(cli, argv):
    parser = cli["_build_parser"]()
    cli["_validate"](parser.parse_args(argv), parser)


def test_every_no_flag_is_documented_on_its_own():
    """``--no_x`` must itself appear in docs/configuration.md (a bare ``x`` is a substring of ``--x``)."""
    doc = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    flags = [n.args[0].value for n in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8")))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
             and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value.startswith("--no_")]
    assert len(flags) == 4
    assert [f for f in flags if f not in doc] == []
