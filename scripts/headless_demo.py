# SPDX-License-Identifier: MIT
"""Headless smoke test and demo: drive the rover automatically, record a bag and check it.

The simulator normally needs a keyboard. This script imports ``pipe_rover_sim.py``, swaps its keyboard
device for a scripted driver, starts the bag recorder, runs for ``--duration`` simulated seconds and then
verifies the recording (topics, rates, motion, spray dots). It exits with a non-zero status if a check fails,
so it doubles as an installation test::

    python3 scripts/headless_demo.py                       # follow the pipe centre line (up to 20 s)
    python3 scripts/headless_demo.py --profile forward     # open-loop forward drive instead
    python3 scripts/headless_demo.py --duration 40 --seed 1 --spray_cluster_size 60

Anything this script does not recognise is passed on to the simulator (``--spray_cluster_size`` above).
Run it with Isaac Sim's Python and ROS 2 Humble sourced (``rosbag2_py``), like the simulator itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import check_bag  # noqa: E402


class ScriptedKeyboard:
    """Stands in for ``Se2Keyboard``: ``advance()`` returns (v_x, v_y, omega_z) from a drive profile."""

    START_RECORDING_AT = 0.5      # s of simulation time: let the rover settle after its drop first

    def __init__(self, profile: str, duration: float, dt: float, linear: float, angular: float,
                 holder: SimpleNamespace, path_file: Path):
        self.callbacks = {}
        self._exit = False
        self._recording_started = False
        self.step = 0
        self.dt = dt
        self.duration = duration
        self.profile = profile
        self.linear, self.angular = linear, angular
        self.holder = holder
        self.path = np.array(json.loads(path_file.read_text())["points"]) if profile == "path" else None
        self.finished_at = None

    # --- Se2Keyboard interface used by the simulator ---
    def add_callback(self, key, func):
        self.callbacks[key] = func

    def reset(self):
        pass

    def advance(self):
        t = self.step * self.dt
        if not self._recording_started and t >= self.START_RECORDING_AT and "B" in self.callbacks:
            self.callbacks["B"]()                 # start the bag recording
            self._recording_started = True
        self.step += 1
        if t >= self.duration:
            self._exit = True
            return np.zeros(3)
        vx, wz = self._command(t)
        return np.array([vx, 0.0, wz])

    # --- drive profiles ---
    def _command(self, t: float):
        if self.profile == "forward":
            if t < 1.0:
                return 0.0, 0.0
            return self.linear, 0.0
        return self._pure_pursuit(t)

    def _pose(self):
        robot = self.holder.scene["robot"]
        p = robot.data.root_pos_w[0].detach().cpu().numpy()
        w, x, y, z = robot.data.root_quat_w[0].detach().cpu().numpy()
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return p[:2], yaw

    def _pure_pursuit(self, t: float):
        if t < 1.0:
            return 0.0, 0.0
        pos, yaw = self._pose()
        d = np.linalg.norm(self.path - pos, axis=1)
        i = int(np.argmin(d))
        if i >= len(self.path) - 2 and d[-1] < 0.25:      # reached the far mouth
            if self.finished_at is None:
                self.finished_at = t
            if t - self.finished_at > 1.0:
                self._exit = True
            return 0.0, 0.0
        j = min(i + 4, len(self.path) - 1)                # look ~0.25 m ahead
        err = math.atan2(self.path[j][1] - pos[1], self.path[j][0] - pos[0]) - yaw
        err = (err + math.pi) % (2 * math.pi) - math.pi
        wz = float(np.clip(2.0 * err, -self.angular, self.angular))
        vx = self.linear * max(0.3, math.cos(err))
        return vx, wz


def load_simulator(path: Path, argv):
    """Import the simulator script as a module (this launches the Isaac Sim app)."""
    sys.argv = [str(path)] + list(argv)
    spec = importlib.util.spec_from_file_location("simulator_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog="Unrecognised arguments are passed to the simulator.")
    parser.add_argument("--script", type=Path, default=REPO_ROOT / "pipe_rover_sim.py")
    parser.add_argument("--profile", choices=["path", "forward"], default="path",
                        help="path: pure-pursuit along the pipe centre line; forward: open-loop straight drive")
    parser.add_argument("--duration", type=float, default=20.0, help="simulated seconds to run")
    parser.add_argument("--bag-dir", type=Path, default=REPO_ROOT / "rosbags" / "headless_demo")
    parser.add_argument("--np-seed", type=int, default=None,
                        help="seed NumPy and PyTorch right before the run starts (works with any simulator version)")
    parser.add_argument("--force-scene-queries", action="store_true",
                        help="force PhysX scene queries on (lets an older simulator version spray when headless)")
    parser.add_argument("--no-checks", action="store_true", help="skip the verification of the recorded bag")
    args, sim_args = parser.parse_known_args()

    argv = ["--headless", "--control_mode", "teleop", "--no_hud", "--ros2_bag_dir", str(args.bag_dir)] + sim_args
    sim = load_simulator(args.script.resolve(), argv)

    holder = SimpleNamespace(scene=None)

    class CapturingScene(sim.InteractiveScene):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            holder.scene = self

    sim.InteractiveScene = CapturingScene
    keyboard = {}

    def make_keyboard(cfg):
        keyboard["obj"] = ScriptedKeyboard(args.profile, args.duration, sim.PHYSICS_DT,
                                           cfg.v_x_sensitivity, cfg.omega_z_sensitivity, holder,
                                           HERE / "demo_path.json")
        return keyboard["obj"]

    sim.Se2Keyboard = make_keyboard
    if args.force_scene_queries:
        cfg_cls = sim.sim_utils.SimulationCfg
        sim.sim_utils.SimulationCfg = lambda **kw: cfg_cls(**{**kw, "enable_scene_query_support": True})
    pre = {}
    existing_bags = _bag_dirs(args.bag_dir)
    if args.np_seed is not None:
        import torch
        np.random.seed(args.np_seed)
        torch.manual_seed(args.np_seed)
    code = 1
    try:
        sim.main()
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/SprayDots")
        pre["spray_dots"] = len(root.GetChildren()) if root and root.IsValid() else 0
        code = 0 if args.no_checks else verify(args, pre, existing_bags)
    except BaseException:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    if code != 0:
        os._exit(code)              # simulation_app.close() would exit with status 0
    sim.simulation_app.close()      # exits the process
    return code


def _bag_dirs(bag_dir) -> set:
    """Bag directories (those holding a .db3 file) below ``bag_dir``."""
    return {p.parent for p in Path(bag_dir).glob("*/*.db3")}


def _distance_travelled(bag_dir: Path):
    """Straight-line distance between the first and last ground-truth position (None without ROS bindings)."""
    try:
        import sqlite3

        from geometry_msgs.msg import PoseStamped
        from rclpy.serialization import deserialize_message
    except ImportError:
        return None
    db = next(bag_dir.glob("*.db3")).resolve().as_uri()
    con = sqlite3.connect(f"{db}?mode=ro", uri=True)
    try:
        tid = con.execute("SELECT id FROM topics WHERE name = '/ground_truth/pose'").fetchone()[0]
        rows = con.execute("SELECT data FROM messages WHERE topic_id = ? ORDER BY id", (tid,)).fetchall()
    finally:
        con.close()
    first, last = (deserialize_message(rows[i][0], PoseStamped).pose.position for i in (0, -1))
    return math.dist((first.x, first.y, first.z), (last.x, last.y, last.z))


def verify(args, pre, existing_bags=frozenset()) -> int:
    """Check the bag written by *this* run (the only new bag below ``--bag-dir``); returns the exit status."""
    new_bags = sorted(_bag_dirs(args.bag_dir) - set(existing_bags))
    ok = True

    def check(cond, text):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + text)
        ok &= bool(cond)

    check(len(new_bags) == 1, f"exactly one new bag was written under {args.bag_dir} (found {len(new_bags)})")
    if len(new_bags) != 1:
        return 1
    bag = new_bags[0]
    stats = check_bag.summarise(bag)
    topics_ok = all(t in stats for t in check_bag.EXPECTED_TOPICS)
    check(topics_ok, "all 8 topics are present")
    if not topics_ok:
        return 1
    for t in ("/imu0", "/imu1", "/imu2", "/ground_truth/pose"):
        check(abs(stats[t]["rate_hz"] - 200.0) < 1.0, f"{t} runs at 200 Hz (measured {stats[t]['rate_hz']:.2f})")
    for t in check_bag.EXPECTED_TOPICS[:4]:
        check(abs(stats[t]["rate_hz"] - 28.57) < 0.5, f"{t} runs at ~28.6 Hz (measured {stats[t]['rate_hz']:.2f})")
    check(all(s["monotonic"] for s in stats.values()), "time stamps are monotonic in write order")
    moved = _distance_travelled(bag)
    check(moved is not None and moved > 0.1,
          "the rover moved" + (" (ROS bindings missing, cannot tell)" if moved is None else f" ({moved:.2f} m)"))
    check(pre.get("spray_dots", 0) > 0, f"spray dots were created ({pre.get('spray_dots', 0)})")
    print(f"bag: {bag}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
