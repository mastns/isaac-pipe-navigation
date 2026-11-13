# SPDX-License-Identifier: MIT
"""Helper modules of the isaac-pipe-navigation simulation.

``textures``, ``imu_noise``, ``spray``, ``recorder`` and ``log`` run without Isaac Sim (NumPy, Pillow and, for the
bag recorder, the ROS 2 Python bindings), so they can be unit-tested on any machine. ``scene_setup``,
``headlights``, ``spray_marker`` and ``hud`` need the running Isaac Sim app. The simulator itself is
``pipe_rover_sim.py`` at the repository root.
"""

__version__ = "1.0.0"
