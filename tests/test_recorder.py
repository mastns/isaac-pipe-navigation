# SPDX-License-Identifier: MIT
"""Recorder helpers that do not need ROS 2 (the bag round trip is in test_recorder_rosbag.py)."""

import numpy as np
import pytest

from pipe_rover.recorder import Ros2StereoImuRecorder, rgb_to_mono8, to_rgb_uint8


def test_disabled_recorder_is_a_noop(tmp_path):
    rec = Ros2StereoImuRecorder(enabled=False, bag_dir=str(tmp_path))
    rec.start_recording()
    rec.publish_imus({}, 0)
    rec.publish_ground_truth([0, 0, 0], [1, 0, 0, 0], 0)
    rec.publish_images(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8), 0)
    rec.shutdown()
    assert not rec.recording and not list(tmp_path.iterdir())


def test_image_cadence_is_every_seventh_physics_step():
    rec = Ros2StereoImuRecorder(enabled=False)
    due = [rec.image_due(0.005, 0.033) for _ in range(70)]
    assert [i for i, d in enumerate(due) if d] == [6 + 7 * k for k in range(10)]
    # 7 steps of 5 ms = 35 ms -> 28.57 Hz, not the 30 Hz suggested by the 0.033 s setting
    assert 1.0 / (7 * 0.005) == pytest.approx(28.571, abs=1e-3)


def test_to_rgb_uint8_conversions():
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert to_rgb_uint8(gray).shape == (3, 4, 3)
    rgba = np.full((2, 2, 4), 7, np.uint8)
    assert to_rgb_uint8(rgba).shape == (2, 2, 3)
    unit = np.full((2, 2, 3), 0.5, np.float32)
    assert to_rgb_uint8(unit).max() == 127  # 0..1 floats are scaled to 0..255
    assert to_rgb_uint8(np.full((2, 2, 3), 300.0)).max() == 255  # out-of-range values are clipped


def test_mono8_uses_bt601_luma():
    rgb = np.zeros((1, 3, 3), np.uint8)
    rgb[0, 0] = (255, 0, 0)
    rgb[0, 1] = (0, 255, 0)
    rgb[0, 2] = (0, 0, 255)
    assert rgb_to_mono8(rgb).tolist() == [[76, 149, 29]]


def test_stamp_splits_integer_nanoseconds():
    assert Ros2StereoImuRecorder._stamp(35_000_000) == (0, 35_000_000)
    assert Ros2StereoImuRecorder._stamp(12_345_678_901) == (12, 345_678_901)
    assert Ros2StereoImuRecorder._stamp(2_000_000_000) == (2, 0)
