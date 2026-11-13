# SPDX-License-Identifier: MIT
"""Pixel-level behaviour of the recorder helpers (no ROS needed)."""

import hashlib
import math

import numpy as np
import pytest

from pipe_rover.recorder import Ros2StereoImuRecorder, rgb_to_mono8, to_rgb_uint8


def _original_gray(rgb):
    """The grayscale conversion of the original script (float32 arithmetic, truncating cast)."""
    return (0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)).clip(0, 255).astype(np.uint8)


def _ramp_and_noise():
    g = np.arange(256, dtype=np.uint8)
    ramp = np.stack([g, g, g], axis=-1)[None]          # R = G = B: the cases that float precision decides
    noise = np.random.RandomState(0).randint(0, 256, (240, 320, 3)).astype(np.uint8)
    return ramp, noise


def test_mono8_is_bit_identical_to_the_original_conversion_on_grey_ramp_and_noise():
    for img in _ramp_and_noise():
        assert np.array_equal(rgb_to_mono8(img), _original_gray(img))


def test_mono8_golden_digest():
    ramp, noise = _ramp_and_noise()
    digest = hashlib.sha256(rgb_to_mono8(ramp).tobytes() + rgb_to_mono8(noise).tobytes()).hexdigest()[:16]
    assert digest == "6cb97368c5f0b4d5"


def test_to_rgb_uint8_keeps_the_values():
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    rgb = to_rgb_uint8(gray)
    assert all(np.array_equal(rgb[..., c], gray) for c in range(3))
    rgba = np.zeros((2, 2, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 10, 20, 30, 40
    assert (to_rgb_uint8(rgba) == (10, 20, 30)).all()
    assert (to_rgb_uint8(np.ones((2, 2, 3), np.float32)) == 255).all()          # 0..1 floats incl. exactly 1.0
    assert to_rgb_uint8(np.zeros((2, 2, 3), np.float32)).max() == 0


def test_image_due_fires_on_equality():
    rec = Ros2StereoImuRecorder(enabled=False)
    assert [rec.image_due(0.25, 0.5) for _ in range(4)] == [False, True, False, True]


@pytest.mark.parametrize("period", [0.01, 0.033, 0.05, 0.055, 0.06, 0.07, 0.1])
def test_image_cadence_matches_the_step_count_written_to_sim_config(period):
    """pipe_rover_sim._run_info records ceil(period / dt - 1e-9) steps per image; image_due must really do that."""
    rec = Ros2StereoImuRecorder(enabled=False)
    due = [i for i in range(400) if rec.image_due(0.005, period)]
    assert {b - a for a, b in zip(due, due[1:], strict=False)} == {math.ceil(period / 0.005 - 1e-9)}
