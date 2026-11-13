# SPDX-License-Identifier: MIT
"""On-screen "Stereo View" window with the four camera images, plus a one-line status print per second.

Needs the Isaac Sim GUI (``omni.ui``); import it after the app has been launched and only use it when
the app is not headless.
"""

from __future__ import annotations

import time

import numpy as np
import omni.ui as ui

from . import log

__all__ = ["create_hud", "update_hud"]

_KEYS = ("front_left", "front_right", "rear_left", "rear_right")


def _to_rgba_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 3:
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        image = np.concatenate([image, alpha], axis=-1)
    return image


def create_hud(width: int, height: int):
    """Open the 2x2 stereo window for ``width`` x ``height`` camera images and return its handles."""
    hud = {}
    dock = None
    if hasattr(ui, "DockPreference"):
        for attr in ("LEFT_TOP", "LEFT", "TOP"):
            if hasattr(ui.DockPreference, attr):
                dock = getattr(ui.DockPreference, attr)
                break
    kwargs = {"width": 2 * width + 24, "height": 2 * height + 48}
    if dock is not None:
        kwargs["dockPreference"] = dock
    hud["window"] = ui.Window("Stereo View", **kwargs)
    init_image = np.zeros((height, width, 4), dtype=np.uint8)
    with hud["window"].frame:
        with ui.VStack(spacing=4):
            for row in (("front_left", "front_right"), ("rear_left", "rear_right")):
                with ui.HStack(spacing=8):
                    for key in row:
                        provider = ui.ByteImageProvider()
                        provider.set_bytes_data(init_image.flatten().data, [width, height])
                        hud[f"{key}_provider"] = provider
                    for key in row:
                        ui.ImageWithProvider(hud[f"{key}_provider"], width=width, height=height)
    hud["_last_print"] = 0.0
    return hud


def update_hud(hud, cmd_vx, cmd_wz, left_w, right_w, images, is_recording=False,
               robot_pos=None, pipe_pos=None):
    """Refresh the four images (``images`` = fl, fr, rl, rr RGB arrays) and print a status line at 1 Hz."""
    if hud is None:
        return
    now = time.monotonic()
    if now - hud["_last_print"] >= 1.0:
        hud["_last_print"] = now
        rec_str = "ON" if is_recording else "OFF"
        pos_str = ""
        if robot_pos is not None:
            pos_str = f"  |  Robot: ({robot_pos[0]:+.3f}, {robot_pos[1]:+.3f}, {robot_pos[2]:+.3f})"
        pipe_str = ""
        if pipe_pos is not None:
            pipe_str = f"  |  Pipe: ({pipe_pos[0]:+.3f}, {pipe_pos[1]:+.3f}, {pipe_pos[2]:+.3f})"
        log.info(f"HUD v={cmd_vx:+.2f} w={cmd_wz:+.2f}  |  L={left_w:+.1f} R={right_w:+.1f}  |  "
                 f"Rec: {rec_str}{pos_str}{pipe_str}")
    for key, img in zip(_KEYS, images, strict=True):
        rgba = _to_rgba_uint8(img)
        hud[f"{key}_provider"].set_bytes_data(rgba.flatten().data, [rgba.shape[1], rgba.shape[0]])
