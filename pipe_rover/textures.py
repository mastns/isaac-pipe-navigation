# SPDX-License-Identifier: MIT
"""Procedural textures for the pipe-rover scene.

Three seeded generators produce the images that dress the simulated scene:

* :func:`generate_mud_texture`        - the muddy ground around the pipe,
* :func:`generate_sky_texture`        - the overcast sky dome,
* :func:`generate_corrosion_texture`  - the (optional) corroded pipe wall.

All functions are deterministic for a given seed, depend only on ``numpy`` and
``Pillow`` and run without Isaac Sim. The order in which random numbers are drawn
is part of the contract: the images of the original script (written to
``assets/textures`` on the first run) were produced with exactly this sequence, and
``tests/test_textures.py`` checks that the generators still reproduce them pixel for pixel.
"""

from __future__ import annotations

import numpy as np

from . import log

__all__ = ["generate_corrosion_texture", "generate_mud_texture", "generate_sky_texture"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _grid(height: int, width: int):
    """Open grids ``(yy, xx)`` used to evaluate blobs on the image plane."""
    return np.ogrid[:height, :width]


def _radial_blob(yy, xx, cx, cy, radius, power):
    """Soft circular blob in [0, 1]: ``clip(1 - dist/radius, 0, 1) ** power``."""
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(1.0 - dist / radius, 0, 1) ** power


def _elliptic_blob(yy, xx, cx, cy, rx, ry, power):
    """Soft elliptic blob in [0, 1]: ``clip(1 - d, 0, 1) ** power`` with ``d`` the squared elliptic distance."""
    dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    return np.clip(1.0 - dist, 0, 1) ** power


def _blend(img, blob, strength, color):
    """Alpha-blend ``color`` into ``img`` (float image, in place) with weight ``blob * strength``.

    The expression order matches the original implementation exactly so that the
    floating point result - and therefore the 8-bit image - is bit-identical.
    """
    keep = 1 - blob * strength
    for c in range(3):
        img[:, :, c] = img[:, :, c] * keep + color[c] * blob * strength


def _gaussian_base(rng, size, means_stds):
    """Per-channel Gaussian noise image (float64, clipped to 0..255)."""
    chans = [rng.normal(mean, std, (size, size)).clip(0, 255) for mean, std in means_stds]
    return np.stack(chans, axis=-1).astype(np.float64)


def _clip_window(size, cx, cy, r):
    """Slices of the square window of half-size ``r`` around (cx, cy), clipped to the image."""
    return (slice(max(0, cy - r), min(size, cy + r + 1)),
            slice(max(0, cx - r), min(size, cx + r + 1)))


def _finish(img, noise_std, rng, shape):
    noise = rng.normal(0, noise_std, shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _save_png(img: np.ndarray, path: str) -> None:
    try:
        from PIL import Image
    except ImportError as err:  # pragma: no cover - Pillow ships with Isaac Sim
        raise ImportError("Pillow is required to write the procedural textures "
                          "(pip install pillow)") from err
    Image.fromarray(img).save(path)


# --------------------------------------------------------------------------- #
# Pipe wall
# --------------------------------------------------------------------------- #

def generate_corrosion_texture(path: str, size: int = 512, seed: int = 42) -> str:
    """Corroded, stained and fouled pipe wall (RGB, ``size`` x ``size``)."""
    rng = np.random.RandomState(seed)
    img = _gaussian_base(rng, size, [(55, 10), (48, 8), (42, 7)])
    yy, xx = _grid(size, size)

    # rust patches: 80 overlapping blobs build a mask, then 3 rust tones are blended through it
    rust_mask = np.zeros((size, size), dtype=np.float64)
    for _ in range(80):
        cx, cy = rng.randint(0, size, 2)
        radius = rng.randint(30, 140)
        blob = _radial_blob(yy, xx, cx, cy, radius, 1.5)
        rust_mask += blob * rng.uniform(0.4, 1.0)
    rust_mask = np.clip(rust_mask, 0, 1)
    for _ in range(3):
        rust_color = np.array([rng.randint(110, 170), rng.randint(40, 80), rng.randint(15, 40)], dtype=np.float64)
        sub_mask = rust_mask * rng.uniform(0.3, 0.6, (size, size))
        _blend(img, sub_mask, 0.85, rust_color)

    # pitting: small dark discs
    num_pits = int(size * size * 0.04)
    pit_x = rng.randint(0, size, num_pits)
    pit_y = rng.randint(0, size, num_pits)
    pit_radius = rng.randint(1, 6, num_pits)
    pit_dark = rng.randint(50, 120, num_pits)
    for px, py, pr, pd in zip(pit_x, pit_y, pit_radius, pit_dark, strict=True):
        wy, wx = np.ogrid[max(0, py - pr):min(size, py + pr + 1), max(0, px - pr):min(size, px + pr + 1)]
        mask = (wx - px) ** 2 + (wy - py) ** 2 <= pr ** 2
        sl_y, sl_x = _clip_window(size, px, py, pr)
        img[sl_y, sl_x][mask] = np.clip(img[sl_y, sl_x][mask] - pd, 0, 255)

    # biological growth (green or black)
    for _ in range(40):
        cx, cy = rng.randint(0, size, 2)
        radius = rng.randint(15, 60)
        blob = _radial_blob(yy, xx, cx, cy, radius, 2)
        strength = rng.uniform(0.3, 0.7)
        if rng.random() < 0.6:
            bio_color = np.array([25, rng.randint(50, 80), 20], dtype=np.float64)
        else:
            bio_color = np.array([20, 20, 18], dtype=np.float64)
        _blend(img, blob, strength, bio_color)

    # mineral deposits (pale)
    for _ in range(35):
        cx, cy = rng.randint(0, size, 2)
        radius = rng.randint(5, 25)
        blob = _radial_blob(yy, xx, cx, cy, radius, 1.5)
        strength = rng.uniform(0.3, 0.6)
        mineral_color = np.array([rng.randint(170, 220), rng.randint(170, 200), rng.randint(120, 160)],
                                 dtype=np.float64)
        _blend(img, blob, strength, mineral_color)

    # vertical water streaks
    for _ in range(50):
        x = rng.randint(0, size)
        y_start = rng.randint(0, size // 3)
        length = rng.randint(60, 250)
        width = rng.randint(2, 6)
        darkness = rng.randint(30, 80)
        for row in range(y_start, min(y_start + length, size)):
            x_cur = int(np.clip(x + rng.normal(0, 0.8), 0, size - width))
            x_e = min(x_cur + width, size)
            fade = 1.0 - (row - y_start) / max(length, 1)
            stain = np.array([40, 25, 15], dtype=np.float64) * fade
            img[row, x_cur:x_e] = np.clip(img[row, x_cur:x_e] - darkness * fade + stain, 0, 255)

    # remnants of old paint
    for _ in range(6):
        cx, cy = rng.randint(size // 4, 3 * size // 4, 2)
        rx, ry = rng.randint(20, 50), rng.randint(5, 15)
        blob = _elliptic_blob(yy, xx, cx, cy, rx, ry, 1.2)
        strength = rng.uniform(0.1, 0.3)
        paint_color = np.array([rng.randint(180, 230), rng.randint(170, 210), rng.randint(80, 130)],
                               dtype=np.float64)
        _blend(img, blob, strength, paint_color)

    out = _finish(img, 8, rng, (size, size, 3))
    _save_png(out, path)
    log.info(f"Abandoned corrosion texture generated: {path} ({size}x{size})")
    return path


# --------------------------------------------------------------------------- #
# Ground
# --------------------------------------------------------------------------- #

def generate_mud_texture(path: str, size: int = 1024, seed: int = 99) -> str:
    """Muddy ground with puddles, grass tufts, pebbles, oil stains and tyre tracks."""
    rng = np.random.RandomState(seed)
    img = _gaussian_base(rng, size, [(85, 12), (70, 10), (50, 8)])
    yy, xx = _grid(size, size)

    # dark wet mud
    for _ in range(60):
        cx, cy = rng.randint(0, size, 2)
        radius = rng.randint(40, 150)
        blob = _radial_blob(yy, xx, cx, cy, radius, 2)
        strength = rng.uniform(0.3, 0.7)
        _blend(img, blob, strength, np.array([35, 30, 25], dtype=np.float64))

    # dry grass
    for _ in range(45):
        cx, cy = rng.randint(0, size, 2)
        radius = rng.randint(15, 70)
        blob = _radial_blob(yy, xx, cx, cy, radius, 1.5)
        strength = rng.uniform(0.2, 0.5)
        grass_color = np.array([rng.randint(100, 140), rng.randint(90, 120), rng.randint(30, 55)],
                               dtype=np.float64)
        _blend(img, blob, strength, grass_color)

    # pebbles
    num_pebbles = int(size * size * 0.008)
    px_arr = rng.randint(0, size, num_pebbles)
    py_arr = rng.randint(0, size, num_pebbles)
    pr_arr = rng.randint(1, 4, num_pebbles)
    sc_arr = rng.randint(70, 145, num_pebbles)
    for px, py, pr, sc in zip(px_arr, py_arr, pr_arr, sc_arr, strict=True):
        wy, wx = np.ogrid[max(0, py - pr):min(size, py + pr + 1), max(0, px - pr):min(size, px + pr + 1)]
        mask = (wx - px) ** 2 + (wy - py) ** 2 <= pr ** 2
        sl_y, sl_x = _clip_window(size, px, py, pr)
        variation = rng.randint(-15, 15, 3)
        for c in range(3):
            img[sl_y, sl_x, c][mask] = np.clip(sc + variation[c], 0, 255)

    # oil stains
    for _ in range(12):
        cx, cy = rng.randint(0, size, 2)
        radius = rng.randint(20, 80)
        blob = _radial_blob(yy, xx, cx, cy, radius, 2.5)
        strength = rng.uniform(0.4, 0.8)
        _blend(img, blob, strength, np.array([15, 12, 10], dtype=np.float64))

    # green moss
    for _ in range(30):
        cx, cy = rng.randint(0, size, 2)
        radius = rng.randint(10, 45)
        blob = _radial_blob(yy, xx, cx, cy, radius, 2)
        strength = rng.uniform(0.2, 0.5)
        green_color = np.array([rng.randint(30, 60), rng.randint(55, 90), rng.randint(20, 40)],
                               dtype=np.float64)
        _blend(img, blob, strength, green_color)

    # tracks
    for _ in range(5):
        x_start = rng.randint(0, size)
        y_start = rng.randint(0, size)
        angle = rng.uniform(0, np.pi)
        length = rng.randint(150, 500)
        width = rng.randint(4, 10)
        darkness = rng.randint(15, 35)
        for step in range(length):
            x = int(np.clip(x_start + step * np.cos(angle) + rng.normal(0, 0.5), 0, size - width - 1))
            y = int(np.clip(y_start + step * np.sin(angle) + rng.normal(0, 0.5), 0, size - 1))
            img[y, x:x + width] = np.clip(img[y, x:x + width] - darkness, 0, 255)

    out = _finish(img, 6, rng, (size, size, 3))
    _save_png(out, path)
    log.info(f"Abandoned area ground texture generated: {path} ({size}x{size})")
    return path


# --------------------------------------------------------------------------- #
# Sky
# --------------------------------------------------------------------------- #

def generate_sky_texture(path: str, width: int = 2048, height: int = 1024, seed: int = 88) -> str:
    """Equirectangular overcast sky (grey gradient, cloud banks, a few blue gaps, dark ground band)."""
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width, 3), dtype=np.float64)
    for y in range(height):
        t = y / height
        if t < 0.25:
            s = t / 0.25
            r = 120 + s * 15
            g = 125 + s * 15
            b = 140 + s * 10
        elif t < 0.45:
            s = (t - 0.25) / 0.20
            r = 135 + s * 20
            g = 140 + s * 18
            b = 150 + s * 10
        elif t < 0.55:
            s = (t - 0.45) / 0.10
            r = 155 + s * 25
            g = 158 + s * 22
            b = 160 + s * 15
        else:
            s = (t - 0.55) / 0.45
            r = 180 - s * 140
            g = 180 - s * 140
            b = 175 - s * 135
        img[y, :] = [r, g, b]
    yy, xx = _grid(height, width)

    # cloud banks
    for _ in range(25):
        base_cx = rng.randint(0, width)
        base_cy = rng.randint(int(height * 0.05), int(height * 0.45))
        for _ in range(rng.randint(8, 18)):
            cx = base_cx + rng.randint(-200, 200)
            cy = base_cy + rng.randint(-40, 40)
            rx = rng.randint(100, 300)
            ry = rng.randint(30, 80)
            blob = _elliptic_blob(yy, xx, cx, cy, rx, ry, 1.5)
            strength = rng.uniform(0.3, 0.6)
            cloud_grey = rng.randint(90, 140)
            cloud_color = np.array([cloud_grey, cloud_grey + 2, cloud_grey + 5], dtype=np.float64)
            _blend(img, blob, strength, cloud_color)

    # gaps of blue sky
    for _ in range(5):
        cx = rng.randint(0, width)
        cy = rng.randint(int(height * 0.10), int(height * 0.35))
        rx = rng.randint(60, 150)
        ry = rng.randint(20, 50)
        blob = _elliptic_blob(yy, xx, cx, cy, rx, ry, 2.0)
        strength = rng.uniform(0.2, 0.4)
        _blend(img, blob, strength, np.array([100, 120, 165], dtype=np.float64))

    # dark cloud bases near the horizon
    for _ in range(10):
        cx = rng.randint(0, width)
        cy = rng.randint(int(height * 0.30), int(height * 0.52))
        rx = rng.randint(150, 400)
        ry = rng.randint(20, 50)
        blob = _elliptic_blob(yy, xx, cx, cy, rx, ry, 1.8)
        strength = rng.uniform(0.2, 0.5)
        _blend(img, blob, strength, np.array([70, 72, 75], dtype=np.float64))

    out = _finish(img, 2.0, rng, (height, width, 3))
    _save_png(out, path)
    log.info(f"Sky texture generated: {path} ({width}x{height})")
    return path
