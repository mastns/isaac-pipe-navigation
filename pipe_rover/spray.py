# SPDX-License-Identifier: MIT
"""Geometry helpers of the spray-marking system.

Each spray burst casts ``spray_cluster_size`` rays from the rover centre inside a cone around the
rover's forward axis; every ray that hits the pipe wall leaves a small coloured sphere (an artificial
visual feature). Only the random sampling and the placement arithmetic live here so that they can be
tested without Isaac Sim; the ray casts and the USD prim authoring live in ``pipe_rover/spray_marker.py``.

Random numbers come from NumPy's global generator by default (``np.random``) and are drawn in a fixed
order, see :func:`sample_cone_direction` and :func:`jitter_radius`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sample_cone_direction", "jitter_radius", "dot_center"]

_EPS = 1e-8


def sample_cone_direction(half_angle_rad: float, rng=np.random) -> np.ndarray:
    """Unit vector in the rover frame, inside a cone around +X (the rover's forward axis).

    The azimuth ``theta`` is uniform in ``[0, 2*pi)`` and the polar angle ``phi`` is uniform in
    ``[0, half_angle_rad]``; i.e. the sampling is uniform in *angle*, not in solid angle, so the
    density of rays is highest near the axis. Draw order: ``theta`` first, then ``phi``.
    """
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0, half_angle_rad)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    local_dir = np.array([
        cos_phi,
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
    ])
    return local_dir / max(np.linalg.norm(local_dir), _EPS)


def jitter_radius(base_radius: float, rng=np.random) -> float:
    """Dot radius with a +-30 % uniform jitter (one random draw)."""
    return base_radius * rng.uniform(0.7, 1.3)


def dot_center(hit_position, hit_normal, radius: float, min_offset: float = 0.003) -> np.ndarray:
    """Centre of a spray dot: the hit point pushed out along the surface normal.

    The offset is ``max(radius, min_offset)`` so that the sphere rests on the wall instead of
    sinking into it (this also keeps very small dots visible above the surface).
    """
    normal = np.asarray(hit_normal, dtype=float)
    normal = normal / max(np.linalg.norm(normal), _EPS)
    return np.asarray(hit_position, dtype=float) + normal * max(radius, min_offset)
