# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from pipe_rover import spray


def _legacy_direction():
    """The inline sampling code of the simulator, kept here as the reference."""
    half = np.radians(70.0)
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, half)
    d = np.array([np.cos(phi), np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta)])
    return d / max(np.linalg.norm(d), 1e-8)


def test_direction_sampling_matches_the_original_code_and_random_stream():
    np.random.seed(11)
    expected = [(_legacy_direction(), np.random.uniform(0.7, 1.3)) for _ in range(200)]
    np.random.seed(11)
    got = [(spray.sample_cone_direction(np.radians(70.0)), spray.jitter_radius(1.0)) for _ in range(200)]
    for (ed, er), (gd, gr) in zip(expected, got, strict=True):
        assert np.array_equal(ed, gd)
        assert er == gr


@pytest.mark.parametrize("half_angle_deg", [5.0, 35.0, 70.0, 89.0])
def test_directions_are_unit_vectors_inside_the_cone(half_angle_deg):
    rng = np.random.RandomState(0)
    half = np.radians(half_angle_deg)
    d = np.array([spray.sample_cone_direction(half, rng) for _ in range(2000)])
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.all(d[:, 0] >= np.cos(half) - 1e-12)  # forward component bounded by the cone angle


def test_sampling_is_uniform_in_polar_angle():
    rng = np.random.RandomState(1)
    half = np.radians(60.0)
    phi = np.arccos([spray.sample_cone_direction(half, rng)[0] for _ in range(20000)])
    assert phi.mean() == pytest.approx(half / 2, rel=0.02)


def test_jitter_radius_range():
    rng = np.random.RandomState(2)
    r = np.array([spray.jitter_radius(0.006, rng) for _ in range(1000)])
    assert r.min() >= 0.7 * 0.006 and r.max() <= 1.3 * 0.006


def test_dot_center_sits_on_the_wall_side_of_the_normal():
    c = spray.dot_center((1.0, 2.0, 3.0), (0.0, 0.0, 2.0), radius=0.006)
    assert np.allclose(c, (1.0, 2.0, 3.006))
    tiny = spray.dot_center((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), radius=0.001)
    assert np.allclose(tiny, (0.003, 0.0, 0.0))  # minimum offset keeps small dots above the surface
