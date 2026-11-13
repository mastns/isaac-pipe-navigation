# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from pipe_rover.imu_noise import ImuNoiseModel

NAMES = ("imu_0", "imu_1", "imu_2")


def _reference(seed, steps, params):
    """The noise loop exactly as it was written inline in the simulator before it became a class."""
    accel_noise, gyro_noise, accel_bias_std, gyro_bias_std = params
    bias = {n: {"accel": np.zeros(3), "gyro": np.zeros(3)} for n in NAMES}
    np.random.seed(seed)
    out = []
    for step in range(steps):
        for n in NAMES:
            accel = np.array([0.1 * step, 0.0, 9.81])
            gyro = np.array([0.0, 0.01 * step, 0.0])
            bias[n]["accel"] += np.random.randn(3) * accel_bias_std
            bias[n]["gyro"] += np.random.randn(3) * gyro_bias_std
            a = accel + bias[n]["accel"] + np.random.randn(3) * accel_noise
            g = gyro + bias[n]["gyro"] + np.random.randn(3) * gyro_noise
            out.append((a, g))
    return out


def test_matches_the_original_inline_loop_bit_for_bit():
    params = (0.05, 0.005, 0.005, 0.001)
    expected = _reference(seed=3, steps=50, params=params)
    model = ImuNoiseModel(NAMES, *params)
    np.random.seed(3)
    got = []
    for step in range(50):
        for n in NAMES:
            got.append(model.apply(n, [0.1 * step, 0.0, 9.81], [0.0, 0.01 * step, 0.0]))
    for (ea, eg), (ga, gg) in zip(expected, got, strict=True):
        assert np.array_equal(ea, ga) and np.array_equal(eg, gg)


def test_white_noise_and_bias_random_walk_statistics():
    rng = np.random.RandomState(0)
    n_imu, steps = 1000, 400
    names = [f"i{k}" for k in range(n_imu)]
    model = ImuNoiseModel(names, accel_noise_std=0.05, gyro_noise_std=0.005,
                          accel_bias_std=0.005, gyro_bias_std=0.001, rng=rng)
    last = None
    for _ in range(steps):
        last = [model.apply(n, np.zeros(3), np.zeros(3)) for n in names]
    # bias std after `steps` samples is bias_std * sqrt(steps)
    accel_bias = np.array([model.bias[n]["accel"] for n in names])
    gyro_bias = np.array([model.bias[n]["gyro"] for n in names])
    assert accel_bias.std() == pytest.approx(0.005 * np.sqrt(steps), rel=0.05)
    assert gyro_bias.std() == pytest.approx(0.001 * np.sqrt(steps), rel=0.05)
    # residual of the last sample around its bias is the white noise
    resid = np.array([a - model.bias[n]["accel"] for n, (a, _) in zip(names, last, strict=True)])
    assert resid.std() == pytest.approx(0.05, rel=0.05)


def test_reset_clears_the_bias():
    model = ImuNoiseModel(NAMES, 0.0, 0.0, 1.0, 1.0)
    model.apply("imu_0", np.zeros(3), np.zeros(3))
    assert np.any(model.bias["imu_0"]["accel"] != 0)
    model.reset()
    assert all(not np.any(v) for state in model.bias.values() for v in state.values())


def test_continuous_time_densities_at_200_hz():
    d = ImuNoiseModel(NAMES, 0.05, 0.005, 0.005, 0.001).continuous_time_densities(0.005)
    assert d["accel_noise_density"] == pytest.approx(0.05 * np.sqrt(0.005))
    assert d["gyro_noise_density"] == pytest.approx(0.005 * np.sqrt(0.005))
    assert d["accel_random_walk"] == pytest.approx(0.005 / np.sqrt(0.005))
    assert d["gyro_random_walk"] == pytest.approx(0.001 / np.sqrt(0.005))
