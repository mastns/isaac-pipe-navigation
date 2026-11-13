# SPDX-License-Identifier: MIT
"""Additive noise and bias model applied to the simulated IMU streams.

The model is intentionally simple and is applied once per physics step to every IMU
(200 Hz with the default 5 ms physics step):

``measured = true + bias + white_noise``  with  ``bias += N(0, bias_std)``  on every call.

* ``*_noise_std`` is the standard deviation of the white noise **per sample** (not a
  continuous-time density) - accelerometer in m/s^2, gyroscope in rad/s;
* ``*_bias_std`` is the standard deviation of the **per-sample increment** of a random-walk
  bias, so the bias standard deviation grows like ``bias_std * sqrt(n_samples)``.

With the defaults of the simulator (accel 0.05 / bias 0.005 m/s^2, gyro 0.005 / bias 0.001 rad/s at
200 Hz) the equivalent continuous-time densities are ``std * sqrt(dt)`` for the white noise and
``bias_std / sqrt(dt)`` for the random walk; see :meth:`ImuNoiseModel.continuous_time_densities`.

Random numbers come from NumPy's global generator by default (``np.random``), so
``np.random.seed(seed)`` makes a run reproducible; the draw order per call is
accel bias, gyro bias, accel noise, gyro noise.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

__all__ = ["ImuNoiseModel"]


class ImuNoiseModel:
    """Noise and random-walk bias for a set of named IMUs.

    Parameters
    ----------
    names:
        Identifiers of the IMUs, e.g. ``("imu_0", "imu_1", "imu_2")``.
    accel_noise_std, gyro_noise_std:
        White-noise standard deviation per sample (m/s^2 and rad/s).
    accel_bias_std, gyro_bias_std:
        Standard deviation of the per-sample bias random-walk increment (m/s^2 and rad/s).
    rng:
        Object exposing ``randn(n)`` (default: the global ``numpy.random`` module).
    """

    def __init__(self, names: Iterable[str], accel_noise_std: float, gyro_noise_std: float,
                 accel_bias_std: float, gyro_bias_std: float, rng=np.random):
        self.accel_noise_std = float(accel_noise_std)
        self.gyro_noise_std = float(gyro_noise_std)
        self.accel_bias_std = float(accel_bias_std)
        self.gyro_bias_std = float(gyro_bias_std)
        self._rng = rng
        self.bias: dict[str, dict[str, np.ndarray]] = {
            name: {"accel": np.zeros(3), "gyro": np.zeros(3)} for name in names
        }

    def reset(self) -> None:
        """Set every bias back to zero."""
        for state in self.bias.values():
            state["accel"][:] = 0.0
            state["gyro"][:] = 0.0

    def apply(self, name: str, accel, gyro) -> tuple[np.ndarray, np.ndarray]:
        """Return the noisy ``(accel, gyro)`` pair of IMU ``name`` and advance its bias random walk."""
        state = self.bias[name]
        state["accel"] += self._rng.randn(3) * self.accel_bias_std
        state["gyro"] += self._rng.randn(3) * self.gyro_bias_std
        accel_noisy = np.asarray(accel) + state["accel"] + self._rng.randn(3) * self.accel_noise_std
        gyro_noisy = np.asarray(gyro) + state["gyro"] + self._rng.randn(3) * self.gyro_noise_std
        return accel_noisy, gyro_noisy

    def continuous_time_densities(self, dt: float) -> dict[str, float]:
        """Continuous-time equivalents of the per-sample parameters for a sample period ``dt`` (s).

        Returns the white-noise densities (m/s^2/sqrt(Hz), rad/s/sqrt(Hz)) and the bias random-walk
        densities (m/s^3/sqrt(Hz), rad/s^2/sqrt(Hz)) as used by Kalibr / VIO configuration files.
        """
        root_dt = math.sqrt(dt)
        return {
            "accel_noise_density": self.accel_noise_std * root_dt,
            "gyro_noise_density": self.gyro_noise_std * root_dt,
            "accel_random_walk": self.accel_bias_std / root_dt,
            "gyro_random_walk": self.gyro_bias_std / root_dt,
        }
