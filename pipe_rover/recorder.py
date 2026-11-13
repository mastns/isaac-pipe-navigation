# SPDX-License-Identifier: MIT
"""In-process ROS 2 bag recorder for the stereo + IMU + ground-truth streams.

The recorder writes a standard rosbag2 (``sqlite3`` storage, CDR serialisation) directly from the
simulation loop, using the ROS 2 Python bindings (``rosbag2_py``, ``rclpy``, ``sensor_msgs``,
``geometry_msgs``). All time stamps are *simulation time* in integer nanoseconds (the simulator passes
``step * dt``), starting at 0 when the simulator starts; header stamps and bag times are identical.

======================================  ===========================  ==========================
Topic                                   Type                         Content
======================================  ===========================  ==========================
``/stereo/front/{left,right}/image_raw``  ``sensor_msgs/msg/Image``   ``mono8``, frame ``stereo_front_{left,right}``
``/stereo/rear/{left,right}/image_raw``   ``sensor_msgs/msg/Image``   ``mono8``, frame ``stereo_rear_{left,right}``
``/imu0`` ``/imu1`` ``/imu2``           ``sensor_msgs/msg/Imu``      noisy accel + gyro, frame ``imu_<k>``;
                                                                     orientation unused (covariance[0] = -1)
``/ground_truth/pose``                  ``geometry_msgs/msg/PoseStamped``  rover root pose in frame ``world``
======================================  ===========================  ==========================

If the ROS 2 bindings cannot be imported the recorder disables itself with a warning and the
simulation keeps running.
"""

from __future__ import annotations

import array
import json
import os
from datetime import datetime

import numpy as np

from . import log

__all__ = ["Ros2StereoImuRecorder", "to_rgb_uint8", "rgb_to_mono8"]

IMU_KEYS = ("imu_0", "imu_1", "imu_2")

DEFAULT_TOPICS = {
    "left": "/stereo/front/left/image_raw",
    "right": "/stereo/front/right/image_raw",
    "rear_left": "/stereo/rear/left/image_raw",
    "rear_right": "/stereo/rear/right/image_raw",
    "imu_0": "/imu0",
    "imu_1": "/imu1",
    "imu_2": "/imu2",
    "ground_truth": "/ground_truth/pose",
}


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Camera buffer (HxW, HxWx3 or HxWx4; uint8 or float) -> contiguous HxWx3 uint8."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        max_val = float(np.max(image)) if image.size > 0 else 1.0
        if max_val <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def rgb_to_mono8(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma of an HxWx3 uint8 image, as uint8 (the ``mono8`` the bag stores)."""
    return (0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)).clip(0, 255).astype(np.uint8)


class Ros2StereoImuRecorder:
    """Writes the simulated sensor streams to a rosbag2 directory.

    Parameters
    ----------
    enabled:
        ``False`` turns the recorder into a no-op.
    bag_dir:
        Directory that receives one sub-folder ``<bag_prefix>_<YYYYmmdd_HHMMSS>`` per recording.
    bag_prefix:
        Prefix of the bag folder name.
    """

    def __init__(self, enabled: bool = True, bag_dir: str = "rosbags", bag_prefix: str = "stereo_imus"):
        self.enabled = bool(enabled)
        self.bag_dir = bag_dir
        self.bag_prefix = bag_prefix
        self.recording = False
        self.toggle_requested = False
        self._writer = None
        self._current_bag_path: str | None = None
        self._topics = dict(DEFAULT_TOPICS)
        self.img_accum = 0.0
        self._msg_count = 0
        self.run_info: dict | None = None   # written next to every bag as sim_config.json when set
        self._warned_disabled = False

        if not self.enabled:
            log.info("ROS2 recording disabled.")
            return

        try:
            import rosbag2_py
            from geometry_msgs.msg import PoseStamped
            from rclpy.serialization import serialize_message
            from sensor_msgs.msg import Image, Imu
        except Exception as err:  # ImportError, or a broken ROS install (missing shared libraries)
            log.warn(f"ROS2 bag recording disabled (import failed): {err}")
            log.warn("Recording needs the ROS 2 Python bindings (rosbag2_py, rclpy, sensor_msgs, geometry_msgs) "
                     "built for the Python version of Isaac Sim (3.11); source that ROS 2 install before "
                     "starting the simulator. The simulation itself runs without them.")
            self.enabled = False
            return

        self._rosbag2_py = rosbag2_py
        self._serialize = serialize_message
        self._Image = Image
        self._Imu = Imu
        self._PoseStamped = PoseStamped
        log.info("ROS2 bag recorder ready (rosbag2_py in-process writer).")

    # ------------------------------------------------------------------ #
    # Recording state
    # ------------------------------------------------------------------ #

    @property
    def bag_path(self) -> str | None:
        """Path of the bag being written (or of the last one written)."""
        return self._current_bag_path

    @property
    def message_count(self) -> int:
        return self._msg_count

    def request_toggle(self) -> None:
        """Ask for start/stop at the next :meth:`process_toggle_request` (safe to call from a key callback)."""
        if not self.enabled and not self._warned_disabled:
            self._warned_disabled = True
            log.warn("Recording was requested but the recorder is disabled (see the earlier message).")
        self.toggle_requested = True

    def process_toggle_request(self) -> None:
        if not self.toggle_requested:
            return
        self.toggle_requested = False
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if not self.enabled or self.recording:
            return

        bag_root = os.path.abspath(self.bag_dir)
        stem = f"{self.bag_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            os.makedirs(bag_root, exist_ok=True)
            bag_path = os.path.join(bag_root, stem)
            suffix = 1
            while os.path.exists(bag_path):         # two recordings within the same second
                bag_path = os.path.join(bag_root, f"{stem}_{suffix}")
                suffix += 1
            self._current_bag_path = bag_path

            rb = self._rosbag2_py
            storage_options = rb.StorageOptions(uri=bag_path, storage_id="sqlite3")
            converter_options = rb.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr")
            self._writer = rb.SequentialWriter()
            self._writer.open(storage_options, converter_options)

            for key in ("left", "right", "rear_left", "rear_right"):
                self._writer.create_topic(rb.TopicMetadata(
                    name=self._topics[key], type="sensor_msgs/msg/Image", serialization_format="cdr"))
            for key in IMU_KEYS:
                self._writer.create_topic(rb.TopicMetadata(
                    name=self._topics[key], type="sensor_msgs/msg/Imu", serialization_format="cdr"))
            self._writer.create_topic(rb.TopicMetadata(
                name=self._topics["ground_truth"], type="geometry_msgs/msg/PoseStamped",
                serialization_format="cdr"))
        except Exception as err:
            log.error(f"Failed to open rosbag2 writer: {err}")
            self._writer = None
            return

        self.recording = True
        self.img_accum = 0.0
        self._msg_count = 0
        self._write_run_info(bag_path)
        log.info(f"ROS2 bag recording STARTED -> {bag_path}")

    def _write_run_info(self, bag_path: str) -> None:
        """Provenance sidecar (script version, arguments, sensor parameters) next to the bag."""
        if not self.run_info:
            return
        try:
            with open(os.path.join(bag_path, "sim_config.json"), "w", encoding="utf-8") as f:
                json.dump(self.run_info, f, indent=2, default=str)
        except OSError as err:
            log.warn(f"Could not write sim_config.json: {err}")

    def stop_recording(self) -> None:
        if not self.recording:
            return
        writer = self._writer
        self._writer = None
        self.recording = False
        if writer is not None:
            # rosbag2 finalises the bag (metadata.yaml) when the writer is closed or destroyed
            close = getattr(writer, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as err:
                    log.warn(f"Closing the rosbag2 writer raised: {err}")
            del writer
        log.info(f"ROS2 bag recording stopped. {self._msg_count} messages written.")
        self._report_bag(self._current_bag_path)

    def shutdown(self) -> None:
        self.stop_recording()

    @staticmethod
    def _report_bag(bag_path: str | None) -> None:
        if not bag_path or not os.path.isdir(bag_path):
            log.warn(f"Bag directory NOT found at: {bag_path}")
            return
        contents = sorted(os.listdir(bag_path))
        log.info(f"Bag saved to: {bag_path}")
        has_metadata = any(f.endswith(".yaml") for f in contents)
        has_data = any(f.endswith((".db3", ".mcap")) for f in contents)
        if not (has_metadata and has_data):
            log.warn(f"Bag directory exists but is missing expected files. Found: {contents}")
            return
        for name in contents:
            size = os.path.getsize(os.path.join(bag_path, name))
            if size >= 1024 * 1024:
                log.info(f"  {name}  ({size / (1024 * 1024):.1f} MB)")
            elif size >= 1024:
                log.info(f"  {name}  ({size / 1024:.1f} KB)")
            else:
                log.info(f"  {name}  ({size} B)")
        log.info(f"Bag looks valid. To inspect:  ros2 bag info {bag_path}")

    # ------------------------------------------------------------------ #
    # Timing helpers
    # ------------------------------------------------------------------ #

    def image_due(self, dt: float, period: float) -> bool:
        """Advance the image accumulator by ``dt``; return True (and reset it) once ``period`` has elapsed.

        The accumulator is *reset to zero*, not decremented, so with ``period`` not a multiple of ``dt``
        the effective period is ``ceil(period / dt) * dt`` (e.g. 0.033 s with 5 ms steps -> 35 ms; a period
        that is an exact multiple of ``dt`` fires on that step).
        """
        self.img_accum += dt
        if self.img_accum >= period - 1e-9:      # tolerance: ten additions of 0.005 are 0.04999999999999999
            self.img_accum = 0.0
            return True
        return False

    @staticmethod
    def _stamp(t_ns: int):
        """Split an integer nanosecond time into the ``(sec, nanosec)`` of a ROS header stamp."""
        return divmod(int(t_ns), 1_000_000_000)

    def _active(self) -> bool:
        return self.enabled and self.recording and self._writer is not None

    # ------------------------------------------------------------------ #
    # Publishers
    # ------------------------------------------------------------------ #

    def publish_imus(self, imu_dict, t_ns: int) -> None:
        """``imu_dict[name] = (linear_acceleration[3], angular_velocity[3])`` for ``imu_0..imu_2``."""
        if not self._active():
            return
        sec, nanosec = self._stamp(t_ns)
        for imu_name in IMU_KEYS:
            lin_acc, ang_vel = imu_dict[imu_name]
            msg = self._Imu()
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
            msg.header.frame_id = imu_name
            msg.orientation.w = 1.0
            msg.orientation_covariance[0] = -1.0
            msg.angular_velocity.x = float(ang_vel[0])
            msg.angular_velocity.y = float(ang_vel[1])
            msg.angular_velocity.z = float(ang_vel[2])
            msg.linear_acceleration.x = float(lin_acc[0])
            msg.linear_acceleration.y = float(lin_acc[1])
            msg.linear_acceleration.z = float(lin_acc[2])
            self._writer.write(self._topics[imu_name], self._serialize(msg), int(t_ns))
            self._msg_count += 1

    def publish_images(self, left_rgb, right_rgb, t_ns: int,
                       rear_left_rgb=None, rear_right_rgb=None) -> None:
        """Write the stereo images as ``mono8`` (BT.601 luma of the rendered RGB)."""
        if not self._active():
            return
        sec, nanosec = self._stamp(t_ns)
        pairs = [
            ("left", "stereo_front_left", left_rgb),
            ("right", "stereo_front_right", right_rgb),
        ]
        if rear_left_rgb is not None:
            pairs.append(("rear_left", "stereo_rear_left", rear_left_rgb))
        if rear_right_rgb is not None:
            pairs.append(("rear_right", "stereo_rear_right", rear_right_rgb))
        for topic_key, frame_id, img in pairs:
            gray = rgb_to_mono8(to_rgb_uint8(img))
            msg = self._Image()
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
            msg.header.frame_id = frame_id
            msg.height = int(gray.shape[0])
            msg.width = int(gray.shape[1])
            msg.encoding = "mono8"
            msg.is_bigendian = 0
            msg.step = int(gray.shape[1])
            msg.data = array.array("B", gray.tobytes())   # ~600x faster than assigning bytes
            self._writer.write(self._topics[topic_key], self._serialize(msg), int(t_ns))
            self._msg_count += 1

    def publish_ground_truth(self, pos, quat_wxyz, t_ns: int) -> None:
        """Rover root pose in the world frame. ``quat_wxyz`` is Isaac's (w, x, y, z); ROS stores x, y, z, w."""
        if not self._active():
            return
        sec, nanosec = self._stamp(t_ns)
        msg = self._PoseStamped()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = "world"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = float(quat_wxyz[0])
        msg.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.orientation.z = float(quat_wxyz[3])
        self._writer.write(self._topics["ground_truth"], self._serialize(msg), int(t_ns))
        self._msg_count += 1
