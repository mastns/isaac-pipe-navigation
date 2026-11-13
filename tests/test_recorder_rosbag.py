# SPDX-License-Identifier: MIT
"""Round trip of the ROS 2 bag recorder: write a synthetic session, read it back with rosbag2_py.

Needs the ROS 2 Python bindings (rosbag2_py, rclpy, sensor_msgs, geometry_msgs); skipped without them.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pipe_rover.recorder import DEFAULT_TOPICS, Ros2StereoImuRecorder

rosbag2_py = pytest.importorskip("rosbag2_py", exc_type=ImportError)
pytest.importorskip("rclpy", exc_type=ImportError)
pytest.importorskip("sensor_msgs", exc_type=ImportError)
pytest.importorskip("geometry_msgs", exc_type=ImportError)
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402


def _read_bag(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    out = {name: [] for name in types}
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        out[topic].append((stamp_ns, deserialize_message(data, get_message(types[topic]))))
    return types, out


def test_recorded_bag_round_trip(tmp_path):
    rec = Ros2StereoImuRecorder(enabled=True, bag_dir=str(tmp_path), bag_prefix="unit")
    assert rec.enabled, "ROS bindings imported but the recorder disabled itself"
    rec.run_info = {"script": "unit-test", "args": {"seed": 1}}
    rec.request_toggle()
    rec.process_toggle_request()
    assert rec.recording and rec.bag_path.startswith(str(tmp_path))

    dt_ns, h, w = 5_000_000, 6, 8
    n_steps, image_every = 20, 7
    for k in range(n_steps):
        t = k * dt_ns
        imu = {f"imu_{i}": ([0.1 * i, 0.2, 9.81 + k], [0.01 * k, 0.0, -0.3]) for i in range(3)}
        rec.publish_imus(imu, t)
        rec.publish_ground_truth([k, 2.0, 0.5], [0.5, 0.1, 0.2, 0.3], t)   # Isaac order: w, x, y, z
        if k % image_every == 0:
            img = np.zeros((h, w, 4), np.uint8)
            img[..., 0] = 255                                                # pure red, alpha ignored
            rec.publish_images(img, img, t, rear_left_rgb=img, rear_right_rgb=img)
    rec.shutdown()
    assert not rec.recording and rec.message_count == n_steps * 4 + 3 * 4 * 1

    bag_files = os.listdir(rec.bag_path)
    assert "metadata.yaml" in bag_files and any(f.endswith(".db3") for f in bag_files)
    assert json.loads(Path(rec.bag_path, "sim_config.json").read_text())["args"] == {"seed": 1}

    types, msgs = _read_bag(rec.bag_path)
    assert set(types) == set(DEFAULT_TOPICS.values())
    assert types["/imu1"] == "sensor_msgs/msg/Imu"
    assert types["/ground_truth/pose"] == "geometry_msgs/msg/PoseStamped"
    assert all(types[t] == "sensor_msgs/msg/Image" for t in types if t.startswith("/stereo/"))

    assert [len(msgs[f"/imu{i}"]) for i in range(3)] == [n_steps] * 3
    assert len(msgs["/ground_truth/pose"]) == n_steps
    assert [len(msgs[t]) for t in types if t.startswith("/stereo/")] == [3] * 4   # steps 0, 7, 14

    # time stamps: simulation time, bag time == header time
    stamp_ns, imu = msgs["/imu2"][3]
    assert stamp_ns == 3 * dt_ns
    assert imu.header.stamp.sec * 10**9 + imu.header.stamp.nanosec == stamp_ns
    assert imu.header.frame_id == "imu_2"
    assert imu.linear_acceleration.z == pytest.approx(9.81 + 3)
    assert imu.angular_velocity.x == pytest.approx(0.03) and imu.angular_velocity.z == pytest.approx(-0.3)
    assert imu.orientation_covariance[0] == -1.0

    # ground truth: Isaac (w, x, y, z) -> ROS (x, y, z, w) fields
    _, gt = msgs["/ground_truth/pose"][5]
    assert gt.header.frame_id == "world"
    assert (gt.pose.position.x, gt.pose.position.y, gt.pose.position.z) == (5.0, 2.0, 0.5)
    q = gt.pose.orientation
    assert (q.w, q.x, q.y, q.z) == (0.5, 0.1, 0.2, 0.3)

    # images: mono8, BT.601 luma of pure red = 76
    _, im = msgs["/stereo/rear/right/image_raw"][1]
    assert (im.encoding, im.height, im.width, im.step, im.is_bigendian) == ("mono8", h, w, w, 0)
    assert im.header.frame_id == "stereo_rear_right"
    assert set(bytes(im.data)) == {76} and len(im.data) == h * w
    assert msgs["/stereo/front/left/image_raw"][1][0] == 7 * dt_ns


def test_stopping_twice_is_harmless_and_leaves_a_bag(tmp_path):
    rec = Ros2StereoImuRecorder(enabled=True, bag_dir=str(tmp_path), bag_prefix="unit")
    rec.start_recording()
    first = rec.bag_path
    rec.stop_recording()
    assert not rec.recording
    rec.stop_recording()                      # stopping twice is harmless
    assert os.path.isdir(first)


def test_two_recordings_in_the_same_second_get_distinct_bags(tmp_path):
    rec = Ros2StereoImuRecorder(enabled=True, bag_dir=str(tmp_path), bag_prefix="unit")
    rec.start_recording()
    first = rec.bag_path
    rec.stop_recording()
    rec.start_recording()
    second = rec.bag_path
    rec.stop_recording()
    assert first != second and os.path.isdir(first) and os.path.isdir(second)


def test_image_payload_serialises_like_plain_bytes():
    """array.array('B') is only a faster way to fill Image.data: the serialised message must not change."""
    import array

    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image

    gray = (np.arange(24, dtype=np.uint8) * 9).reshape(4, 6)
    a, b = Image(), Image()
    for msg in (a, b):
        msg.height, msg.width, msg.encoding, msg.step = 4, 6, "mono8", 6
    a.data = gray.tobytes()
    b.data = array.array("B", gray.tobytes())
    ra, rb = serialize_message(a), serialize_message(b)
    assert len(ra) == len(rb)
    # CDR padding bytes are not initialised, so compare the decoded payloads
    assert bytes(deserialize_message(ra, Image).data) == bytes(deserialize_message(rb, Image).data) == gray.tobytes()
