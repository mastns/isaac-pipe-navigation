# SPDX-License-Identifier: MIT
"""Stricter bag round trip (needs ROS 2): every field gets a distinct value, so a swapped axis, camera or topic shows.

Complements test_recorder_rosbag.py, whose IMU values (accel x == y for imu_2, gyro y == 0) and identical red pictures
cannot reveal such swaps.  Also reads the real rosbag2 file with scripts/check_bag.py."""

from pathlib import Path

import check_bag
import numpy as np
import pytest

from pipe_rover.recorder import DEFAULT_TOPICS, Ros2StereoImuRecorder, rgb_to_mono8

rosbag2_py = pytest.importorskip("rosbag2_py", exc_type=ImportError)
pytest.importorskip("rclpy", exc_type=ImportError)
pytest.importorskip("sensor_msgs", exc_type=ImportError)
pytest.importorskip("geometry_msgs", exc_type=ImportError)
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402


def _read_bag(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    out = {name: [] for name in types}
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        out[topic].append((stamp_ns, deserialize_message(data, get_message(types[topic]))))
    return out


def test_every_field_lands_where_it_belongs_and_check_bag_can_read_it(tmp_path):
    rec = Ros2StereoImuRecorder(enabled=True, bag_dir=str(tmp_path), bag_prefix="unit")
    rec.start_recording()
    shades = {"left": 50, "right": 100, "rear_left": 150, "rear_right": 200}
    pics = {k: np.zeros((2, 3, 3), np.uint8) for k in shades}
    for k, v in shades.items():
        pics[k][..., 0] = v
    steps = [7_000_000_013 + k * 5_000_000 for k in range(3)]            # sec = 7, nanosec = 13 + k * 5 ms
    for k, t in enumerate(steps):
        imus = {f"imu_{i}": ([1.0 + i, 2.0 + i, 3.0 + i + k], [4.0 + i, 5.0 + i, 6.0 + i]) for i in range(3)}
        rec.publish_imus(imus, t)
        rec.publish_ground_truth([1.0, 2.0, 3.0 + k], [0.5, 0.1, 0.2, 0.3], t)
        if k == 1:
            rec.publish_images(pics["left"], pics["right"], t,
                               rear_left_rgb=pics["rear_left"], rear_right_rgb=pics["rear_right"])
    rec.shutdown()
    msgs = _read_bag(rec.bag_path)

    for stamps in msgs.values():                                          # bag time == header time, for every type
        for bag_ns, m in stamps:
            assert bag_ns == m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
    assert [ns for ns, _ in msgs["/imu0"]] == steps

    for i in range(3):
        _, m = msgs[f"/imu{i}"][2]
        a, w = m.linear_acceleration, m.angular_velocity
        assert m.header.frame_id == f"imu_{i}"
        assert (a.x, a.y, a.z) == (1.0 + i, 2.0 + i, 5.0 + i)
        assert (w.x, w.y, w.z) == (4.0 + i, 5.0 + i, 6.0 + i)

    _, gt = msgs["/ground_truth/pose"][2]
    q = gt.pose.orientation
    assert (gt.pose.position.x, gt.pose.position.y, gt.pose.position.z) == (1.0, 2.0, 5.0)
    assert (q.x, q.y, q.z, q.w) == (0.1, 0.2, 0.3, 0.5)

    for key, frame in (("left", "stereo_front_left"), ("right", "stereo_front_right"),
                       ("rear_left", "stereo_rear_left"), ("rear_right", "stereo_rear_right")):
        (_, m), = msgs[DEFAULT_TOPICS[key]]
        assert m.header.frame_id == frame and bytes(m.data) == rgb_to_mono8(pics[key]).tobytes()

    stats = check_bag.summarise(Path(rec.bag_path))
    expected = {f"/imu{i}": 3 for i in range(3)} | {"/ground_truth/pose": 3} | {DEFAULT_TOPICS[k]: 1 for k in shades}
    assert {t: s["count"] for t, s in stats.items()} == expected
    assert stats["/imu1"]["rate_hz"] == pytest.approx(200.0)
