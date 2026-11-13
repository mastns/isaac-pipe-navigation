# SPDX-License-Identifier: MIT
"""Recorder behaviour that CI can check without ROS 2: rosbag2_py / rclpy / sensor_msgs / geometry_msgs are replaced by
tiny *strict* fakes (only the real field names exist, so a typo raises), and ``serialize_message`` is the identity,
so the fake writer stores the message objects that the recorder built."""

import json
import os
import sys
import types
from datetime import datetime as real_datetime

import numpy as np
import pytest

from pipe_rover import recorder
from pipe_rover.recorder import DEFAULT_TOPICS, Ros2StereoImuRecorder, rgb_to_mono8


class _Strict:
    __slots__ = ()

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name, self._defaults().get(name, 0.0)))

    @classmethod
    def _defaults(cls):
        return {}


class Stamp(_Strict):
    __slots__ = ("sec", "nanosec")


class Vec3(_Strict):
    __slots__ = ("x", "y", "z")


class Quat(_Strict):
    __slots__ = ("x", "y", "z", "w")

    @classmethod
    def _defaults(cls):
        return {"w": 1.0}      # as in geometry_msgs/Quaternion.msg


class Header(_Strict):
    __slots__ = ("stamp", "frame_id")

    def __init__(self):
        self.stamp, self.frame_id = Stamp(), ""


class Imu(_Strict):
    __slots__ = ("header", "orientation", "orientation_covariance", "angular_velocity", "angular_velocity_covariance",
                 "linear_acceleration", "linear_acceleration_covariance")

    def __init__(self):
        self.header, self.orientation = Header(), Quat()
        self.orientation_covariance = [0.0] * 9
        self.angular_velocity_covariance = [0.0] * 9
        self.linear_acceleration_covariance = [0.0] * 9
        self.angular_velocity, self.linear_acceleration = Vec3(), Vec3()


class Image(_Strict):
    __slots__ = ("header", "height", "width", "encoding", "is_bigendian", "step", "data")

    def __init__(self):
        self.header, self.height, self.width, self.encoding = Header(), 0, 0, ""
        self.is_bigendian, self.step, self.data = 0, 0, b""


class Pose(_Strict):
    __slots__ = ("position", "orientation")

    def __init__(self):
        self.position, self.orientation = Vec3(), Quat()


class PoseStamped(_Strict):
    __slots__ = ("header", "pose")

    def __init__(self):
        self.header, self.pose = Header(), Pose()


class FakeWriter:
    """Stands in for rosbag2_py.SequentialWriter: like the real one it refuses to open an existing directory."""

    instances = []

    def __init__(self):
        self.uri, self.topics, self.records, self.closed = None, {}, [], False
        FakeWriter.instances.append(self)

    def open(self, storage_options, converter_options):
        if os.path.exists(storage_options.uri):
            raise RuntimeError(f"bag directory already exists: {storage_options.uri}")
        os.makedirs(storage_options.uri)
        self.uri = storage_options.uri

    def create_topic(self, meta):
        self.topics[meta.name] = meta.type

    def write(self, topic, data, t_ns):
        assert topic in self.topics, f"topic {topic} was never created"
        self.records.append((topic, data, t_ns))

    def close(self):
        self.closed = True


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture
def fake_ros(monkeypatch):
    FakeWriter.instances.clear()
    rosbag2_py = _module(
        "rosbag2_py", SequentialWriter=FakeWriter,
        StorageOptions=lambda uri, storage_id: types.SimpleNamespace(uri=uri, storage_id=storage_id),
        ConverterOptions=lambda **kw: types.SimpleNamespace(**kw),
        TopicMetadata=lambda name, type, serialization_format: types.SimpleNamespace(name=name, type=type))
    modules = {
        "rosbag2_py": rosbag2_py,
        "rclpy": _module("rclpy"),
        "rclpy.serialization": _module("rclpy.serialization", serialize_message=lambda msg: msg),
        "sensor_msgs": _module("sensor_msgs"),
        "sensor_msgs.msg": _module("sensor_msgs.msg", Image=Image, Imu=Imu),
        "geometry_msgs": _module("geometry_msgs"),
        "geometry_msgs.msg": _module("geometry_msgs.msg", PoseStamped=PoseStamped),
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return FakeWriter


def _recording(tmp_path):
    rec = Ros2StereoImuRecorder(enabled=True, bag_dir=str(tmp_path), bag_prefix="unit")
    assert rec.enabled
    rec.start_recording()
    assert rec.recording
    return rec, FakeWriter.instances[-1]


def _by_topic(writer):
    out = {}
    for topic, msg, t_ns in writer.records:
        out.setdefault(topic, []).append((msg, t_ns))
    return out


def _stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def test_ground_truth_converts_wxyz_to_the_ros_quaternion_fields(fake_ros, tmp_path):
    rec, writer = _recording(tmp_path)
    rec.publish_ground_truth([1.0, 2.0, 3.0], [0.5, 0.1, 0.2, 0.3], 12_345_678_901)      # Isaac order: w, x, y, z
    (msg, t_ns), = _by_topic(writer)["/ground_truth/pose"]
    stamp = msg.header.stamp
    assert (msg.header.frame_id, stamp.sec, stamp.nanosec, t_ns) == ("world", 12, 345_678_901, 12_345_678_901)
    p, q = msg.pose.position, msg.pose.orientation
    assert (p.x, p.y, p.z) == (1.0, 2.0, 3.0)
    assert (q.x, q.y, q.z, q.w) == (0.1, 0.2, 0.3, 0.5)


def test_every_imu_has_its_own_topic_frame_and_axis_mapping(fake_ros, tmp_path):
    rec, writer = _recording(tmp_path)
    imus = {f"imu_{i}": ([1.0 + i, 2.0 + i, 3.0 + i], [4.0 + i, 5.0 + i, 6.0 + i]) for i in range(3)}
    rec.publish_imus(imus, 3_000_000_007)
    by_topic = _by_topic(writer)
    for i in range(3):
        (msg, t_ns), = by_topic[f"/imu{i}"]
        assert msg.header.frame_id == f"imu_{i}"
        a, w = msg.linear_acceleration, msg.angular_velocity
        assert (a.x, a.y, a.z) == (1.0 + i, 2.0 + i, 3.0 + i)
        assert (w.x, w.y, w.z) == (4.0 + i, 5.0 + i, 6.0 + i)
        assert msg.orientation_covariance[0] == -1.0
        assert (msg.header.stamp.sec, msg.header.stamp.nanosec, t_ns) == (3, 7, 3_000_000_007)


def test_every_camera_goes_to_its_own_topic_and_frame(fake_ros, tmp_path):
    rec, writer = _recording(tmp_path)
    shades = {"left": 50, "right": 100, "rear_left": 150, "rear_right": 200}
    pics = {k: np.zeros((2, 3, 3), np.uint8) for k in shades}
    for k, v in shades.items():
        pics[k][..., 0] = v
    rec.publish_images(pics["left"], pics["right"], 35_000_000,
                       rear_left_rgb=pics["rear_left"], rear_right_rgb=pics["rear_right"])
    by_topic = _by_topic(writer)
    for key, frame in (("left", "stereo_front_left"), ("right", "stereo_front_right"),
                       ("rear_left", "stereo_rear_left"), ("rear_right", "stereo_rear_right")):
        (msg, t_ns), = by_topic[DEFAULT_TOPICS[key]]
        assert msg.header.frame_id == frame
        assert bytes(msg.data) == rgb_to_mono8(pics[key]).tobytes(), f"{key} carries the wrong picture"
        assert (msg.encoding, msg.height, msg.width, msg.step, t_ns) == ("mono8", 2, 3, 3, 35_000_000)
        assert _stamp_ns(msg) == t_ns


@pytest.mark.parametrize("t_ns", [0, 5_000_000, 999_999_999, 3_000_000_007, 12_345_678_901])
def test_header_stamp_equals_bag_time_for_every_message_type(fake_ros, tmp_path, t_ns):
    rec, writer = _recording(tmp_path)
    img = np.zeros((2, 2, 3), np.uint8)
    rec.publish_imus({f"imu_{i}": ([0, 0, 0], [0, 0, 0]) for i in range(3)}, t_ns)
    rec.publish_ground_truth([0, 0, 0], [1, 0, 0, 0], t_ns)
    rec.publish_images(img, img, t_ns, rear_left_rgb=img, rear_right_rgb=img)
    assert len(writer.records) == 3 + 1 + 4 == rec.message_count
    for _, msg, bag_time in writer.records:
        assert bag_time == t_ns and _stamp_ns(msg) == t_ns


def test_stamps_of_a_200_hz_session_are_exact():
    """The original float code was 1 ns low on about a third of the 5 ms multiples; the integer code must be exact."""
    for k in range(0, 40_001, 3):
        sec, nanosec = Ros2StereoImuRecorder._stamp(k * 5_000_000)
        assert sec * 10**9 + nanosec == k * 5_000_000 and nanosec % 5_000_000 == 0


def test_toggle_request_starts_and_stops_once_per_request(fake_ros, tmp_path):
    rec = Ros2StereoImuRecorder(enabled=True, bag_dir=str(tmp_path))
    rec.process_toggle_request()
    assert not rec.recording                      # nothing requested
    rec.request_toggle()
    rec.process_toggle_request()
    assert rec.recording
    rec.process_toggle_request()                  # the simulator calls this every step: the request must be consumed
    assert rec.recording
    rec.request_toggle()
    rec.process_toggle_request()
    assert not rec.recording


def test_image_accumulator_restarts_with_every_recording(fake_ros, tmp_path):
    rec, _ = _recording(tmp_path)
    for _ in range(3):
        assert not rec.image_due(0.005, 0.033)
    rec.stop_recording()
    rec.start_recording()
    assert [rec.image_due(0.005, 0.033) for _ in range(7)] == [False] * 6 + [True]


def test_two_recordings_in_the_same_second_get_distinct_bags_deterministically(fake_ros, tmp_path, monkeypatch):
    class FrozenClock(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 12, 0, 0)

    monkeypatch.setattr(recorder, "datetime", FrozenClock)
    rec, _ = _recording(tmp_path)
    first = rec.bag_path
    rec.stop_recording()
    rec.start_recording()
    assert rec.recording and rec.bag_path != first and rec.bag_path.endswith("unit_20260101_120000_1")


def test_start_failure_leaves_the_recorder_idle(fake_ros, tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("")                                     # bag_dir that cannot be created
    rec = Ros2StereoImuRecorder(enabled=True, bag_dir=str(blocker / "bags"), bag_prefix="unit")
    rec.start_recording()
    rec.publish_ground_truth([0, 0, 0], [1, 0, 0, 0], 0)      # not recording: silently ignored
    rec.stop_recording()
    assert not rec.recording and rec.message_count == 0 and not FakeWriter.instances


def test_stop_closes_the_writer_and_publishing_after_stop_is_ignored(fake_ros, tmp_path):
    rec, writer = _recording(tmp_path)
    rec.publish_ground_truth([0, 0, 0], [1, 0, 0, 0], 0)
    rec.stop_recording()
    assert writer.closed and not rec.recording and rec.message_count == 1
    rec.publish_ground_truth([0, 0, 0], [1, 0, 0, 0], 5_000_000)
    assert len(writer.records) == 1


def test_run_info_is_written_next_to_the_bag(fake_ros, tmp_path):
    rec = Ros2StereoImuRecorder(enabled=True, bag_dir=str(tmp_path), bag_prefix="unit")
    rec.run_info = {"script": "unit", "args": {"seed": 1}, "np": np.float32(1.5)}      # numpy values must not break it
    rec.start_recording()
    info = json.loads((tmp_path / os.path.basename(rec.bag_path) / "sim_config.json").read_text())
    assert info["args"] == {"seed": 1} and info["np"] == "1.5"


def test_recorder_disables_itself_when_ros_cannot_be_imported(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "rosbag2_py", None)          # makes `import rosbag2_py` raise ImportError
    rec = Ros2StereoImuRecorder(enabled=True, bag_dir=str(tmp_path))
    assert not rec.enabled
    rec.request_toggle()
    rec.request_toggle()
    rec.process_toggle_request()
    rec.publish_ground_truth([0, 0, 0], [1, 0, 0, 0], 0)
    assert not rec.recording and not list(tmp_path.iterdir())
    assert capsys.readouterr().out.count("recorder is disabled") == 1      # warned once, not on every key press
