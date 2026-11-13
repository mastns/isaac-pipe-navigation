# SPDX-License-Identifier: MIT
"""scripts/headless_demo.py is the installation test advertised in the README; its logic runs without Isaac Sim."""

import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import check_bag
import headless_demo as hd
import numpy as np
import pytest

PATH_FILE = Path(hd.__file__).with_name("demo_path.json")


class _Tensor:
    """The slice of a torch tensor that ScriptedKeyboard uses: [i].detach().cpu().numpy()."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def __getitem__(self, i):
        return _Tensor(self.a[i])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def _holder(x, y, yaw):
    quat = [[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=_Tensor([[x, y, 0.1]]), root_quat_w=_Tensor(quat)))
    return SimpleNamespace(scene={"robot": robot})


def _keyboard(profile, holder=None, duration=20.0):
    return hd.ScriptedKeyboard(profile, duration, 0.005, 0.25, 2.0, holder or _holder(0.1, 0.0, 0.0), PATH_FILE)


def test_forward_profile_records_after_half_a_second_drives_after_one_and_quits():
    kb = _keyboard("forward", duration=2.0)
    pressed = []
    kb.add_callback("B", lambda: pressed.append(kb.step))
    assert not any(kb.advance().any() for _ in range(100)) and pressed == []     # t < 0.5 s: settling, no bag yet
    assert not kb.advance().any() and pressed == [100]                          # t = 0.5 s: B starts the recording
    assert not any(kb.advance().any() for _ in range(99))                        # t < 1 s: standing still
    assert kb.advance().tolist() == [0.25, 0.0, 0.0] and not kb._exit            # t = 1 s: full speed, no turning
    for _ in range(199):
        kb.advance()
    assert not kb._exit
    assert not kb.advance().any() and kb._exit and pressed == [100]              # t = 2 s: stop and quit


def test_pure_pursuit_drives_straight_when_on_the_path():
    kb = _keyboard("path", _holder(0.1, 0.0, 0.0))
    kb.step = 200
    vx, _, wz = kb.advance()
    assert vx == pytest.approx(0.25) and abs(wz) < 1e-6


@pytest.mark.parametrize(("yaw", "turn_sign"), [(0.5, -1), (-0.5, +1)])
def test_pure_pursuit_turns_back_towards_the_path(yaw, turn_sign):
    kb = _keyboard("path", _holder(0.1, 0.0, yaw))
    kb.step = 200
    vx, _, wz = kb.advance()
    assert math.copysign(1.0, wz) == turn_sign
    assert abs(wz) == pytest.approx(1.0, abs=0.01)                        # 2 * 0.5 rad, below the 2 rad/s clamp
    assert vx == pytest.approx(0.25 * math.cos(yaw), abs=0.01)


def test_pure_pursuit_stops_at_the_far_mouth_and_quits_a_second_later():
    end = np.array(__import__("json").loads(PATH_FILE.read_text())["points"][-1])
    kb = _keyboard("path", _holder(end[0], end[1], 0.0))
    kb.step = 1000
    assert not kb.advance().any() and not kb._exit
    kb.step += 201
    assert not kb.advance().any() and kb._exit


def _write_bag(root, imu_hz=200.0, skip=()):
    bag = root / "run1"
    bag.mkdir(parents=True)
    (bag / "metadata.yaml").write_text("")
    con = sqlite3.connect(bag / "run1_0.db3")
    con.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, "
                "offered_qos_profiles TEXT)")
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
    for tid, name in enumerate(check_bag.EXPECTED_TOPICS, start=1):
        if name in skip:
            continue
        hz = 28.571 if "image_raw" in name else imu_hz
        con.execute("INSERT INTO topics VALUES (?, ?, 't', 'cdr', '')", (tid, name))
        con.executemany("INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, x'')",
                        [(tid, round(k / hz * 1e9)) for k in range(int(2 * hz) + 1)])
    con.commit()
    con.close()


def _verify(tmp_path, monkeypatch, spray_dots=5, **bag):
    _write_bag(tmp_path, **bag)
    monkeypatch.setattr(hd, "_distance_travelled", lambda _dir: 1.0)
    return hd.verify(SimpleNamespace(bag_dir=tmp_path), {"spray_dots": spray_dots})


def test_verify_passes_a_good_recording(tmp_path, monkeypatch, capsys):
    assert _verify(tmp_path, monkeypatch) == 0
    assert "FAIL" not in capsys.readouterr().out


def test_verify_fails_on_a_wrong_imu_rate(tmp_path, monkeypatch, capsys):
    assert _verify(tmp_path, monkeypatch, imu_hz=150.0) == 1
    assert "FAIL  /imu0 runs at 200 Hz" in capsys.readouterr().out


def test_verify_fails_without_spray_dots(tmp_path, monkeypatch):
    assert _verify(tmp_path, monkeypatch, spray_dots=0) == 1


def test_verify_fails_without_a_bag(tmp_path):
    assert hd.verify(SimpleNamespace(bag_dir=tmp_path), {"spray_dots": 5}) == 1


def test_verify_reports_a_missing_topic_as_a_failed_check(tmp_path, monkeypatch, capsys):
    assert _verify(tmp_path, monkeypatch, skip=("/imu2",)) == 1
    assert "FAIL  all 8 topics are present" in capsys.readouterr().out


def test_verify_ignores_bags_that_existed_before_the_run(tmp_path, monkeypatch, capsys):
    _write_bag(tmp_path)
    monkeypatch.setattr(hd, "_distance_travelled", lambda _dir: 1.0)
    before = hd._bag_dirs(tmp_path)
    assert hd.verify(SimpleNamespace(bag_dir=tmp_path), {"spray_dots": 5}, before) == 1
    assert "exactly one new bag" in capsys.readouterr().out


def test_verify_fails_when_the_distance_cannot_be_determined(tmp_path, monkeypatch, capsys):
    _write_bag(tmp_path)
    monkeypatch.setattr(hd, "_distance_travelled", lambda _dir: None)
    assert hd.verify(SimpleNamespace(bag_dir=tmp_path), {"spray_dots": 5}) == 1
    assert "FAIL  the rover moved" in capsys.readouterr().out
