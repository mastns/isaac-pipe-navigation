# SPDX-License-Identifier: MIT
"""scripts/check_bag.py reads the rosbag2 sqlite file directly; test it on a synthetic bag."""

import sqlite3

import check_bag
import pytest

TOPICS = {name: "sensor_msgs/msg/Imu" for name in check_bag.EXPECTED_TOPICS}


def _make_bag(path, rates_hz, duration=2.0, skip=()):
    path.mkdir()
    con = sqlite3.connect(path / "demo_0.db3")
    con.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, "
                "offered_qos_profiles TEXT)")
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
    for tid, name in enumerate(check_bag.EXPECTED_TOPICS, start=1):
        if name in skip:
            continue
        con.execute("INSERT INTO topics VALUES (?, ?, ?, 'cdr', '')", (tid, name, TOPICS[name]))
        rate = rates_hz[name]
        n = int(duration * rate) + 1
        con.executemany("INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
                        [(tid, round(k / rate * 1e9), b"") for k in range(n)])
    con.commit()
    con.close()


def _rates(image_hz=28.571, imu_hz=200.0):
    return {n: (image_hz if "image_raw" in n else imu_hz) for n in check_bag.EXPECTED_TOPICS}


def test_summarise_reports_counts_and_rates(tmp_path):
    _make_bag(tmp_path / "bag", _rates())
    stats = check_bag.summarise(tmp_path / "bag")
    assert set(stats) == set(check_bag.EXPECTED_TOPICS)
    assert stats["/imu0"]["count"] == 401
    assert stats["/imu0"]["rate_hz"] == pytest.approx(200.0, rel=1e-3)
    assert stats["/stereo/front/left/image_raw"]["rate_hz"] == pytest.approx(28.571, rel=1e-2)
    assert all(s["monotonic"] for s in stats.values())


def test_main_succeeds_on_a_complete_bag_and_fails_when_a_topic_is_missing(tmp_path, capsys):
    _make_bag(tmp_path / "good", _rates())
    assert check_bag.main([str(tmp_path / "good")]) == 0
    assert "/ground_truth/pose" in capsys.readouterr().out
    _make_bag(tmp_path / "bad", _rates(), skip=("/imu2",))
    assert check_bag.main([str(tmp_path / "bad")]) == 1
    assert "missing topics: /imu2" in capsys.readouterr().err


def test_missing_database_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        check_bag.summarise(tmp_path / "empty")


def test_messages_written_out_of_time_order_are_flagged(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    con = sqlite3.connect(bag / "x_0.db3")
    con.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, "
                "offered_qos_profiles TEXT)")
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
    con.execute("INSERT INTO topics VALUES (1, '/imu0', 'sensor_msgs/msg/Imu', 'cdr', '')")
    for ts in (0, 10_000_000, 5_000_000):                      # the third message is older than the second
        con.execute("INSERT INTO messages (topic_id, timestamp, data) VALUES (1, ?, x'')", (ts,))
    con.commit()
    con.close()
    assert check_bag.summarise(bag)["/imu0"]["monotonic"] is False


@pytest.mark.parametrize("name", ["with space", "hash#2", "query?x", "percent%20"])
def test_bags_in_directories_with_special_characters_can_be_read(tmp_path, name):
    _make_bag(tmp_path / name, _rates())
    assert check_bag.summarise(tmp_path / name)["/imu1"]["count"] == 401


def test_main_reports_an_unreadable_bag_with_status_2(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert check_bag.main([str(tmp_path / "empty")]) == 2
    assert "cannot read the bag" in capsys.readouterr().err
