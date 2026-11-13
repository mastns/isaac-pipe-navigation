#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarise a bag recorded by the simulator: topics, message counts, duration and rates.

Reads the rosbag2 ``.db3`` file directly with ``sqlite3``, so no ROS installation is needed.

    python3 scripts/check_bag.py rosbags/stereo_imus_20260101_120000

The exit status is 1 if one of the eight topics of the simulator is missing, 2 if the bag cannot be read.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

EXPECTED_TOPICS = (
    "/stereo/front/left/image_raw", "/stereo/front/right/image_raw",
    "/stereo/rear/left/image_raw", "/stereo/rear/right/image_raw",
    "/imu0", "/imu1", "/imu2", "/ground_truth/pose",
)


def summarise(bag_dir: Path) -> dict:
    """Per-topic statistics of the bag in ``bag_dir`` (timestamps are simulation time in seconds).

    ``monotonic`` is evaluated on the messages in the order in which they were written.
    """
    db_files = sorted(Path(bag_dir).glob("*.db3"))
    if not db_files:
        raise FileNotFoundError(f"no .db3 file in {bag_dir}")
    stats = {}
    for db_file in db_files:
        con = sqlite3.connect(f"{db_file.resolve().as_uri()}?mode=ro", uri=True)   # as_uri() escapes '#', '?', '%'
        try:
            topics = {tid: (name, typ) for tid, name, typ in con.execute("SELECT id, name, type FROM topics")}
            for tid, (name, typ) in topics.items():
                t = np.array([row[0] for row in con.execute(
                    "SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY id", (tid,))], dtype=np.int64)
                entry = stats.setdefault(name, {"type": typ, "t": np.empty(0, dtype=np.int64)})
                entry["t"] = np.concatenate([entry["t"], t])
        finally:
            con.close()
    for entry in stats.values():
        written = entry["t"] * 1e-9                       # write order
        t = np.sort(written)
        dt = np.diff(t)
        entry.update(
            count=len(t),
            first=float(t[0]) if len(t) else float("nan"),
            last=float(t[-1]) if len(t) else float("nan"),
            rate_hz=float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else float("nan"),
            dt_min=float(dt.min()) if len(dt) else float("nan"),
            dt_max=float(dt.max()) if len(dt) else float("nan"),
            monotonic=bool(np.all(np.diff(written) >= 0)),
        )
        del entry["t"]
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bag", type=Path, help="bag directory (contains metadata.yaml and the .db3 file)")
    args = parser.parse_args(argv)

    try:
        stats = summarise(args.bag)
    except (FileNotFoundError, sqlite3.DatabaseError) as err:
        print(f"cannot read the bag: {err}", file=sys.stderr)
        return 2
    if not stats:
        print("the bag contains no topics", file=sys.stderr)
        return 1
    width = max(len(n) for n in stats)
    print(f"{'topic':<{width}}  {'type':<34} {'count':>7} {'rate Hz':>8} {'dt min ms':>10} {'dt max ms':>10}")
    for name in sorted(stats):
        s = stats[name]
        print(f"{name:<{width}}  {s['type']:<34} {s['count']:>7} {s['rate_hz']:>8.2f} "
              f"{s['dt_min'] * 1e3:>10.2f} {s['dt_max'] * 1e3:>10.2f}")
    first = min(s["first"] for s in stats.values())
    last = max(s["last"] for s in stats.values())
    print(f"\nduration {last - first:.2f} s (simulation time {first:.3f} .. {last:.3f} s), "
          f"{sum(s['count'] for s in stats.values())} messages")
    if not all(s["monotonic"] for s in stats.values()):
        print("WARNING: time stamps are not monotonic in write order", file=sys.stderr)
    missing = [t for t in EXPECTED_TOPICS if t not in stats]
    if missing:
        print(f"WARNING: missing topics: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
