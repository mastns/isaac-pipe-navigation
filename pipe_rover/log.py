# SPDX-License-Identifier: MIT
"""Minimal console logging with the ``[INFO]`` / ``[WARN]`` / ``[ERROR]`` prefixes used by the simulator.

Plain ``print`` is used on purpose: inside Isaac Sim's Kit app the standard ``logging`` handlers are
re-routed and the output ordering relative to the app's own messages is hard to predict.
``debug`` messages are only shown after ``set_verbose(True)`` (the ``--verbose`` command-line flag).
"""

from __future__ import annotations

_VERBOSE = False


def set_verbose(enabled: bool) -> None:
    """Enable or disable ``debug`` output."""
    global _VERBOSE
    _VERBOSE = bool(enabled)


def is_verbose() -> bool:
    return _VERBOSE


def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def error(message: str) -> None:
    print(f"[ERROR] {message}", flush=True)


def debug(message: str) -> None:
    if _VERBOSE:
        print(f"[DEBUG] {message}", flush=True)
