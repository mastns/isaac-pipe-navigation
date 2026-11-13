# SPDX-License-Identifier: MIT
"""Static checks of pipe_rover_sim.py (it cannot be imported without Isaac Sim, so it is parsed)."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "pipe_rover_sim.py"


def _tree():
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def _cli_flags():
    flags = []
    for node in ast.walk(_tree()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            flags.append(node.args[0].value)
    return flags


def test_script_parses_and_has_the_expected_entry_points():
    names = {n.name for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef)}
    assert {"main", "run_simulator", "reset_rover", "_build_parser", "_validate"} <= names


def test_every_command_line_option_is_documented():
    doc = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    flags = _cli_flags()
    assert len(flags) > 40
    missing = [f for f in flags if f not in doc]
    assert not missing, f"undocumented options: {missing}"


def test_default_asset_paths_exist():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'ASSETS_DIR / "pipe" / "Real_sim.usd"' in text
    assert (ROOT / "assets" / "pipe" / "Real_sim.usd").is_file()
    assert (ROOT / "assets" / "pipe" / "Real_sim.stl").is_file()


def test_no_machine_specific_paths_in_the_sources():
    pattern = re.compile(r"/home/|/Users/|C:\\\\")
    for path in [SCRIPT, *ROOT.glob("pipe_rover/*.py"), *ROOT.glob("scripts/*.py")]:
        assert not pattern.search(path.read_text(encoding="utf-8")), path
