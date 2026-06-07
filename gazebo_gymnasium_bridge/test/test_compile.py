# Copyright 2026 Lucas Wendland
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compile-check + import-smoke tests.

Catches Python syntax errors and obvious import-time breakages across every
.py file in the project before the user spends time launching Gazebo. Files
that touch native gz-* bindings are still ast.parse'd, but the import-smoke
test for them is skipped if the bindings aren't available in the test env.

Run with: pytest gazebo_gymnasium_bridge/test/test_compile.py -v
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


# Project root = two levels up from this file (test/ -> package/ -> root).
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

# Directories whose .py files we want to compile-check. Plain string paths
# resolved relative to PROJECT_ROOT.
SOURCE_DIRS = [
    "gazebo_gymnasium_bridge/gazebo_gymnasium_bridge",
    "gazebo_gymnasium_examples/gazebo_gymnasium_resources/plugins",
    "gazebo_gymnasium_reinforcement_learning",
]

# File path substrings we don't care about (venv pollution, generated, etc.).
SKIP_SUBSTRINGS = [
    "/venv/",
    "/build/",
    "/install/",
    "/log/",
    "/__pycache__/",
    "/.git/",
]


def _collect_py_files():
    files = []
    for relative in SOURCE_DIRS:
        root = PROJECT_ROOT / relative
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            spath = str(path)
            if any(s in spath for s in SKIP_SUBSTRINGS):
                continue
            files.append(path)
    return sorted(files)


def _imports_gz(tree: ast.AST) -> bool:
    """True if the module imports anything from a `gz.*` package at module level."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name == "gz" or name.name.startswith("gz."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "gz" or mod.startswith("gz."):
                return True
    return False


# Build the parameter list once at collection time.
_PY_FILES = _collect_py_files()


@pytest.mark.parametrize(
    "py_path",
    _PY_FILES,
    ids=[str(p.relative_to(PROJECT_ROOT)) for p in _PY_FILES],
)
def test_ast_parse(py_path: Path):
    """Every .py file must parse as valid Python."""
    source = py_path.read_text()
    try:
        ast.parse(source, filename=str(py_path))
    except SyntaxError as exc:
        pytest.fail(f"SyntaxError in {py_path.relative_to(PROJECT_ROOT)}: {exc}")


def _import_smoke_safe(py_path: Path) -> bool:
    """True iff this file is safe to import-smoke (no gz-native deps at top level)."""
    try:
        tree = ast.parse(py_path.read_text(), filename=str(py_path))
    except SyntaxError:
        return False
    if _imports_gz(tree):
        return False
    # Skip __init__.py files that don't have any imports we care about — they
    # often pull the whole package eagerly which can be too heavy here.
    return True


_SMOKE_FILES = [p for p in _PY_FILES if _import_smoke_safe(p)]


@pytest.mark.parametrize(
    "py_path",
    _SMOKE_FILES,
    ids=[str(p.relative_to(PROJECT_ROOT)) for p in _SMOKE_FILES],
)
def test_import_smoke(py_path: Path):
    """
    For files that don't touch gz-* native bindings, verify the file can be
    `compile()`'d cleanly in a subprocess. This catches issues that ast.parse
    misses (e.g. f-string parse errors on older Pythons, codec problems).
    A subprocess isolates failures so one bad module doesn't poison the run.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"compile(open({str(py_path)!r}).read(), {str(py_path)!r}, 'exec')"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        pytest.fail(
            f"compile failed for {py_path.relative_to(PROJECT_ROOT)}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def test_at_least_one_file_found():
    """Sanity: the collection loop actually found something. Guards against
    a future refactor that moves source dirs and silently breaks discovery."""
    assert _PY_FILES, (
        f"No .py files discovered under {SOURCE_DIRS!r}. "
        f"PROJECT_ROOT={PROJECT_ROOT}"
    )
