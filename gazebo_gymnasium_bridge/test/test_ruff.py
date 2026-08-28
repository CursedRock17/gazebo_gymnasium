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
"""Ruff lint + format checks, config in pyproject.toml's [tool.ruff].

Replaces the former test_flake8.py + test_pep257.py + test_yapf.py trio
(ament_flake8, ament_pep257, and yapf respectively) with a single tool
covering the same ground: PEP 8, docstring convention, import order, and
Google-style formatting. test_copyright.py stays separate and untouched --
ament_copyright's Apache-2.0 header check is unrelated to style/lint and
has no Ruff equivalent.
"""

import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, os.pardir, os.pardir)
_TARGETS = [
    os.path.join(_ROOT, "gazebo_gymnasium_bridge"),
    os.path.join(_ROOT, "gazebo_gymnasium_examples"),
    os.path.join(_ROOT, "training_scripts"),
]


@pytest.mark.linter
def test_ruff_check():
    """`ruff check` (pyflakes, pycodestyle, isort, pydocstyle, bugbear)."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *_TARGETS],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ruff check found issues:\n{result.stdout}{result.stderr}"


@pytest.mark.linter
def test_ruff_format():
    """`ruff format --check` -- whitespace/layout only, no docstring content."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", *_TARGETS],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ruff format found unformatted files:\n{result.stdout}{result.stderr}"
    )
