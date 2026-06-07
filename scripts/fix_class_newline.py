#!/usr/bin/env python3
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

"""Insert a blank line between `class Foo:` and the first method (flake8-class-newline / CNL100).

ROS 2's ament_flake8 bundles `flake8-class-newline` which enforces a blank
line between the class header and its body. Most editors strip that line;
this script puts it back.

Only touches classes that currently have `class X:` immediately followed by
an indented method/attribute line — no blank line in between. Idempotent.

Run: ./venv/bin/python scripts/fix_class_newline.py
"""

from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = [
    PROJECT_ROOT / "gazebo_gymnasium_bridge" / "gazebo_gymnasium_bridge",
    PROJECT_ROOT / "gazebo_gymnasium_examples" / "gazebo_gymnasium_resources" / "plugins",
    PROJECT_ROOT / "training_scripts",
    PROJECT_ROOT / "gazebo_gymnasium_reinforcement_learning",
    PROJECT_ROOT / "scripts",
]


CLASS_HEAD_RE = re.compile(r"^(\s*class\s+\w+[^\n]*:)\s*\n(\s+)(\S)", re.MULTILINE)


def fix(text: str) -> tuple[str, int]:
    fixed = 0

    def _repl(m):
        nonlocal fixed
        # Only insert a blank line if what follows isn't already one. The
        # regex required \S right after the class header line — so we're sure
        # there's no blank line between class and first body line.
        fixed += 1
        return f"{m.group(1)}\n\n{m.group(2)}{m.group(3)}"

    new_text = CLASS_HEAD_RE.sub(_repl, text)
    return new_text, fixed


def main(argv):
    targets = [Path(a) for a in argv[1:]] if len(argv) > 1 else DEFAULT_TARGETS
    total = 0
    files = 0
    for t in targets:
        t = Path(t)
        paths = [t] if t.is_file() else sorted(t.rglob("*.py"))
        for p in paths:
            try:
                text = p.read_text()
            except UnicodeDecodeError:
                continue
            new_text, n = fix(text)
            if n == 0:
                continue
            p.write_text(new_text)
            files += 1
            total += n
            print(f"  fixed {p.relative_to(PROJECT_ROOT)} (+{n} class-header blank lines)")
    print(f"\nInserted {total} blank lines across {files} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
