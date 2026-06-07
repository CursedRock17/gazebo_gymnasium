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

"""Remove blank lines between `class Foo:` and its docstring (D211).

D211 says "No blank lines allowed before class docstring". The companion
`fix_class_newline.py` script erred on the side of inserting blank lines
unconditionally; this script removes the ones that landed between a class
header and its docstring.

Idempotent.

Run: ./venv/bin/python scripts/fix_class_docstring_blanks.py
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


# Find `class X(...):\n\n    """` and drop the blank between header and docstring.
PATTERN = re.compile(
    r"^(\s*class\s+\w+[^\n]*:)\n\s*\n(\s+\"\"\")",
    re.MULTILINE,
)


def fix(text: str) -> tuple[str, int]:
    fixed = [0]

    def _repl(m):
        fixed[0] += 1
        return f"{m.group(1)}\n{m.group(2)}"

    new_text = PATTERN.sub(_repl, text)
    return new_text, fixed[0]


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
            print(f"  fixed {p.relative_to(PROJECT_ROOT)} (-{n} stray blanks)")
    print(f"\nRemoved {total} stray blank lines across {files} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
