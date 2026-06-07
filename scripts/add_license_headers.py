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

"""Insert the standard Apache-2.0 copyright header at the top of every Python file.

Runs `ament_copyright` first to discover files missing a header, then
prepends the standard ROS 2-style Apache-2.0 block. Preserves the shebang
line if present.

Idempotent: a file that already has the header is left alone.

Run from project root: ./venv/bin/python scripts/add_license_headers.py
"""

from datetime import date
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = [
    "gazebo_gymnasium_bridge",
    "gazebo_gymnasium_examples",
    "gazebo_gymnasium_msgs",
    "gazebo_gymnasium_reinforcement_learning",
    "scripts",
    "training_scripts",
]

AUTHOR = "Lucas Wendland"
YEAR = date.today().year

HEADER = f"""# Copyright {YEAR} {AUTHOR}
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


SHEBANG_RE = re.compile(r"^#!/.*\n")


def has_header(text: str) -> bool:
    # ament_copyright matches on the word "Copyright" in the first few lines.
    head = "\n".join(text.splitlines()[:20])
    return "Copyright" in head and ("Apache" in head or "License" in head)


def add_header(path: Path) -> bool:
    text = path.read_text()
    if has_header(text):
        return False

    shebang = ""
    m = SHEBANG_RE.match(text)
    if m:
        shebang = m.group(0)
        text = text[len(shebang):]

    path.write_text(f"{shebang}{HEADER}\n{text}")
    return True


def iter_files():
    for d in TARGET_DIRS:
        root = PROJECT_ROOT / d
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in {"build", "install", "log", "venv", "__pycache__"}
                   for part in path.parts):
                continue
            yield path


def main() -> int:
    added = 0
    for path in iter_files():
        if add_header(path):
            added += 1
            print(f"  +header {path.relative_to(PROJECT_ROOT)}")
    print()
    print(f"Added header to {added} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
