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

"""Normalize Python string-literal quote style to double quotes.

Why this exists: flake8-quotes (bundled in ament_flake8) checks that all
string literals use a consistent quote style. The default is single, but we
override to double in `ament_flake8.ini` to match modern Python convention
(black, ruff). After that override, the only remaining Q000 errors are
single-quoted literals we typed by reflex — this script rewrites them in
place.

The classic risk with sed-style rewrites is breaking strings that contain
the *other* quote character (e.g. `'this has "quotes" inside'` → can't
naively flip to double). Using `tokenize` instead means we operate on the
AST-level token stream and only rewrite tokens that are safe to rewrite.

A literal is *safe* to flip from single to double if:
  - the contents contain no unescaped `"` character, AND
  - the literal isn't a triple-quoted (multi-line) string (those are
    handled by docstring tools, not us — we leave them alone)

Run: ./venv/bin/python scripts/normalize_string_quotes.py [files-or-dirs...]
With no arguments, defaults to all Python source dirs in the project.
"""

import io
from pathlib import Path
import sys
import tokenize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = [
    PROJECT_ROOT / "gazebo_gymnasium_bridge" / "gazebo_gymnasium_bridge",
    PROJECT_ROOT / "gazebo_gymnasium_examples" / "gazebo_gymnasium_resources" / "plugins",
    PROJECT_ROOT / "training_scripts",
    PROJECT_ROOT / "gazebo_gymnasium_reinforcement_learning",
    PROJECT_ROOT / "scripts",
]


def _can_flip_to_double(literal: str) -> bool:
    """Decide if a single-quoted literal can be safely rewritten."""
    if not literal:
        return False
    # Strip the prefix (b, r, f, rb, br, fr, rf, etc.) before checking quotes.
    i = 0
    while i < len(literal) and literal[i] not in "'\"":
        i += 1
    body = literal[i:]
    if not body.startswith("'") or body.startswith("'''"):
        return False
    # body is now "'...'" — check the inner content for unescaped double quotes
    inner = body[1:-1]
    # Walk inner, respecting backslash escapes.
    j = 0
    while j < len(inner):
        if inner[j] == "\\" and j + 1 < len(inner):
            j += 2
            continue
        if inner[j] == '"':
            return False
        j += 1
    return True


def _flip_literal(literal: str) -> str:
    i = 0
    while i < len(literal) and literal[i] not in "'\"":
        i += 1
    prefix, body = literal[:i], literal[i:]
    inner = body[1:-1]
    # When we change the outer quote, we no longer need to escape single
    # quotes in the body, and we DO need to escape any inner unescaped
    # double quotes — but `_can_flip_to_double` already certified there are
    # none, so we can leave inner alone.
    # However, `\'` becomes just `'` since it's no longer the outer quote.
    inner = inner.replace("\\'", "'")
    return f'{prefix}"{inner}"'


def rewrite(source: str) -> tuple[str, int]:
    """Return (new_source, n_replaced)."""
    tokens = list(tokenize.tokenize(io.BytesIO(source.encode("utf-8")).readline))
    new_tokens = []
    n_replaced = 0
    for tok in tokens:
        if tok.type == tokenize.STRING and _can_flip_to_double(tok.string):
            new_tokens.append(tok._replace(string=_flip_literal(tok.string)))
            n_replaced += 1
        else:
            new_tokens.append(tok)
    if n_replaced == 0:
        return source, 0
    return tokenize.untokenize(new_tokens).decode("utf-8"), n_replaced


def iter_python_files(targets):
    for target in targets:
        target = Path(target)
        if target.is_file() and target.suffix == ".py":
            yield target
        elif target.is_dir():
            yield from sorted(target.rglob("*.py"))


def main(argv):
    targets = [Path(a) for a in argv[1:]] if len(argv) > 1 else DEFAULT_TARGETS
    total_files = 0
    total_replaced = 0
    for py in iter_python_files(targets):
        try:
            text = py.read_text()
        except UnicodeDecodeError:
            continue
        try:
            new_text, n = rewrite(text)
        except tokenize.TokenizeError as e:
            print(f"  SKIP {py.relative_to(PROJECT_ROOT)}: tokenize error ({e})")
            continue
        if n == 0:
            continue
        py.write_text(new_text)
        total_files += 1
        total_replaced += n
        print(f"  fixed {py.relative_to(PROJECT_ROOT)} (-{n} single-quoted literals)")
    print()
    print(f"Flipped {total_replaced} string literals to double quotes across "
          f"{total_files} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
