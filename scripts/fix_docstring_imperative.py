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

"""Rewrite docstring first-line verbs into imperative mood (PEP 257 / D401).

D401 ("First line should be in imperative mood") is the largest remaining
class after the mechanical formatting pass. Most cases are common
third-person-singular verbs ("Returns ...", "Sets ...", "Calculates ...")
that should be infinitive ("Return ...", "Set ...", "Calculate ...").

This script walks every .py file under the project source dirs, finds the
first non-blank line of each docstring, and rewrites the verb if it matches
one of the known patterns. Only safe rewrites — patterns we're certain
about. Anything ambiguous is left for hand-review.

Run from project root:
    ./venv/bin/python scripts/fix_docstring_imperative.py
"""

import ast
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


# Mapping from third-person-singular present to imperative. Conservative —
# only verbs we're confident about. The ones with `-s` removed straightforwardly
# (Returns → Return) AND irregulars where stripping `-s` doesn't work
# (Tries → Try, Handles → Handle).
VERB_MAP = {
    "Returns": "Return",
    "Sets": "Set",
    "Gets": "Get",
    "Adds": "Add",
    "Creates": "Create",
    "Calculates": "Calculate",
    "Computes": "Compute",
    "Builds": "Build",
    "Decides": "Decide",
    "Loads": "Load",
    "Stores": "Store",
    "Saves": "Save",
    "Handles": "Handle",
    "Checks": "Check",
    "Removes": "Remove",
    "Updates": "Update",
    "Inserts": "Insert",
    "Combines": "Combine",
    "Counts": "Count",
    "Parses": "Parse",
    "Resolves": "Resolve",
    "Tries": "Try",
    "Generates": "Generate",
    "Initializes": "Initialize",
    "Performs": "Perform",
    "Runs": "Run",
    "Walks": "Walk",
    "Finds": "Find",
    "Sends": "Send",
    "Receives": "Receive",
    "Reads": "Read",
    "Writes": "Write",
    "Fetches": "Fetch",
    "Yields": "Yield",
    "Splits": "Split",
    "Maps": "Map",
    "Wraps": "Wrap",
}


VERB_RE = re.compile(
    r"^(?P<lead>\s*)(?P<verb>" + "|".join(VERB_MAP.keys()) + r")\b"
)


def rewrite_first_line(line: str) -> tuple[str, bool]:
    m = VERB_RE.match(line)
    if not m:
        return line, False
    verb = m.group("verb")
    new = m.group("lead") + VERB_MAP[verb] + line[m.end():]
    return new, True


def rewrite_source(source: str) -> tuple[str, int]:
    """Walk the AST, find docstring nodes, rewrite their first lines."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, 0

    # Collect (lineno_start, lineno_end, original_docstring_node) for every
    # docstring in the module. We rewrite by editing the source lines and
    # re-serializing, since ast.unparse loses comments.
    lines = source.splitlines(keepends=True)
    changed = 0

    def visit(node):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                doc_node = node.body[0]
                start = doc_node.lineno - 1
                # The first line of the docstring text might be on the SAME
                # line as the opening quote, e.g. `"""Summary line.` or it
                # could be on the next line (after a bare """).
                first_line_idx = start
                # Find which source line actually contains the summary text.
                while first_line_idx < len(lines):
                    stripped = lines[first_line_idx].lstrip()
                    if stripped.startswith(('"""', "'''")):
                        body_part = stripped[3:].lstrip()
                        if body_part and not body_part.startswith(('"""', "'''")):
                            # Summary is on this same line
                            break
                        first_line_idx += 1
                        continue
                    if stripped:  # First non-blank non-quote line
                        break
                    first_line_idx += 1
                if first_line_idx >= len(lines):
                    return
                # Rewrite that line if its first word is in our verb map.
                # Preserve leading indentation + any quote prefix.
                line = lines[first_line_idx]
                # Skip past leading whitespace and optional quote prefix.
                m_prefix = re.match(r"^(\s*)([\"']{3})?(.*)$", line, re.DOTALL)
                indent, q, rest = m_prefix.group(1), m_prefix.group(2) or "", m_prefix.group(3)
                new_rest, did = rewrite_first_line(rest)
                if did:
                    nonlocal changed
                    changed += 1
                    lines[first_line_idx] = f"{indent}{q}{new_rest}"
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return "".join(lines), changed


def iter_py_files(targets):
    for t in targets:
        t = Path(t)
        if t.is_file() and t.suffix == ".py":
            yield t
        elif t.is_dir():
            yield from sorted(t.rglob("*.py"))


def main(argv):
    targets = [Path(a) for a in argv[1:]] if len(argv) > 1 else DEFAULT_TARGETS
    total_files = 0
    total_rewritten = 0
    for py in iter_py_files(targets):
        try:
            text = py.read_text()
        except UnicodeDecodeError:
            continue
        new_text, n = rewrite_source(text)
        if n == 0:
            continue
        py.write_text(new_text)
        total_files += 1
        total_rewritten += n
        print(f"  fixed {py.relative_to(PROJECT_ROOT)} (-{n} non-imperative docstrings)")
    print()
    print(f"Rewrote {total_rewritten} docstring first lines across {total_files} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
