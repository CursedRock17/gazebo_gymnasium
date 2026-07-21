#!/usr/bin/env bash
# Run the full ROS 2 ament linter suite over project Python sources.
#
# This is the canonical one-command entry point. CI and devs both call
# this. The wrapper exists because each ament_* tool wants slightly
# different invocation syntax (ament_flake8 takes --config, ament_pep257
# takes --add-ignore as variadic positional, ament_copyright takes paths
# only). Hiding that asymmetry behind one script means CONTRIBUTING.md
# can just say "run scripts/lint.sh".
#
# Exit code: 0 if all three linters pass, non-zero otherwise (set by the
# last failing linter via $RC).
#
# Usage:
#   scripts/lint.sh             # all source dirs
#   scripts/lint.sh PATH...     # only the given files/dirs

# Intentionally no `-u` — ROS 2's setup.bash references unset vars and
# would tank the whole script on sourcing.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -gt 0 ]; then
  TARGETS="$@"
else
  TARGETS="gazebo_gymnasium_bridge/gazebo_gymnasium_bridge \
gazebo_gymnasium_examples/gazebo_gymnasium_resources/plugins \
training_scripts \
gazebo_gymnasium_reinforcement_learning \
scripts"
fi

# pep257 D-rules we permanently disable (rationale in ament_pep257.ini):
#   D100-107  Missing docstring (ament's bundled config also ignores these)
#   D202      Blank line after function docstring
#   D203      1 blank line before class docstring (conflicts with D211)
#   D212/D213 Multi-line summary position (we use first-line summaries)
#   D406-D415 Section-format rules (Args:/Returns:/Raises:). Project doesn't
#             use Google/NumPy section docstrings.
PEP257_IGNORE="D100 D101 D102 D103 D104 D105 D106 D107 D202 D203 D212 D213 \
D406 D407 D411 D413 D415"

# Try to source ROS so ament_* are on PATH. Don't fail if it's not there —
# print a clear error.
if ! command -v ament_flake8 >/dev/null 2>&1; then
  if [ -f /opt/ros/jazzy/setup.bash ]; then
    # shellcheck source=/dev/null
    source /opt/ros/jazzy/setup.bash
  fi
fi
if ! command -v ament_flake8 >/dev/null 2>&1; then
  echo "ERROR: ament_flake8 not on PATH. Source your ROS 2 setup first."
  exit 2
fi

RC=0

echo "=== ament_flake8 ==="
ament_flake8 --config "$PROJECT_ROOT/ament_flake8.ini" $TARGETS
RC_FLAKE=$?
[ $RC_FLAKE -ne 0 ] && RC=$RC_FLAKE
echo

echo "=== ament_pep257 ==="
ament_pep257 --add-ignore $PEP257_IGNORE -- $TARGETS
RC_PEP=$?
[ $RC_PEP -ne 0 ] && RC=$RC_PEP
echo

echo "=== ament_copyright ==="
ament_copyright $TARGETS
RC_CR=$?
[ $RC_CR -ne 0 ] && RC=$RC_CR
echo

echo "Done. flake8=$RC_FLAKE pep257=$RC_PEP copyright=$RC_CR"
exit $RC
