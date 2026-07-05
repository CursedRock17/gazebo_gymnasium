# Copyright 2026 Lucas Wendland
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# gazebo_gymnasium environment setup — SOURCE this, don't execute:
#
#     source scripts/env.sh server   # terminal running gz sim (ros2 launch)
#     source scripts/env.sh client   # terminal running RL train/eval/tests
#
# Why two modes: the gz *server* runs the source-built Gazebo whose embedded
# Python (3.13) needs the gz bindings on PYTHONPATH; the RL *client* runs the
# project venv (3.12, torch/SB3 + apt gz.transport) and must NOT carry the
# 3.13 gz libs (ABI clash -> segfault). See docs; Tier-1 work aims to collapse
# this into a single Python so this split goes away.
#
# Override before sourcing:
#   GAZEBO_GYM_ROS_DISTRO   (default: jazzy)
#   GAZEBO_GYM_HARMONIC_WS  (default: $HOME/harmonic_ws, then a known fallback)
#   GAZEBO_GYM_VENV         (default: <repo>/venv)

_ggym_mode="${1:-}"
if [ "$_ggym_mode" != "server" ] && [ "$_ggym_mode" != "client" ]; then
    echo "usage: source scripts/env.sh {server|client}" >&2
    return 2 2>/dev/null || exit 2
fi

# Resolve repo root from this script's own location (works when sourced).
_ggym_src="${BASH_SOURCE[0]:-$0}"
GAZEBO_GYM_ROOT="$(cd "$(dirname "$_ggym_src")/.." && pwd)"
: "${GAZEBO_GYM_ROS_DISTRO:=jazzy}"
: "${GAZEBO_GYM_VENV:=$GAZEBO_GYM_ROOT/venv}"

# Put system Python first and strip conda/pyenv shims that would shadow it
# (the classic "works in one terminal, segfaults in another" trap).
export PATH="/usr/bin:$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -vE 'miniforge|anaconda3?|/conda/|/\.pyenv/' | paste -sd:)"
unset PYTHONHOME

source "/opt/ros/$GAZEBO_GYM_ROS_DISTRO/setup.bash" 2>/dev/null
source "$GAZEBO_GYM_ROOT/install/setup.bash" 2>/dev/null

_ggym_scrub_harmonic() {  # drop harmonic_ws entries from a :-list
    printf '%s' "${1:-}" | tr ':' '\n' | grep -v '/harmonic_ws/' | paste -sd:
}

if [ "$_ggym_mode" = "server" ]; then
    : "${GAZEBO_GYM_HARMONIC_WS:=$HOME/harmonic_ws}"
    if [ ! -d "$GAZEBO_GYM_HARMONIC_WS/install/gz-sim8/lib/python" ] \
       && [ -d /hdd/Documents/open_source/harmonic_ws/install/gz-sim8/lib/python ]; then
        GAZEBO_GYM_HARMONIC_WS=/hdd/Documents/open_source/harmonic_ws
    fi
    if [ ! -d "$GAZEBO_GYM_HARMONIC_WS/install/gz-sim8/lib/python" ]; then
        echo "[env] WARN: gz bindings not found under GAZEBO_GYM_HARMONIC_WS" \
             "($GAZEBO_GYM_HARMONIC_WS); set it to your Gazebo source workspace." >&2
    fi
    export PYTHONUNBUFFERED=1
    _ggym_H="$GAZEBO_GYM_HARMONIC_WS/install"
    for _ggym_p in gz-sim8 sdformat14 gz-math7 gz-msgs10 gz-transport13; do
        export PYTHONPATH="$_ggym_H/$_ggym_p/lib/python:${PYTHONPATH:-}"
    done
    echo "[env] server mode — gz sim (HARMONIC_WS=$GAZEBO_GYM_HARMONIC_WS)"
else
    export LD_LIBRARY_PATH="$(_ggym_scrub_harmonic "${LD_LIBRARY_PATH:-}")"
    export PYTHONPATH="$(_ggym_scrub_harmonic "${PYTHONPATH:-}")"
    # Put the venv interpreter first deterministically. (`source activate`
    # proved unreliable here — the venv's python3 is a bare symlink — so we
    # prepend venv/bin explicitly, which is what activation is meant to do.)
    if [ -x "$GAZEBO_GYM_VENV/bin/python3" ]; then
        export VIRTUAL_ENV="$GAZEBO_GYM_VENV"
        export PATH="$GAZEBO_GYM_VENV/bin:$PATH"
    else
        echo "[env] WARN: venv not found at $GAZEBO_GYM_VENV (see README);" \
             "falling back to system python3" >&2
    fi
    echo "[env] client mode — RL venv (python=$(command -v python3))"
fi

unset _ggym_mode _ggym_src _ggym_H _ggym_p
