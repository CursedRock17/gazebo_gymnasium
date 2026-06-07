#!/usr/bin/env bash
# Print the Python / ROS / Gazebo environment in a shell. Useful when
# debugging "works in one terminal, segfaults in another" — diff the
# output of this script between the working and broken shells to find
# the culprit.
#
# Usage:
#   scripts/diagnose_env.sh > /tmp/working.env   # in the good terminal
#   scripts/diagnose_env.sh > /tmp/broken.env    # in the bad terminal
#   diff /tmp/working.env /tmp/broken.env

echo "=== Python interpreters on PATH ==="
which -a python python3 python3.12 2>/dev/null
echo

echo "=== Python env vars ==="
for v in PYTHONHOME PYTHONPATH PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX \
         CONDA_DEFAULT_ENV CONDA_PYTHON_EXE PYENV_VERSION PYENV_VIRTUAL_ENV; do
    echo "$v=${!v:-<unset>}"
done
echo

echo "=== Loader / library paths ==="
for v in LD_LIBRARY_PATH LD_PRELOAD; do
    echo "$v=${!v:-<unset>}"
done
echo

echo "=== ROS env vars ==="
for v in ROS_DISTRO ROS_VERSION AMENT_PREFIX_PATH CMAKE_PREFIX_PATH; do
    echo "$v=${!v:-<unset>}"
done
echo

echo "=== Gazebo env vars ==="
for v in GZ_VERSION GZ_SIM_RESOURCE_PATH GZ_SIM_SYSTEM_PLUGIN_PATH \
         GZ_TRANSPORT_RELAY GZ_PARTITION; do
    echo "$v=${!v:-<unset>}"
done
echo

echo "=== libpython3.12 visible to the dynamic loader ==="
ldconfig -p 2>/dev/null | grep -E "libpython3\.1[12]" | head -5
echo

echo "=== libpython that 'python3' is actually linked against ==="
ldd "$(which python3)" 2>/dev/null | grep -E "libpython|libc\.so" | head -5
echo

echo "=== gz sim version + binary location ==="
which gz 2>/dev/null
gz sim --version 2>&1 | head -3
echo

echo "=== python-system-loader plugin locations ==="
find /usr -maxdepth 6 -name "libgz-sim-python-system-loader-system.so*" 2>/dev/null
echo

echo "=== Active conda / pyenv (if any) ==="
if command -v conda >/dev/null 2>&1; then
    conda info --envs 2>/dev/null | head -5
fi
if command -v pyenv >/dev/null 2>&1; then
    pyenv version 2>/dev/null
fi
