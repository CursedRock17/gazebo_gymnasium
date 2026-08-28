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
"""gazebo_gymnasium_bridge — Gymnasium envs that drive a Gazebo (gz sim) world.

Per REP-2000 we target ROS 2 Jazzy on Ubuntu Noble 24.04 (Tier 1). The
package may still import on other distros — none of the Python code is
ROS-specific — but launch files, message types, and the gz transport
bindings are version-coupled. We surface a warning if `ROS_DISTRO` is
set to anything other than the supported value so users don't waste
time debugging weird version-mismatch errors.
"""

import os
import warnings

# Supported ROS 2 distributions per REP-2000. Add new entries when we
# explicitly test against them.
_SUPPORTED_ROS_DISTROS = {"jazzy"}


def _check_ros_distro() -> None:
    distro = os.environ.get("ROS_DISTRO")
    if distro is None:
        # No ROS sourced — env tests / docs builds run here. Stay quiet.
        return
    if distro not in _SUPPORTED_ROS_DISTROS:
        warnings.warn(
            f"gazebo_gymnasium_bridge is targeted at ROS 2 distributions "
            f"{sorted(_SUPPORTED_ROS_DISTROS)} (Tier 1 per REP-2000); "
            f"you have ROS_DISTRO={distro!r}. Things may still work, "
            "but launch files, message types, and gz-transport bindings "
            "are version-coupled — file an issue if you hit trouble.",
            RuntimeWarning,
            stacklevel=2,
        )


_check_ros_distro()

# Registering the gymnasium ids (GazeboCartPole-v0, ...) makes
# `import gazebo_gymnasium_bridge; gymnasium.make("GazeboCartPole-v0")` work.
# Importing envs pulls in the gz transport bindings; skip gracefully where
# they're absent (e.g. a docs build) rather than breaking the import.
try:
    from . import envs  # noqa: F401  (side effect: registers env ids)
except ImportError:
    pass
