# Copyright 2017 Open Source Robotics Foundation, Inc.
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

import os

from ament_flake8.main import main_with_errors
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(_HERE, os.pardir, "gazebo_gymnasium_bridge")
# Project flake8 config (double quotes, google import order, ROS D-code
# ignores). main_with_errors(argv=[]) would not discover it, and '.' would
# depend on the CWD, so pass both explicitly and stay CWD-independent.
_CONFIG = os.path.join(_HERE, os.pardir, os.pardir, "ament_flake8.ini")


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    rc, errors = main_with_errors(argv=["--config", _CONFIG, _PKG, _HERE])
    assert rc == 0, \
        "Found %d code style errors / warnings:\n" % len(errors) + \
        "\n".join(errors)
