# Copyright 2015 Open Source Robotics Foundation, Inc.
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

from ament_pep257.main import main
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(_HERE, os.pardir, "gazebo_gymnasium_bridge")


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    # Keep this consistent with ament_flake8.ini: ROS 2 style puts the
    # docstring summary on the FIRST line (D212), so ignore D213 (second-line
    # summary) plus the numpy-style section checks (D407/D413/D416) the default
    # 'ament' convention would otherwise enforce. Without this the flake8 and
    # pep257 conventions contradict each other. '--' ends option parsing so the
    # paths aren't swallowed by --add-ignore's nargs='+'.
    rc = main(argv=["--convention", "ament",
                    "--add-ignore", "D213", "D407", "D413", "D416",
                    "--", _PKG, _HERE])
    assert rc == 0, "Found code style errors / warnings"
