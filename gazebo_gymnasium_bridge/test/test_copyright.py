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

from ament_copyright.main import main
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(_HERE, os.pardir, "gazebo_gymnasium_bridge")


@pytest.mark.copyright
@pytest.mark.linter
def test_copyright():
    # Every source file carries the Apache-2.0 header. Explicit paths keep this
    # CWD-independent (bare '.' would also scan build/venv artifacts).
    rc = main(argv=[_PKG, _HERE])
    assert rc == 0, "Found errors"
