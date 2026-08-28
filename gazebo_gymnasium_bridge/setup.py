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

from setuptools import find_packages
from setuptools import setup

package_name = "gazebo_gymnasium_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Lucas Wendland",
    author_email="lwendlan@umd.edu",
    maintainer="Lucas Wendland",
    maintainer_email="lwendlan@umd.edu",
    description="Gymnasium env classes that drive a Gazebo (gz sim) world over gz-transport.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        # No console scripts. The package exposes Gymnasium env classes
        # (imported via `from gazebo_gymnasium_bridge.envs import ...`)
        # and training scripts live under training_scripts/ at the project
        # root, not as installed entry points.
        "console_scripts": [],
    },
)
