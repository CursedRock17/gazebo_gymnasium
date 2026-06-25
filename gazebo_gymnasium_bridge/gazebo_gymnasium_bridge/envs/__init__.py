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

"""Gymnasium env classes that talk to a running gz sim world."""

from .agent_spec import AgentSpec
from .agent_spec import JointObs
from .agent_spec import get_spec
from .agent_spec import register_spec
from .agent_spec import registered_specs
from .cartpole import GazeboCartPoleEnv
from .inverted_double_pendulum import GazeboInvertedDoublePendulumEnv
from .inverted_pendulum import GazeboInvertedPendulumEnv
from .line_follower import GazeboLineFollowerEnv
from .multi_agent_env import MultiAgentGazeboVecEnv
from .multi_agent_env import make_multi
from .multi_cartpole import MultiCartPoleVecEnv
from .point import GazeboPointEnv
from .reacher import GazeboReacherEnv

__all__ = [
    # Generalized multi-agent VecEnv (the path forward)
    "MultiAgentGazeboVecEnv",
    "make_multi",
    "AgentSpec",
    "JointObs",
    "get_spec",
    "register_spec",
    "registered_specs",
    "MultiCartPoleVecEnv",
    # Legacy single-agent envs (slated for removal — use n_agents=1)
    "GazeboCartPoleEnv",
    "GazeboInvertedPendulumEnv",
    "GazeboInvertedDoublePendulumEnv",
    "GazeboLineFollowerEnv",
    "GazeboReacherEnv",
    "GazeboPointEnv",
]
