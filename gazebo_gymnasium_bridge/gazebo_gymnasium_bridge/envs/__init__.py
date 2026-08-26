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
"""Generalized multi-agent Gymnasium VecEnv for Gazebo.

There is a single env path: ``MultiAgentGazeboVecEnv``, parameterized by an
``AgentSpec`` and built for N agents in one gz world. Single-agent training
is just ``n_agents=1`` — there are no per-model or single-agent env classes.

    from gazebo_gymnasium_bridge.envs import make_multi
    env = make_multi("cartpole", n_agents=16)
"""

from .agent_spec import AgentSpec
from .agent_spec import forward_progress_reward
from .agent_spec import get_spec
from .agent_spec import JointObs
from .agent_spec import planar_health_termination
from .agent_spec import pos_then_vel_obs
from .agent_spec import proportional_forces
from .agent_spec import proportional_velocities
from .agent_spec import register_spec
from .agent_spec import registered_specs
from .agent_spec import uniform_reset
from .gym_env import GazeboEnv
from .gym_env import register_envs
from .gym_vector_env import GazeboVectorEnv
from .harness_vec_env import HarnessVecEnv
from .harness_vec_env import make_harness
from .inprocess_vec_env import InProcessHarnessVecEnv
from .inprocess_vec_env import make_inprocess
from .multi_agent_env import make_multi
from .multi_agent_env import MultiAgentGazeboVecEnv
from .multi_cartpole import MultiCartPoleVecEnv
from .obs_wrap import is_image_space
from .obs_wrap import wrap_for_observations

# Register the gymnasium ids (GazeboCartPole-v0, ...) on import, so
# `import gazebo_gymnasium_bridge; gymnasium.make("GazeboCartPole-v0")` works.
register_envs()

__all__ = [
    "MultiAgentGazeboVecEnv",
    "make_multi",
    "HarnessVecEnv",
    "make_harness",
    "InProcessHarnessVecEnv",
    "make_inprocess",
    "GazeboEnv",
    "GazeboVectorEnv",
    "register_envs",
    "is_image_space",
    "wrap_for_observations",
    "AgentSpec",
    "JointObs",
    "get_spec",
    "register_spec",
    "registered_specs",
    "proportional_forces",
    "proportional_velocities",
    "pos_then_vel_obs",
    "uniform_reset",
    "forward_progress_reward",
    "planar_health_termination",
    "MultiCartPoleVecEnv",
]
