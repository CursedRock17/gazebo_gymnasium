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

"""Observation-aware SB3 wrapping shared by the training entry points.

State observations pass through with ``MlpPolicy``. Image observations (uint8
H×W×C) get frame stacking (a single frame carries no motion information) and
channel-first transposition, with ``CnnPolicy`` — so
``train.py --agent line_follower`` just works.
"""

import numpy as np


def is_image_space(space) -> bool:
    """Return whether an observation space is an uint8 image (H, W, C)."""
    return (len(getattr(space, "shape", ())) == 3
            and getattr(space, "dtype", None) == np.uint8)


def wrap_for_observations(vec_env, frame_stack: int = 4):
    """Return (wrapped_env, policy_name) appropriate for the obs space.

    frame_stack applies to image observations only (use 1 to disable).
    """
    if not is_image_space(vec_env.observation_space):
        return vec_env, "MlpPolicy"
    from stable_baselines3.common.vec_env import VecFrameStack
    from stable_baselines3.common.vec_env import VecTransposeImage
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return VecTransposeImage(vec_env), "CnnPolicy"
