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

"""Per-agent specification for the generalized multi-agent Gazebo VecEnv.

One ``AgentSpec`` fully describes a *single* agent: which model to spawn,
how to turn its sensor topics into an observation vector, the action /
observation spaces, and the reward / termination rules. The
``MultiAgentGazeboVecEnv`` then runs N identical agents from one spec in a
single Gazebo world — so "MultiCartPole", "MultiAnt", etc. are just named
specs, not separate env classes.

This module is deliberately free of any ``gz.*`` import: it operates on
duck-typed messages (anything exposing ``msg.joint[k].name`` /
``.axis1.position`` / ``.axis1.velocity``). That keeps the spec layer unit
testable without a running simulator or the native bindings.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Optional
from typing import Sequence

from gymnasium import spaces
import numpy as np


@dataclass(frozen=True)
class JointObs:
    """Pull one joint's position and/or velocity into the observation vector.

    The order of fields in the final observation follows the order of the
    ``JointObs`` entries in ``AgentSpec.joint_obs``; within one entry,
    position precedes velocity.
    """

    joint: str
    position: bool = True
    velocity: bool = True

    @property
    def width(self) -> int:
        return int(self.position) + int(self.velocity)


@dataclass
class AgentSpec:
    """Everything the VecEnv needs to run N copies of one agent.

    Parameters
    ----------
    name:
        Base model name; agent ``i`` spawns as ``f"{name}_{i}"``.
    model_uri:
        SDF ``<uri>`` merged into each spawned model wrapper.
    observation_space / action_space:
        Per-agent Gymnasium spaces (SB3 batches across agents).
    joint_obs:
        Ordered joint reads that build the observation from a model's
        ``joint_state`` message. The summed widths must equal
        ``observation_space.shape[0]``.
    reward_fn(obs, action) -> float:
        Per-agent reward for a step.
    terminated_fn(obs) -> bool:
        Per-agent natural-termination test (e.g. pole fell).
    spawn_z / x_spacing / y_spacing:
        Spawn geometry. ``spawn_z`` must lift the model clear of the ground
        plane (z=0 buries it and contact friction pins the joints).
    """

    name: str
    model_uri: str
    observation_space: spaces.Space
    action_space: spaces.Space
    joint_obs: Sequence[JointObs]
    reward_fn: Callable[[np.ndarray, object], float]
    terminated_fn: Callable[[np.ndarray], bool]
    spawn_z: float = 0.10
    x_spacing: float = 3.0
    y_spacing: float = 8.5
    frame_skip: int = 5
    max_episode_steps: int = 500
    self_collide: bool = True
    # Extra SDF snippet spliced into the model wrapper (e.g. a world-fixed
    # joint pinning a rail). Kept generic so each model supplies its own.
    extra_sdf: str = ""
    extra_joints: tuple = field(default_factory=tuple)

    # --- In-sim harness (ECM) actuation + reset ---
    # action_to_commands(action) -> [(joint_name, mode, value), ...] with mode
    # in {"velocity", "force", "position"}. The in-sim harness applies these
    # via ECM (Joint.set_velocity / set_force / reset_position). None means
    # this spec isn't ECM-actuatable yet (harness path unavailable).
    action_to_commands: Optional[Callable[[object], Sequence]] = None
    # reset_joint_state(rng) -> {joint_name: (position, velocity)} set via ECM
    # at episode start — in place, no respawn. This is where clean per-agent
    # randomization lives (e.g. a small random initial pole angle), since ECM
    # reset_position sets the joint coordinate directly.
    reset_joint_state: Optional[Callable] = None

    def __post_init__(self):
        obs_dim = self.observation_space.shape[0]
        joint_dim = sum(j.width for j in self.joint_obs)
        if joint_dim != obs_dim:
            raise ValueError(
                f"AgentSpec({self.name!r}): joint_obs widths sum to "
                f"{joint_dim} but observation_space has {obs_dim} dims"
            )
        if self.spawn_z <= 0.0:
            raise ValueError(
                f"AgentSpec({self.name!r}): spawn_z must be > 0 to clear the "
                f"ground plane (got {self.spawn_z})"
            )

    @property
    def obs_dim(self) -> int:
        return int(self.observation_space.shape[0])

    def obs_from_joint_state(self, model_msg) -> np.ndarray:
        """Build the observation vector from a gz ``Model`` joint_state msg.

        Duck-typed: ``model_msg.joint`` is iterated; each element must expose
        ``.name`` and ``.axis1.position`` / ``.axis1.velocity``. Missing
        joints contribute zeros (matches the original cartpole behavior, and
        keeps a dropped message from raising mid-episode).
        """
        by_name = {j.name: j for j in model_msg.joint}
        out = np.zeros(self.obs_dim, dtype=np.float32)
        k = 0
        for spec in self.joint_obs:
            j = by_name.get(spec.joint)
            if spec.position:
                out[k] = float(j.axis1.position) if j is not None else 0.0
                k += 1
            if spec.velocity:
                out[k] = float(j.axis1.velocity) if j is not None else 0.0
                k += 1
        return out


# --------------------------------------------------------------------------- #
# Built-in agent specs (the registry the make_multi() factory draws from)
# --------------------------------------------------------------------------- #

_POLE_ANGLE_THRESHOLD = 0.20944
_CART_POSITION_THRESHOLD = 2.4


_CART_SPEED = 1.0  # m/s commanded on the slider joint for the discrete action


def _cartpole_action_to_commands(action):
    # Discrete 1 -> +v, 0 -> -v, applied as a slider velocity command.
    # Robust to a scalar (offline) or a length-1 array (the harness passes each
    # agent's action as a row of the (n_agents, act_dim) matrix).
    a = int(round(float(np.ravel(action)[0])))
    v = _CART_SPEED if a == 1 else -_CART_SPEED
    return [("slider_to_cart", "velocity", v)]


def _cartpole_reset_joint_state(rng):
    # Cart centered, pole at a small random angle (standard CartPole ~+/-0.05).
    # Set directly via ECM — no free-fall/respawn, so it's exact per agent.
    return {
        "slider_to_cart": (0.0, 0.0),
        "cart_to_pole": (float(rng.uniform(-0.05, 0.05)), 0.0),
    }


def _cartpole_spec() -> AgentSpec:
    obs_space = spaces.Box(
        low=np.array([-4.8, -np.inf, -0.41887903, -np.inf], dtype=np.float32),
        high=np.array([4.8, np.inf, 0.41887903, np.inf], dtype=np.float32),
        shape=(4,),
        dtype=np.float32,
    )
    return AgentSpec(
        name="cartpole",
        model_uri="package://gazebo_gymnasium_resources/models/cartpole",
        observation_space=obs_space,
        action_space=spaces.Discrete(2),
        # obs order: cart_pos, cart_vel, pole_angle, pole_ang_vel
        joint_obs=(JointObs("slider_to_cart"), JointObs("cart_to_pole")),
        reward_fn=lambda obs, action: 1.0,
        terminated_fn=lambda obs: bool(
            abs(obs[2]) > _POLE_ANGLE_THRESHOLD
            or abs(obs[0]) > _CART_POSITION_THRESHOLD
        ),
        spawn_z=0.10,
        extra_joints=(("world_to_slider", "world", "slider"),),
        action_to_commands=_cartpole_action_to_commands,
        reset_joint_state=_cartpole_reset_joint_state,
    )


_SPEC_FACTORIES = {
    "cartpole": _cartpole_spec,
}


def get_spec(name: str) -> AgentSpec:
    """Return a fresh AgentSpec for a registered agent name."""
    if name not in _SPEC_FACTORIES:
        raise KeyError(
            f"unknown agent spec {name!r}; registered: "
            f"{sorted(_SPEC_FACTORIES)}"
        )
    return _SPEC_FACTORIES[name]()


def register_spec(name: str, factory: Callable[[], AgentSpec]) -> None:
    """Register a new named agent spec (factory returning a fresh AgentSpec)."""
    _SPEC_FACTORIES[name] = factory


def registered_specs() -> list:
    return sorted(_SPEC_FACTORIES)
