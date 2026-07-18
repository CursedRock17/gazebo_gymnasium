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
from dataclasses import replace
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
    # (defaulted fields below) bare_model_uri: geometry-only model for the ECM
    # backends (harness / in-process) — no controllers, no effort limit. Falls
    # back to model_uri when empty.
    action_space: spaces.Space
    joint_obs: Sequence[JointObs]
    reward_fn: Callable[[np.ndarray, object], float]
    terminated_fn: Callable[[np.ndarray], bool]
    bare_model_uri: str = ""
    # Population-based dynamics randomization: fraction by which each agent's
    # mass + inertia is scaled, sampled per agent as U(1-x, 1+x) when the world
    # is built (0.0 = off). With N agents in one world this samples N points
    # from the dynamics distribution, so a policy trained across them is robust
    # to mass error — cheap domain randomization for sim-to-real, no ECM needed.
    mass_randomization: float = 0.0
    # Control-authority randomization: fraction by which each agent's actuator
    # command (velocity/force) is scaled, sampled per agent as U(1-x, 1+x)
    # (0.0 = off). Models real actuator-gain uncertainty and, unlike mass, bites
    # even under velocity control. Reproducible from the construction seed.
    action_gain_randomization: float = 0.0
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


# N pushed on the slider (bang-bang force), the classic CartPole actuation.
# The pole is genuinely unstable under force control: a constant push tips it
# in ~3 steps and a random policy survives ~7 steps (median 6), while a trained
# PPO policy reaches the 500-step cap — a textbook benchmark. NOTE: force
# actuation requires the model to spawn CLEAR of the ground (see spawn_z below);
# a cart resting on the ground plane is pinned by contact friction, which
# velocity control would silently override (kinematic) but force cannot.
_CART_FORCE = 10.0


def _cartpole_action_to_commands(action):
    # Discrete 1 -> +F, 0 -> -F, applied as a slider force command.
    # Robust to a scalar (offline) or a length-1 array (the harness passes each
    # agent's action as a row of the (n_agents, act_dim) matrix).
    a = int(round(float(np.ravel(action)[0])))
    f = _CART_FORCE if a == 1 else -_CART_FORCE
    return [("slider_to_cart", "force", f)]


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
        bare_model_uri="package://gazebo_gymnasium_resources/models/cartpole_bare",
        observation_space=obs_space,
        action_space=spaces.Discrete(2),
        # obs order: cart_pos, cart_vel, pole_angle, pole_ang_vel
        joint_obs=(JointObs("slider_to_cart"), JointObs("cart_to_pole")),
        reward_fn=lambda obs, action: 1.0,
        terminated_fn=lambda obs: bool(
            abs(obs[2]) > _POLE_ANGLE_THRESHOLD
            or abs(obs[0]) > _CART_POSITION_THRESHOLD
        ),
        # Clear of the ground plane: the cart's collision box must NOT rest on
        # the ground or contact friction pins it against force actuation.
        spawn_z=0.60,
        extra_joints=(("world_to_slider", "world", "slider"),),
        action_to_commands=_cartpole_action_to_commands,
        reset_joint_state=_cartpole_reset_joint_state,
    )


def _cartpole_continuous_action_to_commands(action):
    # Box(-1, 1) -> proportional slider force in [-_CART_FORCE, +_CART_FORCE].
    # The continuous analog of the discrete bang-bang spec (same model, same
    # dynamics), mirroring Gymnasium's MuJoCo InvertedPendulum.
    a = float(np.clip(np.ravel(action)[0], -1.0, 1.0))
    return [("slider_to_cart", "force", a * _CART_FORCE)]


def _cartpole_continuous_spec() -> AgentSpec:
    """Continuous-force cartpole: exercises Box actions (SAC/TD3/DDPG)."""
    return replace(
        _cartpole_spec(),
        name="cartpole_continuous",
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,),
                                dtype=np.float32),
        action_to_commands=_cartpole_continuous_action_to_commands,
    )


# --------------------------------------------------------------------------- #
# InvertedDoublePendulum (MuJoCo port): cart + two hinged 0.6 m poles.
# Obs layout: [cart_pos, cart_vel, th1, w1, th2, w2]  (th2 relative to pole 1).
# --------------------------------------------------------------------------- #

_IDP_POLE_LEN = 0.6
_IDP_TIP_MAX = 2 * _IDP_POLE_LEN
_IDP_TIP_THRESHOLD = 1.0     # terminate when the tip drops below this height
_IDP_FORCE = 100.0           # N full-scale slider force for a Box(-1,1) action


def _idp_tip(obs):
    """Tip (x, height) from joint state — x absolute, height cart-relative."""
    th1, th2 = float(obs[2]), float(obs[4])
    x = float(obs[0]) + _IDP_POLE_LEN * (np.sin(th1) + np.sin(th1 + th2))
    h = _IDP_POLE_LEN * (np.cos(th1) + np.cos(th1 + th2))
    return x, h


def _idp_reward(obs, action):
    # Shaped like MuJoCo's: alive bonus minus tip-distance and velocity
    # penalties (coefficients tuned to this model's scale, documented variant).
    x, h = _idp_tip(obs)
    w1, w2 = float(obs[3]), float(obs[5])
    return float(10.0 - 0.01 * x * x - 10.0 * (_IDP_TIP_MAX - h) ** 2
                 - 1e-3 * (w1 * w1 + w2 * w2))


def _idp_action_to_commands(action):
    a = float(np.clip(np.ravel(action)[0], -1.0, 1.0))
    return [("slider_to_cart", "force", a * _IDP_FORCE)]


def _idp_reset_joint_state(rng):
    return {
        "slider_to_cart": (0.0, 0.0),
        "cart_to_pole": (float(rng.uniform(-0.05, 0.05)), 0.0),
        "pole_to_pole2": (float(rng.uniform(-0.05, 0.05)), 0.0),
    }


def _inverted_double_pendulum_spec() -> AgentSpec:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,),
                           dtype=np.float32)
    return AgentSpec(
        name="inverted_double_pendulum",
        model_uri=("package://gazebo_gymnasium_resources/models/"
                   "inverted_double_pendulum_bare"),
        bare_model_uri=("package://gazebo_gymnasium_resources/models/"
                        "inverted_double_pendulum_bare"),
        observation_space=obs_space,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,),
                                dtype=np.float32),
        joint_obs=(JointObs("slider_to_cart"), JointObs("cart_to_pole"),
                   JointObs("pole_to_pole2")),
        reward_fn=_idp_reward,
        terminated_fn=lambda obs: bool(_idp_tip(obs)[1] <= _IDP_TIP_THRESHOLD),
        spawn_z=0.60,        # force actuation: keep the cart off the ground
        max_episode_steps=1000,
        extra_joints=(("world_to_slider", "world", "slider"),),
        action_to_commands=_idp_action_to_commands,
        reset_joint_state=_idp_reset_joint_state,
    )


# --------------------------------------------------------------------------- #
# Hopper (MuJoCo port): planar monopod, root modeled as joints (MuJoCo-style),
# forward = +Y. Obs (11) = [z_disp, pitch, thigh, leg, foot,
#                           v_fwd, v_z, v_pitch, v_thigh, v_leg, v_foot].
# --------------------------------------------------------------------------- #

_HOPPER_TORQUE = 200.0        # N*m full-scale per actuated joint (MuJoCo gear)
_HOPPER_SPAWN_Z = 1.25        # torso-root height at spawn (MuJoCo initial z)
_HOPPER_MIN_Z = 0.7           # unhealthy below this absolute torso height
_HOPPER_MAX_PITCH = 0.2       # rad, unhealthy beyond


def _hopper_action_to_commands(action):
    # np.resize pads scalar probes to 3; real actions come in as (3,).
    a = np.resize(np.clip(np.asarray(action, dtype=float).ravel(),
                          -1.0, 1.0), 3)
    return [("thigh_joint", "force", float(a[0]) * _HOPPER_TORQUE),
            ("leg_joint", "force", float(a[1]) * _HOPPER_TORQUE),
            ("foot_joint", "force", float(a[2]) * _HOPPER_TORQUE)]


def _hopper_reward(obs, action):
    # forward velocity + alive bonus - control cost (MuJoCo weights).
    a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
    return float(obs[5] + 1.0 - 1e-3 * float(np.square(a).sum()))


def _hopper_terminated(obs):
    z = _HOPPER_SPAWN_Z + float(obs[0])
    healthy = (z > _HOPPER_MIN_Z
               and abs(float(obs[1])) < _HOPPER_MAX_PITCH
               and bool(np.all(np.abs(obs) < 100.0)))
    return not healthy


def _hopper_reset_joint_state(rng):
    def d():
        return (float(rng.uniform(-0.005, 0.005)),
                float(rng.uniform(-0.005, 0.005)))
    return {j: d() for j in ("root_fwd", "root_up", "root_pitch",
                             "thigh_joint", "leg_joint", "foot_joint")}


def _hopper_spec() -> AgentSpec:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,),
                           dtype=np.float32)
    return AgentSpec(
        name="hopper",
        model_uri="package://gazebo_gymnasium_resources/models/hopper_bare",
        bare_model_uri=("package://gazebo_gymnasium_resources/models/"
                        "hopper_bare"),
        observation_space=obs_space,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(3,),
                                dtype=np.float32),
        # MuJoCo layout: positions (root x excluded), then all velocities.
        joint_obs=(
            JointObs("root_up", velocity=False),
            JointObs("root_pitch", velocity=False),
            JointObs("thigh_joint", velocity=False),
            JointObs("leg_joint", velocity=False),
            JointObs("foot_joint", velocity=False),
            JointObs("root_fwd", position=False),
            JointObs("root_up", position=False),
            JointObs("root_pitch", position=False),
            JointObs("thigh_joint", position=False),
            JointObs("leg_joint", position=False),
            JointObs("foot_joint", position=False),
        ),
        reward_fn=_hopper_reward,
        terminated_fn=_hopper_terminated,
        spawn_z=_HOPPER_SPAWN_Z,
        frame_skip=4,
        max_episode_steps=1000,
        extra_joints=(("world_to_anchor", "world", "anchor"),),
        action_to_commands=_hopper_action_to_commands,
        reset_joint_state=_hopper_reset_joint_state,
    )


_SPEC_FACTORIES = {
    "cartpole": _cartpole_spec,
    "cartpole_continuous": _cartpole_continuous_spec,
    "inverted_double_pendulum": _inverted_double_pendulum_spec,
    "hopper": _hopper_spec,
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
