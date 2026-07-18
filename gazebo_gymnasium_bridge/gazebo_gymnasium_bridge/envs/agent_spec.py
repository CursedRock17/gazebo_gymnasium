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
    # --- Image observations (vision-in-the-loop) ---
    # (H, W, C) when the observation is the agent's onboard camera instead of
    # joint state. The world builder then loads the render/sensors system,
    # rewrites the model's camera <topic> to /rl/camera_<i> per agent, and the
    # in-process env reads frames from those topics. joint_obs may be empty.
    image_obs: Optional[tuple] = None
    # Static scenery included once per agent at that agent's spawn offset
    # (e.g. each rover gets its own line track). Any package:// model URI.
    per_agent_include_uri: str = ""
    # If True the whole model's world pose is restored to its spawn pose on
    # reset (mobile bases aren't world-pinned, so joint resets alone won't
    # bring them home).
    reset_model_pose: bool = False
    # Population-based dynamics randomization: fraction by which each agent's
    # mass + inertia is scaled, sampled per agent as U(1-x, 1+x) when the world
    # is built (0.0 = off). With N agents in one world this samples N points
    # from the dynamics distribution, so a policy trained across them is robust
    # to mass error — cheap domain randomization for sim-to-real, no ECM needed.
    spawn_y: float = 0.0
    spawn_yaw: float = 0.0
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
        if self.image_obs is not None:
            if tuple(self.observation_space.shape) != tuple(self.image_obs):
                raise ValueError(
                    f"AgentSpec({self.name!r}): observation_space shape "
                    f"{self.observation_space.shape} != image_obs "
                    f"{self.image_obs}")
        else:
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


# --------------------------------------------------------------------------- #
# Walker2d (MuJoCo port): planar biped — the hopper recipe with two legs.
# Obs (17) = [z, pitch, 6 leg angles | v_fwd, v_z, v_pitch, 6 leg velocities].
# --------------------------------------------------------------------------- #

_WALKER_TORQUE = 100.0
_WALKER_SPAWN_Z = 1.25
_WALKER_Z_RANGE = (0.8, 2.0)
_WALKER_MAX_PITCH = 1.0
_WALKER_LEG_JOINTS = ("thigh_joint", "leg_joint", "foot_joint",
                      "thigh_left_joint", "leg_left_joint", "foot_left_joint")


def _walker_action_to_commands(action):
    a = np.resize(np.clip(np.asarray(action, dtype=float).ravel(),
                          -1.0, 1.0), 6)
    return [(j, "force", float(a[i]) * _WALKER_TORQUE)
            for i, j in enumerate(_WALKER_LEG_JOINTS)]


def _walker_reward(obs, action):
    a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
    return float(obs[8] + 1.0 - 1e-3 * float(np.square(a).sum()))


def _walker_terminated(obs):
    z = _WALKER_SPAWN_Z + float(obs[0])
    healthy = (_WALKER_Z_RANGE[0] < z < _WALKER_Z_RANGE[1]
               and abs(float(obs[1])) < _WALKER_MAX_PITCH
               and bool(np.all(np.abs(obs) < 100.0)))
    return not healthy


def _walker_reset_joint_state(rng):
    def d():
        return (float(rng.uniform(-0.005, 0.005)),
                float(rng.uniform(-0.005, 0.005)))
    return {j: d() for j in (("root_fwd", "root_up", "root_pitch")
                             + _WALKER_LEG_JOINTS)}


def _walker2d_spec() -> AgentSpec:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,),
                           dtype=np.float32)
    pos = tuple(JointObs(j, velocity=False)
                for j in ("root_up", "root_pitch") + _WALKER_LEG_JOINTS)
    vel = tuple(JointObs(j, position=False)
                for j in ("root_fwd", "root_up", "root_pitch")
                + _WALKER_LEG_JOINTS)
    return AgentSpec(
        name="walker2d",
        model_uri="package://gazebo_gymnasium_resources/models/walker2d_bare",
        bare_model_uri=("package://gazebo_gymnasium_resources/models/"
                        "walker2d_bare"),
        observation_space=obs_space,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(6,),
                                dtype=np.float32),
        joint_obs=pos + vel,
        reward_fn=_walker_reward,
        terminated_fn=_walker_terminated,
        spawn_z=_WALKER_SPAWN_Z,
        frame_skip=4,
        max_episode_steps=1000,
        extra_joints=(("world_to_anchor", "world", "anchor"),),
        action_to_commands=_walker_action_to_commands,
        reset_joint_state=_walker_reset_joint_state,
    )


# --------------------------------------------------------------------------- #
# HalfCheetah (MuJoCo port): planar, horizontal torso, back/front legs.
# No health termination — episodes end on truncation only (MuJoCo semantics).
# Obs (17) = [z, pitch, 6 leg angles | v_fwd, v_z, v_pitch, 6 leg velocities].
# --------------------------------------------------------------------------- #

_CHEETAH_SPAWN_Z = 0.77
_CHEETAH_JOINTS = ("bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot")
_CHEETAH_GEARS = (120.0, 90.0, 60.0, 120.0, 60.0, 30.0)


def _cheetah_action_to_commands(action):
    a = np.resize(np.clip(np.asarray(action, dtype=float).ravel(),
                          -1.0, 1.0), 6)
    return [(j, "force", float(a[i]) * _CHEETAH_GEARS[i])
            for i, j in enumerate(_CHEETAH_JOINTS)]


def _cheetah_reward(obs, action):
    a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
    return float(obs[8] - 0.1 * float(np.square(a).sum()))


def _cheetah_reset_joint_state(rng):
    def d():
        return (float(rng.uniform(-0.005, 0.005)),
                float(rng.uniform(-0.005, 0.005)))
    return {j: d() for j in (("root_fwd", "root_up", "root_pitch")
                             + _CHEETAH_JOINTS)}


def _half_cheetah_spec() -> AgentSpec:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,),
                           dtype=np.float32)
    pos = tuple(JointObs(j, velocity=False)
                for j in ("root_up", "root_pitch") + _CHEETAH_JOINTS)
    vel = tuple(JointObs(j, position=False)
                for j in ("root_fwd", "root_up", "root_pitch")
                + _CHEETAH_JOINTS)
    return AgentSpec(
        name="half_cheetah",
        model_uri=("package://gazebo_gymnasium_resources/models/"
                   "half_cheetah_bare"),
        bare_model_uri=("package://gazebo_gymnasium_resources/models/"
                        "half_cheetah_bare"),
        observation_space=obs_space,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(6,),
                                dtype=np.float32),
        joint_obs=pos + vel,
        reward_fn=_cheetah_reward,
        # MuJoCo half-cheetah has no health termination; keep a pure numerical
        # guard so a solver blow-up can't silently poison training.
        terminated_fn=lambda obs: not bool(np.all(np.abs(obs) < 1000.0)),
        spawn_z=_CHEETAH_SPAWN_Z,
        frame_skip=5,
        max_episode_steps=1000,
        extra_joints=(("world_to_anchor", "world", "anchor"),),
        action_to_commands=_cheetah_action_to_commands,
        reset_joint_state=_cheetah_reset_joint_state,
    )


# --------------------------------------------------------------------------- #
# Reacher (MuJoCo port): 2-link horizontal-plane arm + goal-as-joints target.
# Obs (6) = [th0, th1, target_x, target_y, v0, v1]; dense distance reward.
# --------------------------------------------------------------------------- #

_REACHER_L0, _REACHER_L1 = 0.1, 0.11
_REACHER_TORQUE = 5.0


def _reacher_fingertip(obs):
    th0, th1 = float(obs[0]), float(obs[1])
    x = _REACHER_L0 * np.cos(th0) + _REACHER_L1 * np.cos(th0 + th1)
    y = _REACHER_L0 * np.sin(th0) + _REACHER_L1 * np.sin(th0 + th1)
    return x, y


def _reacher_reward(obs, action):
    fx, fy = _reacher_fingertip(obs)
    dist = float(np.hypot(fx - float(obs[2]), fy - float(obs[3])))
    a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
    return float(-dist - 0.1 * float(np.square(a).sum()))


def _reacher_action_to_commands(action):
    a = np.resize(np.clip(np.asarray(action, dtype=float).ravel(),
                          -1.0, 1.0), 2)
    return [("joint0", "force", float(a[0]) * _REACHER_TORQUE),
            ("joint1", "force", float(a[1]) * _REACHER_TORQUE)]


def _reacher_reset_joint_state(rng):
    # arm pose randomized like MuJoCo; the GOAL moves through the same reset
    # path because it is two prismatic joints (goal within reach, |g|<=0.198).
    return {
        "joint0": (float(rng.uniform(-0.1, 0.1)), 0.0),
        "joint1": (float(rng.uniform(-0.1, 0.1)), 0.0),
        "target_x": (float(rng.uniform(-0.14, 0.14)), 0.0),
        "target_y": (float(rng.uniform(-0.14, 0.14)), 0.0),
    }


def _reacher_spec() -> AgentSpec:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,),
                           dtype=np.float32)
    return AgentSpec(
        name="reacher",
        model_uri="package://gazebo_gymnasium_resources/models/reacher_bare",
        bare_model_uri=("package://gazebo_gymnasium_resources/models/"
                        "reacher_bare"),
        observation_space=obs_space,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,),
                                dtype=np.float32),
        joint_obs=(JointObs("joint0", velocity=False),
                   JointObs("joint1", velocity=False),
                   JointObs("target_x", velocity=False),
                   JointObs("target_y", velocity=False),
                   JointObs("joint0", position=False),
                   JointObs("joint1", position=False)),
        reward_fn=_reacher_reward,
        terminated_fn=lambda obs: not bool(np.all(np.abs(obs) < 100.0)),
        spawn_z=0.05,
        frame_skip=2,           # MuJoCo reacher dt = 0.02
        max_episode_steps=50,   # MuJoCo reacher truncates at 50
        extra_joints=(("world_to_base", "world", "base"),
                      ("world_to_target_anchor", "world", "target_anchor")),
        action_to_commands=_reacher_action_to_commands,
        reset_joint_state=_reacher_reset_joint_state,
    )


# --------------------------------------------------------------------------- #
# Line follower: differential-drive rover, onboard 64x64 camera as the WHOLE
# observation. Reward/termination are computed from the image itself (line
# centroid in the lower half), so the env is self-contained vision-in-the-loop.
# --------------------------------------------------------------------------- #

_LF_IMAGE = (64, 64, 3)
_LF_WHEEL_SPEED = 15.0        # rad/s full scale (~0.5 m/s at r=0.034)
_LF_DARK = 60                 # a pixel is "line" when max(R,G,B) < this


def _lf_line_centroid(img):
    """Return the x-centroid (0..1) of dark pixels in the lower half, or None."""
    img = np.asarray(img)
    bottom = img[img.shape[0] // 2:, :, :]
    mask = bottom.max(axis=2) < _LF_DARK
    if not mask.any():
        return None
    xs = np.nonzero(mask)[1]
    return float(xs.mean()) / (bottom.shape[1] - 1)


def _lf_reward(obs, action):
    c = _lf_line_centroid(obs)
    if c is None:
        return 0.0
    a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
    centered = 1.0 - 2.0 * abs(c - 0.5)
    forward = float(np.resize(a, 2).mean())     # same-sign commands = forward
    return float(centered + 0.5 * forward)


def _lf_action_to_commands(action):
    a = np.resize(np.clip(np.asarray(action, dtype=float).ravel(),
                          -1.0, 1.0), 2)
    return [("left_axle", "velocity", float(a[0]) * _LF_WHEEL_SPEED),
            ("right_axle", "velocity", float(a[1]) * _LF_WHEEL_SPEED)]


def _lf_reset_joint_state(rng):
    return {"left_axle": (0.0, 0.0), "right_axle": (0.0, 0.0)}


def _line_follower_spec() -> AgentSpec:
    return AgentSpec(
        name="line_follower",
        model_uri="package://gazebo_gymnasium_resources/models/rover_bare",
        bare_model_uri=("package://gazebo_gymnasium_resources/models/"
                        "rover_bare"),
        observation_space=spaces.Box(low=0, high=255, shape=_LF_IMAGE,
                                     dtype=np.uint8),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,),
                                dtype=np.float32),
        joint_obs=(),
        image_obs=_LF_IMAGE,
        per_agent_include_uri=("package://gazebo_gymnasium_resources/models/"
                               "line_track"),
        reward_fn=_lf_reward,
        terminated_fn=lambda obs: _lf_line_centroid(obs) is None,
        # the old known-good start pose: on the y=-1 straight, aligned with it
        spawn_y=-1.0,
        spawn_yaw=1.5708,
        spawn_z=0.085,
        x_spacing=6.0,          # each agent gets its own ~2.4 m track loop
        frame_skip=5,
        max_episode_steps=300,
        reset_model_pose=True,  # mobile base: restore chassis pose on reset
        action_to_commands=_lf_action_to_commands,
        reset_joint_state=_lf_reset_joint_state,
    )


_SPEC_FACTORIES = {
    "cartpole": _cartpole_spec,
    "cartpole_continuous": _cartpole_continuous_spec,
    "inverted_double_pendulum": _inverted_double_pendulum_spec,
    "hopper": _hopper_spec,
    "walker2d": _walker2d_spec,
    "half_cheetah": _half_cheetah_spec,
    "reacher": _reacher_spec,
    "line_follower": _line_follower_spec,
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
