# Creating Your Own Agent

This guide shows how to add a brand-new reinforcement-learning environment to
Gazebo Gymnasium and train it. The whole library runs on one dataclass,
`AgentSpec`, describing a single agent through its model, observation,
action, reward, and termination, while the framework runs N identical
copies of it in one Gazebo world as a
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) `VecEnv`.
There is no per-environment Python class to write. "MultiCartPole,"
"MultiAnt," and "MultiHopper" are just named specifications.

```python
from gazebo_gymnasium_bridge.envs import make_inprocess

# 16 agents share one simulation, with no separate launch step.
vec_env = make_inprocess("cartpole", n_agents=16)
```

Registered specifications also expose themselves as a standard
`gymnasium.Env`, so any Gymnasium-speaking tool, RLlib, CleanRL, Tianshou,
or TorchRL among them, works out of the box. Importing the package alone
registers the ids.

```python
import gazebo_gymnasium_bridge          # registers GazeboCartPole-v0, and others
import gymnasium as gym

env = gym.make("GazeboCartPole-v0")     # a standard 5-tuple step, (obs, info) reset
```

Two worked tutorials accompany this guide: [CartPole](examples/cartpole.md),
the reference environment, and
[Porting Hopper, annotated](examples/porting_hopper.md), the narrated story
of a real port, including how each physics trap was diagnosed from headless
probes. The complete reference implementation is the cartpole specification
in
[`gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/agent_spec.py`](../gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/agent_spec.py).
Copying it and changing the parts described below is the fastest way to
start.

Six steps take a new agent from nothing to a trained policy: writing the
model's Simulation Description Format (SDF), defining the `AgentSpec`,
registering it, testing it offline, wiring a world and launch file, and
finally running it.

## The Three Backends

Everything written below, the specification and the model SDF, stays
shared across backends. What differs is how the environment talks to
Gazebo, chosen with `make_inprocess`, `make_harness`, or `make_multi` (or
`train.py --backend`).

| | In-Process (Default) | Harness | Per-Agent |
|---|---|---|---|
| Simulation location | In the training process (`TestFixture`) | Launched `gz sim` | Launched `gz sim` |
| Transport | None, direct Entity Component Manager (ECM) access | 3 topics total, O(1) in N | 2 topics per agent |
| Reset | In place via the ECM | In place via the ECM | Delete and re-spawn |
| Needs a launch? | No, one `python` command | Yes, `ros2 launch` | Yes, `ros2 launch` |
| Best for | Training and continuous integration, fastest | A running or GUI simulation to watch | Small N, quick starts |
| Model | Bare, no controllers | Bare, no controllers | Model with controllers |

All three backends stay spec-driven and share the same `reward_fn`,
`terminated_fn`, and group-auto-reset logic. Starting with `inprocess` makes
the most sense: it needs no launch, runs headless anywhere the Gazebo
Python bindings import, and stays the fastest option, since there is no
Inter-Process Communication (IPC) once `<real_time_factor>0</real_time_factor>`
stops the simulation from throttling to real time. The harness backend
suits watching a launched simulation live, and both the harness and
per-agent paths are what a deployment actually runs against.

## Writing The Model SDF

Creating
`gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/<name>/` with a
`model.config` and a `model.sdf` starts the process. Starting from an
existing CAD model instead of writing SDF by hand is also a real,
tested path; see
[Importing CAD Models](importing_cad_models.md) for the export-to-URDF,
convert-to-SDF pipeline this project's own rover model went through,
before returning here for the rules below. For the harness
backend, the model should carry geometry only, with no `JointController`,
no `JointStatePublisher`, and no `<effort>` limit on any joint meant for
actuation, since an effort limit silently disables ECM velocity control in
this DART build. The world-level harness plugin owns actuation and sensing
instead.

The reference bare model lives at
[`models/cartpole_bare/model.sdf`](../gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/cartpole_bare/model.sdf).
Three rules matter most.

- Every actuated or observed joint needs a stable name, since the
  specification refers to joints by name, `slider_to_cart` and
  `cart_to_pole` for example.
- Every `<link>` carrying a `<collision>` should also carry a `<visual>`,
  or it loads physically but renders invisibly.
- The model should not pin itself to the world inside `model.sdf`. The
  spawner adds a world-scope fixed joint instead, through `extra_joints`
  described below, so the model survives being included elsewhere.

## Defining The AgentSpec

An `AgentSpec` is a plain dataclass. The annotated cartpole specification
below shows every field; the task is to change the values, not the
structure.

```python
from gymnasium import spaces
import numpy as np
from gazebo_gymnasium_bridge.envs.agent_spec import AgentSpec, JointObs, register_spec


def _my_agent_spec() -> AgentSpec:
    return AgentSpec(
        name="myagent",
        # The SDF <uri> merged into each spawned model wrapper.
        model_uri="package://gazebo_gymnasium_resources/models/myagent_bare",

        # Per-agent Gymnasium spaces; SB3 batches these across the N agents.
        observation_space=spaces.Box(low=-np.inf, high=np.inf,
                                     shape=(4,), dtype=np.float32),
        action_space=spaces.Discrete(2),            # or spaces.Box(...) for continuous

        # Builds the observation from the model's joint_state message,
        # ordered so each JointObs contributes [position, velocity] by
        # default. The summed widths must equal observation_space.shape[0].
        joint_obs=(JointObs("slider_to_cart"), JointObs("cart_to_pole")),

        # Per-agent reward for one step; obs is this agent's own row.
        reward_fn=lambda obs, action: 1.0,

        # Per-agent natural-termination test, here checking the pole fell.
        terminated_fn=lambda obs: bool(abs(obs[2]) > 0.20944),

        # Force-actuated models must spawn clear of the ground plane;
        # resting on it pins the model with contact friction (this exact
        # trap, at spawn_z=0.10, once masked cartpole's own force control).
        spawn_z=0.60,

        # Extra world-scope joints the spawner adds around the model, each
        # a (joint_name, parent, child) tuple; "world" as parent pins to it.
        extra_joints=(("world_to_slider", "world", "slider"),),

        # Harness (ECM) actuation and reset.
        # action -> [(joint_name, mode, value), ...], mode one of
        # "velocity", "force", "position". Applied via the ECM each tick.
        action_to_commands=_my_action_to_commands,
        # rng -> {joint_name: (position, velocity)}, set at episode start,
        # in place with no respawn. Per-agent randomization belongs here.
        reset_joint_state=_my_reset_joint_state,
    )
```

The two harness callbacks, written cartpole-style, follow.

```python
_FORCE = 10.0

def _my_action_to_commands(action):
    # np.ravel handles both a scalar (offline) and a length-1 row (the
    # harness passes each agent's action as one row of an (n_agents, act_dim) matrix).
    a = int(round(float(np.ravel(action)[0])))
    f = _FORCE if a == 1 else -_FORCE
    return [("slider_to_cart", "force", f)]

def _my_reset_joint_state(rng):
    return {
        "slider_to_cart": (0.0, 0.0),
        "cart_to_pole": (float(rng.uniform(-0.05, 0.05)), 0.0),   # a small random tilt
    }
```

Continuous actions work the same way: setting `action_space=spaces.Box(...)`
and having `action_to_commands` return force or velocity values from
`np.ravel(action)` covers it, and the built-in `cartpole_continuous`
specification is the reference. Force-actuated models must still
spawn clear of the ground plane, with `spawn_z` above the collision box,
since a model resting on the ground gets pinned by contact friction, which
velocity commands silently override but force commands cannot. Deliberate
contact, a hopper's foot for example, is fine, since contact is the actual
mechanism there.

Image and camera observations are supported too. Setting `image_obs=(H, W,
C)` and a `uint8` Box observation space, placing a `<sensor
type="camera">` with `<topic>camera</topic>` on the model, so the world
builder rewrites it to `/rl/camera_<i>` per agent and loads the render
system, and computing reward and termination from the image in plain
Python covers the pattern; the built-in `line_follower` specification shows
it in full. Mobile bases should set `reset_model_pose=True` so the chassis
returns to its spawn pose on reset, and `per_agent_include_uri` gives each
agent its own static scenery, a line track for example, so `n_agents=16`
genuinely builds 16 tracks, one per rover, with each agent's camera
publishing to its own `/rl/camera_<i>` topic.

Only one camera environment may exist per process, since gz-sim's
rendering scene behaves as a process-wide singleton that `close()` does not
tear down, making a second image-observation environment in the same
process a hard native crash. The library raises a clear `RuntimeError`
instead of crashing silently. This limit applies to environments, not
agents, so one environment can host many agents; `n_agents=16` is the right
call, not 16 separate environments. A genuine need for two calls for a
second, separate process, through `multiprocessing` or SB3's
`SubprocVecEnv`. Rendering costs roughly 50 times the physics-only
throughput, so keeping frames small, `64x64` for example, matters.

### Porting MuJoCo Environments

Three lessons carried over from the ports already built, `inverted_double_pendulum`
and `hopper` among them.

- **Planar floating bases are just joints.** MuJoCo's hopper and walker
  root is a slide-slide-hinge chain, not a free joint, so modeling it the
  same way in SDF, a world-pinned anchor into a prismatic forward joint
  into a prismatic vertical joint into a revolute pitch joint into the
  torso, lets the standard joint-based observation, reset, and actuation
  cover the whole robot. Pointing the forward axis along the Y axis keeps
  agents spaced along X from ever colliding. Only true three-dimensional
  free bases, ant and humanoid for example, need anything beyond joints.
- **MuJoCo joints carry hidden dynamics that need reproducing.** Its
  defaults add armature, reflected rotor inertia around 1 kilogram meter
  squared, and damping per joint; without them, MuJoCo-scale torques (gear
  around 200) make an SDF model explode within one step. SDF carries no
  armature tag, so emulating it means adding the armature value to each
  articulated link's inertia about its hinge axis and setting
  `<dynamics><damping>` explicitly. Keeping MuJoCo's joint range limits too
  matters, since position limits stay safe while `<effort>` limits are what
  breaks ECM actuation.
- **Derived quantities belong in Python, not the simulator.** Tip
  positions, forward-progress terms, and health checks should get computed
  from the joint observation inside `reward_fn` and `terminated_fn`, `_idp_tip`
  for example, keeping the specification layer simulator-free and
  unit-testable.

### The Spec Toolkit

Hand-writing the same plumbing every port needs gets replaced by a set of
core helpers, all living in `agent_spec.py` and exported from `envs`. A
complete planar locomotor specification runs about 15 lines.

```python
from gazebo_gymnasium_bridge.envs import (
    proportional_forces, pos_then_vel_obs, uniform_reset,
    forward_progress_reward, planar_health_termination)

ACT = ("thigh_joint", "leg_joint", "foot_joint")
ROOT = ("root_fwd", "root_up", "root_pitch")
spec = AgentSpec(
    ...,
    joint_obs=pos_then_vel_obs(("root_up", "root_pitch") + ACT, ROOT + ACT),
    action_to_commands=proportional_forces(ACT, 200.0),   # or per-joint gears
    reward_fn=forward_progress_reward(vel_index=5),
    terminated_fn=planar_health_termination(spawn_z=1.25, min_z=0.7,
                                            max_pitch=0.2),
    reset_joint_state=uniform_reset(ROOT + ACT, 0.005),
)
```

Every helper clips actions to the Box bounds and survives scalar probes,
and the built-in hopper, walker2d, half_cheetah, and reacher specifications
are all written this way.

### Domain Randomization For Sim To Real

Two `AgentSpec` fields randomize dynamics across the N agents in one world,
so each agent draws a different sample from the distribution and a policy
trained across them stays robust. Both fields reproduce from the
construction seed and default to off.

```python
AgentSpec(
    ...,
    mass_randomization=0.3,          # each agent's mass and inertia scaled U(0.7, 1.3)
    action_gain_randomization=0.2,   # each agent's actuator command scaled U(0.8, 1.2)
)
```

`action_gain_randomization` models actuator-gain uncertainty, since a
nominally trained policy can collapse under a plus-or-minus 20 percent
gain shift, exactly the sim-to-real brittleness Domain Randomization (DR)
exists to fix. `mass_randomization` bites whenever actuation is
force-based, since force equals mass times acceleration couples the two,
while pure velocity control largely masks it. Per-episode physics DR would
need a respawn path that does not exist yet; observation and sensor-noise
DR is available today through `TransformObservation`.
[`domain_randomization.md`](domain_randomization.md) covers the general
concept, why physics-level DR stays fixed per agent rather than varying
per episode, and a worked survey of further mechanisms, friction,
actuation latency, and visual or lighting variation among them, triaged by
feasibility.

## Registering The Spec

Registering the specification once at import time lets `make_multi("myagent",
...)` and `--agent myagent` find it. Adding the factory to `_SPEC_FACTORIES`
in `agent_spec.py`, or calling the public hook from a separate module, both
work.

```python
from gazebo_gymnasium_bridge.envs.agent_spec import register_spec

register_spec("myagent", _my_agent_spec)
```

Verifying it needs no launch, since this spins up the real simulator
in-process.

```python
from gazebo_gymnasium_bridge.envs import registered_specs, make_inprocess

print(registered_specs())              # [..., 'myagent']
env = make_inprocess("myagent", n_agents=4)
obs = env.reset()                      # raises loudly if the model is broken
```

Giving it a standard Gymnasium id as well lets any tool call `gym.make` on
it directly.

```python
import gymnasium as gym

gym.register(id="MyAgent-v0",
             entry_point="gazebo_gymnasium_bridge.envs.gym_env:GazeboEnv",
             kwargs={"agent": "myagent"})
```

## Testing The Spec Offline

The quickest physics check is the in-process environment itself, following
the probes in
[`test/test_mujoco_specs.py`](../gazebo_gymnasium_bridge/test/test_mujoco_specs.py):
passive instability, observation finiteness, and determinism, each easy to
copy and adapt by swapping the name. Below that layer, the specification
stays deliberately free of any `gz.*` import, so unit-testing observation,
reward, and termination with a duck-typed fake message is possible; see
[`test/test_agent_spec.py`](../gazebo_gymnasium_bridge/test/test_agent_spec.py).
Exercising real ECM actuation headlessly, without a full launch, means
driving `HarnessCore` from a `gz.sim8.TestFixture`, shown in
[`test/test_harness_core.py`](../gazebo_gymnasium_bridge/test/test_harness_core.py).
Both paths run under `pixi run test` without a running Gazebo instance.

## Wiring A World And Launch

For the harness backend, copying the two cartpole files and swapping the
model name covers it.

- **World**:
  [`worlds/cartpole_harness.sdf`](../gazebo_gymnasium_examples/gazebo_gymnasium_resources/worlds/cartpole_harness.sdf)
  is an empty world that loads the `MultiAgentHarness` plugin. It reads
  `GAZEBO_GYM_N_AGENTS` and `GAZEBO_GYM_AGENT` from the environment, so one
  fixed world SDF works for any N and any agent.
- **Launch**:
  [`launch/cartpole_harness.launch.py`](../gazebo_gymnasium_examples/gazebo_gymnasium_bringup/launch/cartpole_harness.launch.py)
  sets those environment variables and spawns N bare models through
  `spawn_multi_cartpoles.py --model-uri package://…/models/myagent_bare`.

Rebuilding resources afterward lets Gazebo find the new model and world.

```bash
pixi run build          # or: colcon build --symlink-install
```

## Running It

The fastest path needs one command and no launch, since the default
in-process backend picks the right policy and image wrappers
automatically.

```bash
python training_scripts/train.py --agent myagent --n_agents 16
python training_scripts/sweep.py --agent myagent --n_agents 16   # hyperparameter sweep
python training_scripts/deploy.py --agent myagent --n_agents 4   # evaluation
```

Watching it live needs two terminals, the Step 5 launch followed by the
harness client.

```bash
ros2 launch gazebo_gymnasium_bringup cartpole_harness.launch.py n_agents:=4 headless:=false
python training_scripts/deploy.py --agent myagent --n_agents 4 --backend harness
```

Bringing an entirely different trainer also works, since the environment
is a standard SB3 `VecEnv`, with agents auto-resetting independently under
the same-step convention.

```python
import stable_baselines3 as sb3
from gazebo_gymnasium_bridge.envs import make_inprocess, wrap_for_observations

vec_env, policy = wrap_for_observations(make_inprocess("myagent", n_agents=16))
sb3.PPO(policy, vec_env, n_steps=64).learn(total_timesteps=1_000_000)
```

## Checklist

- [ ] `models/<name>_bare/{model.config,model.sdf}` exists, geometry only,
      with no effort limit on actuated joints and visuals on every
      collision link.
- [ ] `AgentSpec` exists, with `joint_obs` widths summing to
      `observation_space.shape[0]`.
- [ ] `action_to_commands` and `reset_joint_state` exist for the harness
      backend.
- [ ] `register_spec("<name>", factory)` runs at import.
- [ ] `pixi run build` succeeds, followed by `train.py --agent <name>`
      in-process, with no launch needed.
- [ ] Optionally, to watch it live: a world SDF loading `MultiAgentHarness`
      plus a launch that spawns N models, then `--backend harness`.

[`docs/examples/cartpole.md`](examples/cartpole.md) carries the fully
worked reference, and [`README.md`](../README.md) covers environment setup.

## Glossary

| Term | Definition |
|---|---|
| **Entity Component Manager (ECM)** | Gazebo's runtime interface for reading and writing simulated joint and link state |
| **Simulation Description Format (SDF)** | The XML format Gazebo uses to describe a world and its models |
| **Domain Randomization (DR)** | Training across randomized simulation parameters so a policy generalizes past one fixed configuration |
| **Inter-Process Communication (IPC)** | Data exchange between separate processes, such as the harness backend's topic-based transport |
| **AgentSpec** | The dataclass describing one agent's model, observation, action, reward, and DR configuration |
