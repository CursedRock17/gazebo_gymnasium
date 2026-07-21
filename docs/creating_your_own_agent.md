# Creating your own agent

This guide shows how to add a brand-new RL environment to Gazebo Gymnasium and
train it. The whole library is driven by **one dataclass, `AgentSpec`** — you
describe a single agent (its model, observation, action, reward, termination)
and the framework runs *N* identical copies of it in one Gazebo world as a
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) `VecEnv`. There
is no per-environment Python class to write: "MultiCartPole", "MultiAnt",
"MultiHopper" are just **named specs**.

```python
from gazebo_gymnasium_bridge.envs import make_inprocess
vec_env = make_inprocess("cartpole", n_agents=16)   # 16 agents, one sim, no launch
```

Registered specs are also exposed as a standard **`gymnasium.Env`**, so any
Gymnasium-speaking tool (RLlib, CleanRL, Tianshou, TorchRL, `env_checker`) works
out of the box — importing the package registers the ids:

```python
import gazebo_gymnasium_bridge          # registers GazeboCartPole-v0, ...
import gymnasium as gym
env = gym.make("GazeboCartPole-v0")     # standard 5-tuple step / (obs, info) reset
```

Two worked tutorials accompany this guide:
[CartPole](examples/cartpole.md) (the reference environment) and
[Porting Hopper, annotated](examples/porting_hopper.md) — the narrated story of
a real port, including how each physics trap was diagnosed from headless
probes. The complete reference implementation is the cartpole spec in
[`gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/agent_spec.py`](../gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/agent_spec.py).
Copy it and change the parts described below.

---

## The three backends

Everything you write below (the spec + the model SDF) is shared. What differs
is *how the env talks to Gazebo* — pick with `make_inprocess` / `make_harness`
/ `make_multi` (or `train.py --backend`):

| | **inprocess** (default) | **harness** | **peragent** |
|---|---|---|---|
| Sim location | **in the training process** (TestFixture) | launched `gz sim` | launched `gz sim` |
| Transport | none (direct ECM) | 3 topics total (O(1) in N) | 2 topics per agent |
| Reset | in place via ECM | in place via ECM | delete + re-spawn |
| Needs a launch? | **no** — one `python` command | yes (`ros2 launch`) | yes (`ros2 launch`) |
| Best for | **training + CI, fastest** | a running/GUI sim you want to watch | small N, quick start |
| Model | bare (no controllers) | bare (no controllers) | model with controllers |

All three are spec-driven and share the *same* `reward_fn` / `terminated_fn` /
group-auto-reset. **Start with `inprocess`** — it needs no launch, runs headless
anywhere the gz bindings import, and is the fastest (no IPC; set
`<real_time_factor>0</real_time_factor>` so the sim isn't throttled to real
time). Use **harness** when you want to *watch* a launched sim; the per-agent
and harness paths are what you deploy against a live/visualized `gz sim`.

---

## Step 1 — Write the model SDF

Create `gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/<name>/`
with a `model.config` and `model.sdf`. For the **harness backend the model is
geometry only** — no `JointController`, no `JointStatePublisher`, and **no
`<effort>` limit on any joint you intend to actuate** (an effort limit silently
disables ECM velocity control in this DART build). The world-level harness
plugin owns actuation and sensing.

The reference bare model is
[`models/cartpole_bare/model.sdf`](../gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/cartpole_bare/model.sdf).
Key rules:

- Give every actuated/observed **joint a stable name** — the spec refers to
  joints by name (`slider_to_cart`, `cart_to_pole`).
- Every `<link>` with a `<collision>` should also have a `<visual>` (or it
  loads physically but renders invisibly).
- Don't pin the model to the world inside `model.sdf`. The spawner adds a
  world-scope fixed joint (see `extra_joints` below) so the model survives
  being `<include>`d.

---

## Step 2 — Define the `AgentSpec`

An `AgentSpec` is a plain dataclass. Here is the cartpole spec annotated field
by field — your job is to change the values, not the structure:

```python
from gymnasium import spaces
import numpy as np
from gazebo_gymnasium_bridge.envs.agent_spec import AgentSpec, JointObs, register_spec


def _my_agent_spec() -> AgentSpec:
    return AgentSpec(
        name="myagent",
        # SDF <uri> merged into each spawned model wrapper:
        model_uri="package://gazebo_gymnasium_resources/models/myagent_bare",

        # Per-agent Gymnasium spaces. SB3 batches these across the N agents.
        observation_space=spaces.Box(low=-np.inf, high=np.inf,
                                     shape=(4,), dtype=np.float32),
        action_space=spaces.Discrete(2),            # or spaces.Box(...) continuous

        # How to build the observation from the model's joint_state message.
        # Ordered: each JointObs contributes [position, velocity] by default.
        # The summed widths MUST equal observation_space.shape[0] (checked).
        joint_obs=(JointObs("slider_to_cart"), JointObs("cart_to_pole")),

        # Per-agent reward for a step (obs is this agent's observation row).
        reward_fn=lambda obs, action: 1.0,

        # Per-agent natural-termination test (e.g. the pole fell over).
        terminated_fn=lambda obs: bool(abs(obs[2]) > 0.20944),

        # Spawn geometry. Force-actuated models MUST spawn clear of the
        # ground plane — resting on it pins the model with contact friction
        # (this exact trap, at spawn_z=0.10, masked cartpole force control).
        spawn_z=0.60,

        # Extra world-scope joints the spawner adds around the model. Each is
        # (joint_name, parent, child); "world" as parent pins to the world.
        extra_joints=(("world_to_slider", "world", "slider"),),

        # --- Harness (ECM) actuation + reset ---
        # action -> [(joint_name, mode, value), ...]; mode in
        # {"velocity", "force", "position"}. Applied via ECM each tick.
        action_to_commands=_my_action_to_commands,
        # rng -> {joint_name: (position, velocity)} set at episode start,
        # in place (no respawn). Put per-agent randomization here.
        reset_joint_state=_my_reset_joint_state,
    )
```

The two harness callbacks, cartpole-style:

```python
_FORCE = 10.0

def _my_action_to_commands(action):
    # np.ravel handles a scalar (offline) or a length-1 row (the harness passes
    # each agent's action as a row of the (n_agents, act_dim) matrix).
    a = int(round(float(np.ravel(action)[0])))
    f = _FORCE if a == 1 else -_FORCE
    return [("slider_to_cart", "force", f)]

def _my_reset_joint_state(rng):
    return {
        "slider_to_cart": (0.0, 0.0),
        "cart_to_pole": (float(rng.uniform(-0.05, 0.05)), 0.0),   # small random tilt
    }
```

> **Continuous actions** work the same way — set `action_space=spaces.Box(...)`
> and have `action_to_commands` return force/velocity values from
> `np.ravel(action)` (the built-in `cartpole_continuous` spec is the reference).
> **Force-actuated models must spawn clear of the ground plane** (`spawn_z`
> above the collision box) — a model resting on the ground is pinned by contact
> friction, which velocity commands silently override but forces cannot.
> (Deliberate contact like a hopper's foot is fine — there contact *is* the
> mechanism.)
> **Image / camera observations** are supported: set `image_obs=(H, W, C)` and
> an uint8 Box observation space, put a `<sensor type="camera">` with
> `<topic>camera</topic>` on the model (the world builder rewrites it to
> `/rl/camera_<i>` per agent and loads the render system), and compute
> reward/termination from the image in plain Python — see the built-in
> `line_follower` spec. Mobile bases set `reset_model_pose=True` so the
> chassis returns to its spawn pose on reset, and `per_agent_include_uri`
> gives each agent its own static scenery (e.g. a line track) — so
> `n_agents=16` really does build 16 tracks, one per rover, and each agent's
> camera publishes to its own `/rl/camera_<i>` topic.
>
> **One camera environment per process.** gz-sim's rendering scene is a
> process-wide singleton that `close()` does not tear down, so creating a
> second image-observation env in the same process is a hard native crash.
> The library raises a clear `RuntimeError` instead. This is a limit on
> *environments*, not agents: one env can host many agents, so use
> `n_agents=16` rather than 16 separate envs. If you genuinely need two, put
> the second in a separate process (`multiprocessing`, SB3's `SubprocVecEnv`).
>
> Rendering costs ~50× the physics-only throughput; keep frames small (64×64).

### Porting MuJoCo environments

Three lessons from the ports (`inverted_double_pendulum`, `hopper`):

- **Planar "floating" bases are just joints.** MuJoCo's hopper/walker root is
  a slide-slide-hinge chain, not a free joint — model it the same way in SDF
  (world-pinned anchor → prismatic forward → prismatic vertical → revolute
  pitch → torso) and the standard joint-based obs/reset/actuation covers the
  whole robot. Point the forward axis along +Y so agents spaced along X never
  collide. Only true 3D free bases (ant, humanoid) need anything beyond joints.
- **MuJoCo joints carry hidden dynamics you must reproduce.** Its defaults add
  `armature` (reflected rotor inertia, ~1 kg·m²) and `damping` per joint;
  without them, MuJoCo-scale torques (gear ≈ 200) make an SDF model explode in
  one step. SDF has no armature tag — emulate it by adding the armature value
  to each articulated link's inertia about its hinge axis, and set
  `<dynamics><damping>` explicitly. Keep MuJoCo's joint *range* limits too
  (position limits are safe; it's `<effort>` limits that break ECM actuation).
- **Derived quantities live in Python, not the sim.** Tip positions,
  forward-progress terms, health checks — compute them from the joint
  observation inside `reward_fn`/`terminated_fn` (see `_idp_tip`), keeping the
  spec layer sim-free and unit-testable.

### The spec toolkit — don't hand-write the plumbing

The patterns every port repeats are core helpers (all in `agent_spec.py`,
exported from `envs`). A complete planar locomotor spec is ~15 lines:

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

All helpers clip actions to the Box bounds and survive scalar probes; the
built-in hopper/walker2d/half_cheetah/reacher specs are written this way.

### Domain randomization (sim-to-real)

Two `AgentSpec` fields randomize dynamics *across the N agents in the one
world* — each agent is a different sample from the distribution, so a policy
trained across them is robust. Both are reproducible from the construction seed
and off by default:

```python
AgentSpec(
    ...,
    mass_randomization=0.3,          # each agent's mass+inertia scaled U(0.7, 1.3)
    action_gain_randomization=0.2,   # each agent's actuator command scaled U(0.8, 1.2)
)
```

`action_gain_randomization` models actuator-gain uncertainty (a nominal-trained
policy can collapse under ±20% gain — exactly the sim-to-real brittleness DR
exists to fix), and `mass_randomization` bites whenever actuation is
force-based (F = ma couples them; under pure velocity control mass is largely
masked). Per-*episode* physics DR would need a respawn path;
observation/sensor-noise DR is available today via `TransformObservation`.

---

## Step 3 — Register the spec

Register it once at import time so `make_multi("myagent", ...)` /
`--agent myagent` can find it. Either add your factory to `_SPEC_FACTORIES` in
`agent_spec.py`, or call the public hook from your own module:

```python
from gazebo_gymnasium_bridge.envs.agent_spec import register_spec
register_spec("myagent", _my_agent_spec)
```

Verify — no launch needed, this spins the real sim in-process:

```python
from gazebo_gymnasium_bridge.envs import registered_specs, make_inprocess
print(registered_specs())              # [..., 'myagent']
env = make_inprocess("myagent", n_agents=4)
obs = env.reset()                      # raises loudly if the model is broken
```

Optionally give it a standard Gymnasium id too (any tool can then
`gym.make` it):

```python
import gymnasium as gym
gym.register(id="MyAgent-v0",
             entry_point="gazebo_gymnasium_bridge.envs.gym_env:GazeboEnv",
             kwargs={"agent": "myagent"})
```

---

## Step 4 — Test the spec offline (no simulator)

The quickest physics check is the in-process env itself (see the probes in
[`test/test_mujoco_specs.py`](../gazebo_gymnasium_bridge/test/test_mujoco_specs.py):
passive instability, obs finiteness, determinism — copy one and swap the name).
Below that, the spec layer is deliberately free of any `gz.*` import, so you
can unit-test observation/reward/termination with a duck-typed fake message —
see
[`test/test_agent_spec.py`](../gazebo_gymnasium_bridge/test/test_agent_spec.py).
To exercise real ECM actuation headlessly (no full launch), drive `HarnessCore`
from a `gz.sim8.TestFixture` as in
[`test/test_harness_core.py`](../gazebo_gymnasium_bridge/test/test_harness_core.py).
Both run under `pixi run test` without a running Gazebo.

---

## Step 5 — Wire a world + launch

For the **harness** backend, copy the two cartpole files and swap the model
name:

- World:
  [`worlds/cartpole_harness.sdf`](../gazebo_gymnasium_examples/gazebo_gymnasium_resources/worlds/cartpole_harness.sdf)
  — an empty world that loads the `MultiAgentHarness` plugin. It reads
  `GAZEBO_GYM_N_AGENTS` / `GAZEBO_GYM_AGENT` from the environment, so **one
  fixed world SDF works for any N and any agent**.
- Launch:
  [`launch/cartpole_harness.launch.py`](../gazebo_gymnasium_examples/gazebo_gymnasium_bringup/launch/cartpole_harness.launch.py)
  — sets those env vars and spawns N bare models via
  `spawn_multi_cartpoles.py --model-uri package://…/models/myagent_bare`.

Rebuild resources so Gazebo can find the new model/world:

```bash
pixi run build          # or: colcon build --symlink-install
```

---

## Step 6 — Run it

**Fastest path — one command, no launch** (the default in-process backend;
policy and image wrappers are picked automatically):

```bash
python training_scripts/train.py --agent myagent --n_agents 16
python training_scripts/sweep.py --agent myagent --n_agents 16   # hyperparameters
python training_scripts/deploy.py --agent myagent --n_agents 4   # evaluate
```

**Watch it live** (two terminals — the Step 5 launch, then the harness client):

```bash
ros2 launch gazebo_gymnasium_bringup cartpole_harness.launch.py n_agents:=4 headless:=false
python training_scripts/deploy.py --agent myagent --n_agents 4 --backend harness
```

Or bring your own trainer — the env is a standard SB3 `VecEnv` (agents
auto-reset independently, same-step convention):

```python
import stable_baselines3 as sb3
from gazebo_gymnasium_bridge.envs import make_inprocess, wrap_for_observations

vec_env, policy = wrap_for_observations(make_inprocess("myagent", n_agents=16))
sb3.PPO(policy, vec_env, n_steps=64).learn(total_timesteps=1_000_000)
```

---

## Checklist

- [ ] `models/<name>_bare/{model.config,model.sdf}` — geometry only, no effort
      limit on actuated joints, visuals on every collision link.
- [ ] `AgentSpec` with `joint_obs` widths summing to `observation_space.shape[0]`.
- [ ] `action_to_commands` + `reset_joint_state` for the harness backend.
- [ ] `register_spec("<name>", factory)` runs at import.
- [ ] `pixi run build`, then `train.py --agent <name>` (in-process — no launch).
- [ ] Optional, to watch live: world SDF loading `MultiAgentHarness` + a
      launch that spawns N models (Step 5), then `--backend harness`.

See [`docs/examples/cartpole.md`](examples/cartpole.md) for the fully worked
reference, and [`README.md`](../README.md) for environment setup.
