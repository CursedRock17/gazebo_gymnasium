# Creating your own agent

This guide shows how to add a brand-new RL environment to Gazebo Gymnasium and
train it. The whole library is driven by **one dataclass, `AgentSpec`** — you
describe a single agent (its model, observation, action, reward, termination)
and the framework runs *N* identical copies of it in one Gazebo world as a
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) `VecEnv`. There
is no per-environment Python class to write: "MultiCartPole", "MultiAnt",
"MultiHopper" are just **named specs**.

```python
from gazebo_gymnasium_bridge.envs import make_multi
vec_env = make_multi("cartpole", n_agents=16)   # 16 agents, one sim
```

The complete reference implementation is the cartpole spec in
[`gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/agent_spec.py`](../gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/agent_spec.py).
Copy it and change the parts described below.

---

## The two backends

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

        # Spawn geometry. spawn_z MUST be > 0 to clear the ground plane.
        spawn_z=0.10,

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
_SPEED = 1.0

def _my_action_to_commands(action):
    # np.ravel handles a scalar (offline) or a length-1 row (the harness passes
    # each agent's action as a row of the (n_agents, act_dim) matrix).
    a = int(round(float(np.ravel(action)[0])))
    v = _SPEED if a == 1 else -_SPEED
    return [("slider_to_cart", "velocity", v)]

def _my_reset_joint_state(rng):
    return {
        "slider_to_cart": (0.0, 0.0),
        "cart_to_pole": (float(rng.uniform(-0.05, 0.05)), 0.0),   # small random tilt
    }
```

> **Continuous actions** work the same way — set `action_space=spaces.Box(...)`
> and have `action_to_commands` return force/velocity values from
> `np.ravel(action)`. **Image / camera observations** (e.g. a line-follower)
> need an obs source beyond `joint_obs`; that's a planned `AgentSpec` extension,
> not yet wired.

---

## Step 3 — Register the spec

Register it once at import time so `make_multi("myagent", ...)` /
`--agent myagent` can find it. Either add your factory to `_SPEC_FACTORIES` in
`agent_spec.py`, or call the public hook from your own module:

```python
from gazebo_gymnasium_bridge.envs.agent_spec import register_spec
register_spec("myagent", _my_agent_spec)
```

Verify:

```python
from gazebo_gymnasium_bridge.envs import registered_specs, make_harness
print(registered_specs())          # ['cartpole', 'myagent']
env = make_harness("myagent", n_agents=4)
```

---

## Step 4 — Test the spec offline (no simulator)

The spec layer is deliberately free of any `gz.*` import, so you can unit-test
observation/reward/termination with a duck-typed fake message — see
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

Two terminals (single-agent is just `n_agents:=1`):

```bash
# Terminal 1 — the simulator + in-sim harness
ros2 launch gazebo_gymnasium_bringup cartpole_harness.launch.py n_agents:=16 headless:=true

# Terminal 2 — train, deploy, or bring your own RL library
python training_scripts/train.py  --agent myagent --n_agents 16 --backend harness
python training_scripts/deploy.py --agent myagent --n_agents 16 --backend harness --model models/final.zip
```

Because the env is a standard SB3 `VecEnv`, any Gymnasium-speaking trainer
(SB3, RLlib, CleanRL, your own loop) drives it unchanged:

```python
import stable_baselines3 as sb3
from gazebo_gymnasium_bridge.envs import make_harness

vec_env = make_harness("myagent", n_agents=16)
model = sb3.PPO("MlpPolicy", vec_env, n_steps=64)
model.learn(total_timesteps=1_000_000)
model.save("models/final.zip")
```

`train.py` scales the transport timeouts with `n_agents` for you and prints an
`[EpisodeSummary]` line each time the group auto-resets.

---

## Checklist

- [ ] `models/<name>_bare/{model.config,model.sdf}` — geometry only, no effort
      limit on actuated joints, visuals on every collision link.
- [ ] `AgentSpec` with `joint_obs` widths summing to `observation_space.shape[0]`.
- [ ] `action_to_commands` + `reset_joint_state` for the harness backend.
- [ ] `register_spec("<name>", factory)` runs at import.
- [ ] World SDF loading `MultiAgentHarness` + a launch that spawns N models.
- [ ] `pixi run build`, then launch + `train.py --agent <name> --backend harness`.

See [`docs/examples/cartpole.md`](examples/cartpole.md) for the fully worked
reference, and [`README.md`](../README.md) for environment setup.
