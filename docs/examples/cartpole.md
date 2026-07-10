# CartPole

Classic CartPole reproduced in Gazebo Harmonic, and the **reference example**
for the whole library. A cart slides along a rail on the Y axis; a pole is
hinged on top and rotates around the X axis. The agent applies a velocity
command to the cart and keeps the pole upright. If you're adding a new agent,
read this alongside [Creating your own agent](../creating_your_own_agent.md).

- **Observation** — `Box(4,)`: `[cart_pos, cart_vel, pole_angle, pole_ang_vel]`
- **Action** — `Discrete(2)`: push the cart −/+ along the rail
- **Reward** — `+1` per step alive
- **Termination** — `|pole_angle| > 0.209 rad` or `|cart_pos| > 2.4 m`

## How it's built

Everything is one `AgentSpec` (`_cartpole_spec()` in
[`envs/agent_spec.py`](../../gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/agent_spec.py)).
The framework runs *N* copies of it in a single Gazebo world as an SB3
`VecEnv` — there is no per-env Python class:

```python
from gazebo_gymnasium_bridge.envs import make_multi, make_harness
vec_env = make_multi("cartpole",  n_agents=16)   # per-agent backend
vec_env = make_harness("cartpole", n_agents=16)  # batched-harness backend
```

Two backends drive the same spec (see
[the backend table](../creating_your_own_agent.md#the-two-backends)):

- **peragent** — a world-level controller plugin, 2 topics/agent, reset by
  re-spawning each model. Model:
  [`models/cartpole`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/cartpole).
- **harness** — one `MultiAgentHarness` plugin, 3 topics total (O(1) in N),
  reset **in place** via the ECM (no respawn race, clean per-agent pole
  randomization). Bare model:
  [`models/cartpole_bare`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/cartpole_bare).

## Difficulty

The cart is bang-bang **velocity**-controlled (force control is non-functional
in this DART build). The commanded speed (`_CART_SPEED`, 2.5 m/s) is tuned so
this is a *real* RL problem: a **random policy survives only ~210 steps**
(median ~166, sometimes <10), while a trained PPO policy reaches the 500-step
cap. Below ~1 m/s a random policy already near-solves it; above ~3 m/s bang-bang
is too coarse to balance and learning plateaus.

## Running it

**Fastest — one command, no launch** (the in-process backend hosts the sim in
the training process):

```bash
python training_scripts/train.py --agent cartpole --n_agents 16
# ...or sweep hyperparameters headlessly to solve it:
python training_scripts/sweep.py --n_agents 16 --timesteps 250000
```

**Watch it in a launched sim** (two terminals; `headless:=false` for the GUI):

```bash
# Terminal 1 — simulator
ros2 launch gazebo_gymnasium_bringup cartpole_harness.launch.py n_agents:=16 headless:=true
# Terminal 2 — train against it
python training_scripts/train.py --agent cartpole --n_agents 16 --backend harness
```

Or the pixi shortcuts:

```bash
pixi run sim      # cartpole_multi.launch.py, GUI
pixi run train    # train.py against it
pixi run deploy   # roll out a trained policy
```

### Launch arguments

| Arg | Default | What it does |
|-----|---------|--------------|
| `n_agents` | `4` | Number of cartpoles spawned into the one world. |
| `headless` | `true` | `false` runs the Gazebo GUI (lower RTF; fewer agents recommended). |

### Training arguments (`train.py`)

| Arg | Default | What it does |
|-----|---------|--------------|
| `--agent` | `cartpole` | Registered spec name. |
| `--n_agents` | `4` | Must match the launch. Timeouts scale with this automatically. |
| `--backend` | `peragent` | `peragent` or `harness` — must match the launched world. |
| `--algo` | `ppo` | `ppo`/`a2c` (discrete). |
| `--timesteps` | `200000` | Total env steps. |

## What "good" looks like

Each group auto-reset prints an `[EpisodeSummary]` line. Mean episode length
climbs from a few steps toward the `max_episode_steps` cap (500) as PPO learns
to balance. With the harness backend you should see **no** `Visual: … already
exists` warnings and poles starting upright with small per-agent variation —
that's the in-place ECM reset working.

## Bring your own trainer

The env is a standard SB3 `VecEnv`, so any Gymnasium-speaking library drives it
unchanged:

```python
import stable_baselines3 as sb3
from gazebo_gymnasium_bridge.envs import make_harness

vec_env = make_harness("cartpole", n_agents=16)
sb3.PPO("MlpPolicy", vec_env, n_steps=64).learn(total_timesteps=1_000_000)
```
