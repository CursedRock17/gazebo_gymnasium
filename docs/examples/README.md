# Examples

Every environment in this library is **one `AgentSpec`** (model + observation +
action + reward + termination). The framework runs *N* copies of a spec in a
single Gazebo world as a Stable-Baselines3 `VecEnv` — so there's no per-env
Python class, and "MultiCartPole", "MultiAnt", … are just named specs. To add
your own, follow [Creating your own agent](../creating_your_own_agent.md).

```python
from gazebo_gymnasium_bridge.envs import make_multi, make_harness
vec_env = make_multi("cartpole",  n_agents=16)   # per-agent backend
vec_env = make_harness("cartpole", n_agents=16)  # batched-harness backend (O(1) transport)
```

## Status

| Example | Spaces | Backends | Status |
|---------|--------|----------|--------|
| [CartPole](cartpole.md) | `Discrete(2)` / `Box(4,)` | inprocess + harness | ✅ Verified — force-controlled, PPO solves to the 500 cap |
| [CartPole — continuous](cartpole.md#continuous-variant) | `Box(1,)` / `Box(4,)` | inprocess + harness | ✅ Working — SAC/TD3 entry point (InvertedPendulum analog) |
| InvertedDoublePendulum | `Box(1,)` / `Box(6,)` | inprocess + harness | ✅ Verified — PPO 84 → 3650+ ep reward and climbing |
| Hopper | `Box(3,)` / `Box(11,)` | inprocess + harness | ✅ Verified — PPO 4 → 272 ep reward (4 → 134-step episodes) |
| Walker2d | `Box(6,)` / `Box(17,)` | inprocess + harness | ✅ Verified — PPO walks: 6.5 → 800 ep reward (600-step episodes) |
| HalfCheetah | `Box(6,)` / `Box(17,)` | inprocess + harness | ✅ Verified — PPO runs at ~1.15 m/s sustained (reward 0 → 1150) |
| Reacher | `Box(2,)` / `Box(6,)` | inprocess + harness | ✅ Working — goal-as-joints target, dense distance reward |
| Pusher | continuous | — | 🚧 Same goal-as-joints trick, next in line |
| Ant / Humanoid | continuous | — | 🚧 Need link-state obs extension (3D free base) |
| Swimmer | continuous | — | ❌ Not portable faithfully — swims via MuJoCo's viscous fluid medium; DART has no fluid drag |
| Line follower | `Box(2,)` wheels / `Box(64,64,3)` camera | inprocess | ✅ Pipeline verified — learns from pixels (4.8 → ~92); masters the straight, fails the first corner (see solved bars) |

## Solved bars — what "passing" means per environment

A ✅ above means **learnability is verified** (a clean training curve from the
random baseline, real physics, headless). *Solved* is a stricter, per-env bar:

| Env | Solved bar | Best verified so far |
|---|---|---|
| CartPole (both) | mean ep reward at the **500-step cap** | **500.0 — SOLVED** (sweep, 3 perfect trials) |
| InvertedDoublePendulum | mean ep length at the **1000-step cap** | 365 steps / 3652 reward @300k, still climbing |
| Hopper | 1000-step cap + reward ≥ 1000 | 134 steps / 272 @400k, climbing |
| Walker2d | 1000-step cap + reward ≥ 1500 | 710 steps / 1013 @400k, climbing |
| HalfCheetah | open-ended — report sustained m/s (MuJoCo-solved ≈ 5–6 m/s) | ~1.46 m/s @400k |
| Reacher | mean ep reward ≥ −5 (near-goal most of the episode) | −13 @100k (learning signal) |
| Line follower | mean ep length at the **300-step cap** (never loses the line ⇒ takes corners) | 76–82 steps ≈ the first straight; frame-stacked attempt in progress |

The state-based numbers above are single unswept PPO probes at modest budgets
(300–400k steps) — they establish *learnability*, and the remaining gap to the
bars is expected to close with longer budgets/tuning (as the cartpole sweep
demonstrated), not architecture changes.

The MuJoCo models have world SDFs and visualization launches
(`ant.launch.py`, …) you can load in Gazebo/Foxglove, but they are **not RL
environments yet** — each needs an `AgentSpec` (obs/action/reward/actuation)
written for it. That's exactly the porting exercise the
[create-your-own-agent guide](../creating_your_own_agent.md) walks through.

## The architecture (shared by every spec)

1. **The model SDF** — the robot geometry, in
   [`resources/models/`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/).
   For the harness backend it's geometry only (no controllers, no effort limit
   on actuated joints).

2. **The world + in-sim plugin** — a world SDF in
   [`resources/worlds/`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/worlds/)
   that loads exactly one world-level plugin (the `MultiAgentHarness`, or the
   per-agent controller). That plugin is the only Python running inside
   `gz sim`; it applies actions and reads sensors via the ECM. **No training
   logic lives in the sim.**

3. **The `AgentSpec`** — in
   [`envs/agent_spec.py`](../../gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/agent_spec.py).
   Pure Python, no `gz.*` import, unit-testable offline.

4. **The `VecEnv`** — `make_multi` / `make_harness` return a standard SB3
   `VecEnv`. Any Gymnasium-speaking library (SB3, RLlib, CleanRL, your own
   loop) drives it unchanged.

5. **The training script** — [`training_scripts/train.py`](../../training_scripts/train.py)
   (and `deploy.py`), parameterized by `--agent` / `--n_agents` / `--backend`.

## Training and deploying

```bash
# Terminal 1 — simulator (single-agent is just n_agents:=1)
ros2 launch gazebo_gymnasium_bringup cartpole_multi.launch.py n_agents:=16 headless:=true

# Terminal 2 — train, then roll out
python training_scripts/train.py  --agent cartpole --n_agents 16 --timesteps 200000
python training_scripts/deploy.py --agent cartpole --n_agents 16 --model models/final.zip
```

Two terminals let you swap trainers without restarting the sim. Add
`--backend harness` (and launch `cartpole_harness.launch.py`) to use the
batched-harness backend — O(1) transport and in-place reset, which scales to
large N without the per-agent discovery timeouts or respawn race. `train.py`
scales those timeouts with `n_agents` for you and prints an `[EpisodeSummary]`
each group auto-reset.

## Background docs (deeper dives)

- [`gymnasium_api_reference.md`](../gymnasium_api_reference.md) — Gymnasium API
  surface; useful when porting envs from other simulators.
- [`sb3_api_reference.md`](../sb3_api_reference.md) — what SB3 does to your env
  (Monitor wrapping, VecEnv auto-reset). Read before debugging SB3 weirdness.
- [`verbose_sb3_coexistence.md`](../verbose_sb3_coexistence.md) — how per-step
  logging interacts with SB3's verbose table.
