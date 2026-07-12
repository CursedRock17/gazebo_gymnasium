# Roadmap

Where `gazebo_gymnasium` is headed. Current state: the spec-driven,
N-in-one-sim architecture is in place — one `AgentSpec` runs *N* agents in a
single Gazebo world as an SB3 `VecEnv`, with three backends: `make_inprocess`
(sim hosted in the training process, no launch — the default and fastest),
`make_harness` (batched O(1) transport to a launched sim), and `make_multi`
(per-agent). CartPole is the fully working, verified reference (PPO learns it
from a ~210-step random baseline to the 500-step cap). See
[docs/creating_your_own_agent.md](docs/creating_your_own_agent.md).

## Near term

- **Finish RL-tool integration (tier 1 started).** Done: per-agent independent
  same-step autoreset in the in-process backend, and a registered, `check_env`-
  passing single-agent `gymnasium.Env` (`gymnasium.make("GazeboCartPole-v0")`).
  Remaining: extend per-agent reset to the harness/multi backends (needs a
  per-agent reset-mask in the plugin protocol), and add a native
  `gymnasium.vector.VectorEnv` + `vector_entry_point` so
  `gymnasium.make_vec(...)` returns the efficient N-in-one-sim env.
- **Standard wrappers + normalization** — verify `VecNormalize` /
  `RecordEpisodeStatistics`; obs are unbounded (velocities), so normalization
  matters. Then determinism/seeding audit and domain-randomization hooks.
- **Port the stub MuJoCo envs to `AgentSpec`s.** The SDFs are auto-converted
  and load in Gazebo (`ant`, `hopper`, `walker2d`, `humanoid`,
  `humanoidstandup`, `reacher`, `pusher`, `swimmer`, `point`), but none has an
  `AgentSpec` yet — each needs obs/action/reward/actuation written. Reacher
  (TD3) is the natural next slot.
- **`half_cheetah`** — hand-rename the duplicate frame names in the converted
  SDF so it loads, then give it a spec.
- **Image-observation `AgentSpec` extension** for the line-follower (camera →
  `Twist`/DiffDrive); today `AgentSpec` only reads joint state.

- **Force control for cartpole.** `Joint.set_force` is non-functional in this
  DART build (a 30 N command barely moves the cart), so cartpole uses bang-bang
  velocity control and the difficulty is tuned via `_CART_SPEED`. Fixing force
  control would make it the textbook force-controlled CartPole.

## Quality / infrastructure

- **CI**: a `.github/workflows/colcon-test.yml` that runs `colcon test`
  (functional + lint) on Ubuntu Noble + ROS 2 Jazzy. This is the main gating
  item for [Quality Level 3](QUALITY_DECLARATION.md).
- **Performance & stress benchmarks**: wall-clock steps/second per env vs
  canonical MuJoCo, and scaling curves across `n_agents` for both backends.
  Stress coverage lives in `test/test_stress.py`; a published benchmark script
  is still TODO.
- **Read the Docs** hosting for the Sphinx site once the API surface is stable.

## Done

- Single-Python environment via Pixi + RoboStack (killed the dual-Python split).
- Generalized, parameterized `MultiAgentGazeboVecEnv` + `make_multi`.
- Batched in-sim harness: `HarnessCore` (ECM), `MultiAgentHarness` plugin
  (O(1) transport), `HarnessVecEnv` client, in-place reset.
- In-process backend (`InProcessHarnessVecEnv`) — headless training with no
  launch; `<real_time_factor>0</real_time_factor>` unthrottles it (~88× to
  ~4600 agent-steps/s at N=16).
- Hyperparameter sweep (`training_scripts/sweep.py`, CSV + optional W&B) that
  verifies cartpole to the 500-step cap.
- Per-agent independent same-step autoreset (in-process backend) — correct
  vectorized-RL episode boundaries, not a shared group reset.
- Standard `gymnasium.Env` + registration (`gymnasium.make("GazeboCartPole-v0")`)
  that passes `gymnasium.utils.env_checker` — universal RL-tool interop.
- flake8 + pep257 clean across the package under the project config.
