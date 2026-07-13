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

- **Finish RL-tool integration.** Remaining: extend per-agent reset to the
  harness/multi (launched-sim) backends — needs a per-agent reset-mask in the
  plugin's `/rl/reset` protocol (the in-process + gymnasium paths already do
  per-agent reset). Lower priority: those backends are for *watching* a live
  sim, where group-vs-per-agent reset is cosmetic.
- **Per-episode physics domain randomization.** ECM exposes no mass/friction
  *setters* (read-only), so per-reset physics randomization would need a respawn
  path. The population-based variant (below) sidesteps this and is usually
  enough. Observation/sensor-noise DR is achievable today via a standard
  `TransformObservation` wrapper.
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
- Native `gymnasium.vector.VectorEnv` (`gymnasium.make_vec(...)` via a
  `vector_entry_point`) — the efficient N-in-one sim as a standard vector env
  (SAME_STEP autoreset, `final_obs` info).
- Verified compatibility with the ecosystem wrappers: SB3 `VecNormalize`,
  Gymnasium `RecordEpisodeStatistics` / `NormalizeObservation`.
- Determinism verified: same seed + actions → bit-identical trajectories
  (DART is deterministic; the seed threads through reset randomization).
- Population-based domain randomization (`AgentSpec.mass_randomization`,
  `action_gain_randomization`) — each of the N agents is a different dynamics
  sample; seed-reproducible. ±20% gain collapses a nominal policy 500→~18.
- flake8 + pep257 clean across the package under the project config.
