# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] — 2026-07-20

First public release. Gazebo Gymnasium exposes Gazebo Harmonic robots as
Gymnasium environments trainable with Stable-Baselines3, driven by a single
`AgentSpec` dataclass per environment.

### Added

**Core architecture**

- `AgentSpec` — one dataclass describes an agent (model, observation, action,
  reward, termination); the framework runs *N* copies in one Gazebo world as a
  vectorized environment. No per-environment Python class to write.
- Three backends behind one interface:
  - `make_inprocess` (default) — hosts the simulator inside the training
    process via `gz.sim8.TestFixture`. No launch, fully headless, fastest.
  - `make_harness` — batched O(1) transport to a launched `gz sim`, for
    watching a live/GUI simulation.
  - `make_multi` — per-agent topics with respawn reset.
- `<real_time_factor>0</real_time_factor>` on the in-process worlds removes
  gz-sim's wall-clock throttle (~88× speedup; ~4,600 agent-steps/s at N=16).
- Per-agent independent same-step autoreset — correct vectorized-RL episode
  boundaries rather than a shared group reset.
- Determinism: same seed + same actions reproduce bit-identical trajectories.

**Environments** (8 registered specs, all learnability-verified headless)

| Spec | Gymnasium id |
|---|---|
| `cartpole` | `GazeboCartPole-v0` |
| `cartpole_continuous` | `GazeboCartPoleContinuous-v0` |
| `inverted_double_pendulum` | `GazeboInvertedDoublePendulum-v0` |
| `hopper` | `GazeboHopper-v0` |
| `walker2d` | `GazeboWalker2d-v0` |
| `half_cheetah` | `GazeboHalfCheetah-v0` |
| `reacher` | `GazeboReacher-v0` |
| `line_follower` | `GazeboLineFollower-v0` |

CartPole is solved (PPO reaches the 500-step cap). See
[docs/examples/README.md](docs/examples/README.md) for per-environment
"solved bars" and verified results.

**Standard API surface**

- `gymnasium.Env` registration passing `gymnasium.utils.env_checker`.
- Native `gymnasium.vector.VectorEnv` via `vector_entry_point`
  (`gymnasium.make_vec(...)`), with SAME_STEP autoreset and `final_obs` info.
- Verified against ecosystem wrappers: SB3 `VecNormalize`/`VecMonitor`,
  Gymnasium `RecordEpisodeStatistics`/`NormalizeObservation`.

**Observations and actuation**

- Joint-state observations via `JointObs` (per-joint position/velocity).
- Image observations (`AgentSpec.image_obs`) — an agent's onboard camera as
  the observation, with per-agent camera topics and headless GPU rendering.
- ECM actuation: force, velocity, and position joint commands.
- Mobile bases: `reset_model_pose` restores chassis pose on reset.

**Spec toolkit** — the patterns every port repeats, as composable helpers:
`proportional_forces`, `proportional_velocities`, `pos_then_vel_obs`,
`uniform_reset`, `forward_progress_reward`, `planar_health_termination`.

**Domain randomization** — population-based, seed-reproducible:
`mass_randomization` and `action_gain_randomization` make each of the N agents
a different dynamics sample.

**Tooling**

- `training_scripts/train.py`, `deploy.py`, `sweep.py`, parameterized by
  `--agent` / `--n_agents` / `--backend`, with automatic policy selection
  (`MlpPolicy` for state, `CnnPolicy` + frame stacking for image observations).
- Pixi + RoboStack environment: ROS 2 Jazzy, Gazebo Harmonic, and the RL
  libraries in one locked conda environment — a single Python for the whole
  stack. `pixi run train --agent hopper --n_agents 16` after two install steps.
- 260+ test suite (spec math, real-physics probes, SB3/Gymnasium contract,
  in-sim harness, stress, vision, SDF validity) plus the three ament linters,
  all runnable headless with no simulator launch.

### Documentation

- [Creating your own agent](docs/creating_your_own_agent.md) — the reference
  guide, including the MuJoCo porting recipe and the physics traps.
- [Porting Hopper, annotated](docs/examples/porting_hopper.md) — a narrated
  real port with the diagnosis reasoning and measured numbers.
- [CartPole walkthrough](docs/examples/cartpole.md), per-environment status
  and solved bars, ROS 2 REP compliance mapping, and a quality declaration
  (REP-2004 Level 4).

### Known limitations

- Physics runs on CPU — Gazebo's DART backend has no GPU path. Throughput
  comes from running N agents in one world, not from GPU physics.
- Vision environments train ~40× slower than state-based ones; the learner is
  the bottleneck on CPU. Set `GAZEBO_GYM_DEVICE=cuda` on an NVIDIA host.
- Only one camera-based environment can exist per process (gz-sim's rendering
  scene is a process-wide singleton). This limits environments, not agents —
  one env hosts any number of agents, each with its own camera and scenery.
  A clear `RuntimeError` is raised rather than crashing.
- The line follower learns from pixels but does not yet complete the track
  (it masters the straight and loses the line at the first corner).
- Swimmer is not portable faithfully — it relies on MuJoCo's viscous fluid
  medium, and DART has no fluid drag.
- `ant`/`humanoid` need a link-state observation extension for their 3D free
  bases; `pusher` is not yet ported.

<!-- TODO(release): replace CursedRock17/gazebo_gymnasium with the published GitHub path. -->
[Unreleased]: https://github.com/CursedRock17/gazebo_gymnasium/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CursedRock17/gazebo_gymnasium/releases/tag/v0.1.0
