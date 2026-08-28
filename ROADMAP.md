# Roadmap

This document tracks where `gazebo_gymnasium` is headed. The current
state centers on a spec-driven, N-in-one-simulation architecture: one
`AgentSpec` runs N agents in a single Gazebo world as an SB3 `VecEnv`,
across three backends. `make_inprocess` hosts the simulation in the
training process with no launch needed, the default and fastest option;
`make_harness` provides batched O(1) transport to a launched simulation;
`make_multi` runs per-agent. CartPole stands as the fully working,
verified reference, with PPO learning it from a roughly 210-step random
baseline up to the 500-step cap. See
[docs/creating_your_own_agent.md](docs/creating_your_own_agent.md).

## Near Term

**Tracking the next Gazebo/ROS 2 LTS pair (Jetty + Lyrical).** This
project currently targets the latest recommended LTS pairing, ROS 2
Jazzy Jalisco plus Gazebo Harmonic, per
[gazebosim.org's own compatibility matrix](https://gazebosim.org/docs/latest/ros_installation/).
That same matrix marks Gazebo Jetty (LTS, supported September 2025
through May 2031) as recommended against ROS 2 Lyrical (LTS) and ROS 2
Rolling, not against Jazzy, so following the trend means moving to Jetty
rather than staying on Harmonic. Since this project's actual dependency
is Gazebo, not ROS 2 (`ros_gz_bridge` and launch files are a thin layer
on top, not load-bearing for the core `AgentSpec`/backend architecture),
versioning should track GZ releases, with ROS 2 support following
whatever each GZ release recommends. This needs at least three branches,
not a straight cutover: `rolling` (this repository's own ongoing
development, tracking whatever pairing is newest), a `harmonic` branch
(pinned to the current Jazzy + Harmonic pair this project ships on
today, kept alive for existing users), and a `jetty` branch (the Lyrical
+ Jetty pair, once Pixi's RoboStack channel carries it). Not started;
this entry tracks the intent, not a committed timeline.

**Hugging Face Hub integration, first slice done (2026-08-25).** `train.py
--push-to-hub <repo_id>` and `deploy.py --from-hub <repo_id>` are built and
lint-clean; end-to-end verification (a live push + re-download) is still
pending as of this writing. Uses `huggingface_hub`'s `HfApi` directly
rather than `huggingface_sb3`'s `package_to_hub`, a deliberate choice, not
a placeholder: `package_to_hub` also renders a replay video through the
camera/GUI path, which the one-camera-environment-per-process limit makes
awkward to automate, so this slice pushes the model, an auto-generated
card, and real evaluation metrics, leaving video out of scope entirely
rather than half-building it.

```bash
pixi run train --agent line_follower --push-to-hub <user>/<repo>
pixi run deploy --agent line_follower --from-hub <user>/<repo>
```

- `train.py --push-to-hub <repo_id>`, after training succeeds (not on a
  crashed or partial run), runs a real deterministic eval right there,
  the same precise per-agent tracking this project's DR-tuning work
  already validated, not the training curve's own rolling metric, then
  uploads the model plus a card documenting the algorithm, the exact
  hyperparameters, and that real eval result. Verifies the upload landed
  with `file_exists()` rather than trusting a non-exception return.
- `deploy.py --from-hub <repo_id>` downloads via `hf_hub_download` and
  runs it exactly like a local `--model` path.
- `huggingface_hub` is a soft dependency, the same treatment `wandb`
  already gets: not in `pixi.toml`, install with `pixi run python -m pip
  install huggingface_hub` inside the pixi env specifically (`pixi run
  pip` resolves to a different Python entirely and silently installs to
  the wrong place -- a real footgun hit twice this session).
- Still open: `sweep.py` pushing its winning configuration with its
  learning curve, closing the loop between a sweep and a shareable
  artifact.

**True multi-agent support (MARL)** does not exist yet. Right now, N
agents in one world amount to N independent copies of one specification;
`reward_fn` and `terminated_fn` only ever see one agent's own observation
and action, never another agent's state, so no pursuit, cooperative, or
competitive task is possible today. Extending those function signatures
to see the whole batch would be needed. This is the single most
interesting capability gap given the framework's own
N-agents-in-one-world pitch; see `docs/comparison.md`.

**Sweeping the three non-SB3 libraries on `cartpole_continuous`** remains
open. skrl (484 of 500), rl_games (458 of 500), and rsl_rl (382 of 500)
all trained with reasonable but unswept hyperparameters, none receiving
SB3's `sweep.py`-style treatment. A real sweep per library would very
plausibly close some or all of the gap to SB3's swept 500 of 500; right
now this stands as an honest first-attempt comparison, not evidence these
libraries run weaker on this task. See `docs/rl_libraries.md`.

**Investigating skrl's roughly 15 times wall-clock gap** is still open.
Same task, same machine, same 250,000 steps: SB3, rl_games, and rsl_rl
land in a 43-to-50-second band, while skrl took 714 seconds. The likely
cause is per-step Python and logging overhead in `SequentialTrainer` and
`RunningStandardScaler`, not the GPU or the environment, though this has
not been profiled to confirm yet.

**SB3's `sweep.py` still does not log value loss**, a real, separate gap
from the item above; the other three libraries' own runners do this by
default, `train.py` now does too (`--tensorboard`), but `sweep.py` still
does not.

**PufferLib as a fifth RL library** looks possible but not a same-shape
integration as skrl, rl_games, or rsl_rl, per a real skim of
[PufferAI/PufferLib](https://github.com/PufferAI/PufferLib)'s own source
(the README is sparse; the real docs live at puffer.ai). The right
integration point is `pufferlib.PufferEnv`, its native, already
multi-agent interface (`num_agents`, batched in-place buffer writes on
`step`/`reset`), not `pufferlib.emulation.GymnasiumPufferEnv` wrapping a
single-agent `gymnasium.Env`; the latter hands parallelism to PufferLib's
own `pufferlib.vector.make(..., num_envs=N)`, which would spawn N
separate instances itself, the same ownership collision rl_games'
default path had. A `GazeboPufferVecEnv(pufferlib.PufferEnv)` wrapping
`make_inprocess` and copying its batched results into Puffer's buffers
each tick would be the same shape as the custom `IVecEnv` already built
for rl_games. Action-space coverage looks fine: both categorical-logits
(Discrete) and `torch.distributions.Normal` (Box) sampling exist in its
trainer code, covering `cartpole` and the continuous specs alike. The
real new obstacle: **PufferLib 4.0 is not a plain pip package.**
`pufferlib.emulation`, `pufferlib.vector`, and `pufferlib.PufferEnv` all
live inside a compiled `_C` extension that a `build.sh` script produces
from C/C++ (and, in practice, CUDA; a `--cpu` build mode exists on paper,
but the whole architecture, its own neural-net primitives, zero-copy CUDA
pointer plumbing, `bf16`/`cudnn_conv2d.cu` kernels, is built GPU-first,
so CUDA is almost required to get the library's actual intended
performance rather than a degraded fallback path), materially heavier
than skrl/rl_games/rsl_rl's zero-compile pip installs. Not started; a
small spike, confirming `build.sh` produces a working `_C.so` inside the
`rl-libs` pixi environment at all against this project's pinned
torch/CUDA versions, should come before writing a real
`train_pufferlib.py`.

**Finishing reinforcement-learning tool integration** means extending
per-agent reset to the harness and per-agent (launched-simulation)
backends, which needs a per-agent reset mask in the plugin's `/rl/reset`
protocol; the in-process and gymnasium paths already do per-agent reset.
This carries lower priority, since those backends exist mainly for
watching a live simulation, where group-versus-per-agent reset stays
cosmetic.

**Per-episode physics domain randomization** would need a respawn path,
since the Entity Component Manager (ECM) exposes no mass or friction
setters, only readers. The population-based variant, already shipped,
sidesteps this and usually suffices. Observation and sensor-noise domain
randomization (DR) is already achievable today through a standard
`TransformObservation` wrapper.

**A dedicated actuator-model toolkit for sim-to-real transfer** does not
exist yet. `action_gain_randomization`, `action_noise_randomization`, and
`battery_discharge_randomization` (`docs/domain_randomization.md`)
perturb an idealized proportional actuator, gain scaling, per-step
jitter, and per-episode authority decay, but none of them model actual
actuator dynamics: PID controller lag, PWM deadband, mechanical
backlash, or command-to-motion latency. `action_to_commands` already
gives every `AgentSpec` a raw extensibility hook, a free-form callable
mapping a normalized action to per-joint commands, so a user can write a
custom actuator model today with no framework change needed, but nothing
built-in ships one, and there is no shared toolkit for it the way
`proportional_forces`/`proportional_velocities` cover the ideal case. The
gap is real, not theoretical: the one physical sim-to-real deployment
this framework has driven treated velocity-actuator mismatch (commanded
versus actual wheel speed under real PID, PWM, and slip behavior) as
unmodeled, leaning on `action_gain_randomization` as a workaround rather
than an actual model of it, and deliberately excluded real encoder
feedback from the trained policy's own observation for the same reason;
see `docs/examples/line_follower.md`'s Sim To Real section. Building a
composable lag or deadband/backlash helper, matching the spec toolkit's
existing pattern, would close this.

**Porting the remaining MuJoCo environments to `AgentSpec`s** has made
real progress. Done and learnability-verified: `cartpole_continuous`
(InvertedPendulum), `inverted_double_pendulum`, `hopper`, `walker2d`, and
`half_cheetah`/`reacher` (see `docs/examples/README.md` for status).
`hopper` established the planar-joint-root recipe plus MuJoCo armature
and damping emulation (`docs/creating_your_own_agent.md`). `ant` is
specified and mechanically verified, with sane observations and 100
percent survival under random actions, through a new
free-floating-3D-base recipe (`AgentSpec.base_obs` plus `HarnessCore`'s
`Link.world_pose`, `world_linear_velocity`, and
`world_angular_velocity`), alongside a real pybind11 gotcha this
surfaced: `gz.math7` must be imported before `gz.sim8` reads a Pose3d or
Vector3d back out of the ECM, or the C++-to-Python conversion silently
fails (see the comment above `harness_core.py`'s imports). PPO at 150,000
steps converges to a stand-still local optimum, 0.0 meters per second, 100
percent survival, rather than walking, a known Ant difficulty under PPO's
reward weights (`alive_bonus` and `ctrl_cost` match MuJoCo Ant-v4's own
defaults), not a broken port. SAC was tried (2026-08-14) and does not fix
it: a real sweep hit 960.3 reward, but verified sustained forward velocity
still reads 0.00 meters per second, mean absolute action 0.06, alive the
whole episode; SAC found a more stable version of the identical
stand-still-and-collect-the-alive-bonus solution, not a walking gait. This
rules out a PPO-specific exploration failure as the explanation, since the
literature suggested SAC's entropy bonus should counter exactly this
local optimum, and it did not, here. The real lever is a reward change,
lowering or removing `alive_bonus`, or adding an explicit
standing-still penalty, not a different algorithm; see
`docs/examples/README.md`'s status table. `humanoid` and `humanoidstandup`
form the direct next application of the same `base_obs` recipe, with
converted models already sitting in `models/humanoid{,standup}`, unused.
`pusher` needs a target body, following reacher's target-as-joints trick;
`swimmer` stays out of scope, since it needs fluid drag and DART carries
no fluid-dynamics engine.

**Line follower reaching the solved bar** was completed after this item
was originally written; see `docs/examples/README.md` and
`docs/examples/line_follower.md` for the current, verified 300-of-300
result. The three newest domain-randomization mechanisms,
`track_color_randomization`, `action_noise_randomization`, and
`battery_discharge_randomization`, have since been through that same
empirical strength-tuning pass, individually and combined; the combined
configuration reaches 37 of 40 (92.5 percent) at the best strength found so
far, real progress but short of the full solved bar, so it has not
replaced the published Hugging Face policy. A follow-up hyperparameter
sweep found a configuration that looked stronger at a short budget but
regressed once extended to a full training run. See
`docs/domain_randomization.md` for the full tuning history and open
threads.

That solved bar needs one correction, added 2026-08-27. "Solved" meant
keeping the line in frame for 300 steps, and nothing in the definition
required the rover to cover ground. It did not: measured across every
checkpoint in that history, the policies reversed 38 to 45 percent of their
steps and crept the rest, and the better a checkpoint scored the LESS it
drove, 4.68 m of wheel travel down to 1.08 m as the solve rate climbed. The
cause is structural rather than a tuning miss. `terminated_fn` fires only on
line loss, so creeping is always safe and committing to speed always risks
ending the episode early.

This is the same shape as the Ant stand-still local optimum above, and the
two took the same lesson. Reward weighting did not fix either one: forward
motion was already weighted 1.0x here for exactly this reason, and an
action-magnitude penalty collapsed the policy outright. What worked was
removing the degenerate behavior from the action space instead of pricing it,
via a forward-only clamped action mapping (`clamped_velocities`) that leaves
no stop and no reverse anywhere in the range. The equivalent move for Ant
would be a reward change, since standing still cannot be removed from a
torque-actuated action space the same way. `training_scripts/validate_rover.py`
now checks camera angle, forward motion and speed realism from the
simulator's own odometry, and `sweep.py --validate` runs it per trial, so a
return figure and a motion figure arrive together rather than one standing in
for the other.

**Demo footage: a training run, then the same policy on the rover.** The
pipeline's claims are easier to believe when you can watch them. The plan is
one video of a real training run at 4, 8 and 16 agents in a single world,
uploaded unlisted, then a second clip of the resulting policy driving the
physical rover, so the sim run and the hardware run sit side by side as
evidence the whole path holds together. Nothing about this is blocked; it
needs a machine with a live display and someone to point a camera at the
rover.

Recording the Gazebo GUI turns out to be the fiddly part, and it is worth
knowing why before anyone tries it. Screen capture failed outright on the
training box: the Pixi `ffmpeg` build ships without `x11grab`, and PIL or
`xwd` grabs of the root window come back pure black whenever the monitor has
blanked, which is the normal state for an unattended machine. The first
attempt produced a perfectly plausible one megabyte MP4 that was entirely
black, so check a frame rather than a file size. Two routes work instead.
Gazebo's own `VideoRecorder` GUI plugin records the render window from the
inside and needs no display capture, which is the right answer for headless
runs. Or record from a machine you are sitting at, where the screen is awake.

One trap specific to `line_follower`: the harness backend implements no track
randomization at all, so a GUI launched that way shows the original single
fixed oval, not the seven-shape task the current policy is measured on.
Footage of the real task has to come from the in-process backend, which
publishes its scene over gz-transport, so `gz sim -g` attaches to a running
world and renders it live. `docs/images/line_follower_multitrack_eval.mp4`
is the interim stand-in, built from the rovers' own camera feeds rather than
the 3D scene.

## GPU Acceleration

An audit from 2026-07 produced the following picture of where a GPU
actually helps today.

| Layer | GPU Possible? | Current Status |
|---|---|---|
| Physics (DART) | No, upstream: gz-physics has no GPU engine | CPU by design, mitigated by N-agents-in-one-simulation plus `real_time_factor=0` |
| Camera rendering (ogre2) | Yes | Already on the GPU, verified through `GL_RENDERER = Mesa Intel Graphics (MTL)` via headless EGL; the roughly 180-ticks-per-second vision ceiling reflects GPU-rendered readback and sensor pipeline overhead, not raster cost |
| Learner (PyTorch/SB3) | Yes, NVIDIA only | Wired (`GAZEBO_GYM_DEVICE=cuda`); the development laptop carries no NVIDIA device, so CNN training runs CPU-bound there |

**GPU-accelerating the physics engine (a PhysX backend)** is blocked
upstream, not something this project can build alone. NVIDIA's PhysX
supports simulating physics on the GPU, scaling to hundreds or thousands
of models at once, the same GPU-batched regime `docs/comparison.md`
covers for MJX, Isaac Lab, and Brax, but `gz-physics` itself has no PhysX
engine plugin today; DART, the only engine every backend here uses, stays
CPU-only, matching the table above.
[gazebosim/gz-physics#153](https://github.com/gazebosim/gz-physics/issues/153)
tracks the ask at the gz-physics level: open, unassigned, no pull request
yet. If PhysX support lands there, and its engine-plugin contract stays
close to DART's (joint position/velocity/force commands, ECM-queryable
link state), picking it up here would mostly mean a `physics_engine`
selection in the world SDF; a different ECM contract would mean more
work. Worth revisiting once the upstream issue shows real movement, not
before.

**The per-agent backend's dynamics** still drive the controller-equipped
`cartpole` model with velocity commands (`JointController`), so its
dynamics differ from the force-based specification the ECM backends
(in-process and harness) use, and policies do not transfer to it. Either
porting it to `gz-sim-apply-joint-force-system`, or retiring it once the
harness backend is live-validated, would resolve this.

## Quality And Infrastructure

**Continuous integration** runs through `.github/workflows/ci.yml`, which
builds with pixi and runs the full suite plus linters on Ubuntu Noble,
the REP-2000 Tier 1 platform, on every push and pull request, nightly,
and on demand. Its first green run on GitHub remains the last gating item
for [Quality Level 3](QUALITY_DECLARATION.md).

**Performance benchmarks** run through `training_scripts/benchmark.py`,
publishing steps per second per environment and scaling curves across
`n_agents` (`--scale 1,4,16,32`, CSV output). Comparison against canonical
MuJoCo, PyBullet, and a real GPU-batched-physics number through MJX is
done; see [`docs/comparison.md`](docs/comparison.md). Still open: the
same numbers for the launched-simulation (harness) backend, not just
in-process.

**Read the Docs** support, through `.readthedocs.yaml` and
`docs/requirements.txt`, is in place, and the Sphinx API reference builds
warning-free. Still open: connecting the repository on readthedocs.org
and replacing the `CursedRock17/gazebo_gymnasium` placeholders.

## Done

**Research-informed hyperparameter sweeps for the unsolved MuJoCo ports**
(2026-08-14) anchored a real per-agent hyperparameter grid in `sweep.py`
(a new `AGENT_CONFIGS` plus `_configs_for()`, replacing the old
one-size-fits-all grids for these six agents) on real published
hyperparameters, rl-baselines3-zoo's own tuned configurations for the
closest standard Gymnasium/MuJoCo task, plus its finding that real MuJoCo
environments barely need tuning beyond `learning_starts` for SAC. New
`sac` support followed as well (`sweep.py` previously only carried ppo,
a2c, and ddpg). A sweep-winning configuration still climbing steeply, not
plateaued, at its 300,000-step cutoff got a follow-up: `train.py`'s
existing resume-from-checkpoint feature carried it to 600,000 steps total
on the exact same hyperparameters, with no re-searching. Results, most
significant first, follow.

- **InvertedDoublePendulum: newly solved**, 4357 reward at 300,000 steps,
  verified across 10 of 10 real evaluation episodes at the 1000-step cap.
- **Hopper: newly solved**, climbing from 272 to 533 at 300,000 steps
  (the zoo's `Hopper-v4` configuration used verbatim) to **1002.2 reward,
  999 of 1000 steps, 10 of 10 real evaluation episodes at 600,000 steps**
  after the resume.
- **Walker2d: newly solved.** The zoo's own tuned configuration actually
  did poorly here (82.7 at 300,000 steps); a hand-picked variation
  reached 485.5 at 300,000 steps, still climbing, then **1522.9 reward,
  999 of 1000 steps, 10 of 10 real evaluation episodes at 600,000 steps**
  after the resume. This follows the same past-best comparison pattern as
  HalfCheetah below: the previous 1013 figure is now superseded, not a
  parallel result worth keeping alongside the new one.
- **HalfCheetah** climbed from 1.46 to 2.51 meters per second sustained at
  300,000 steps, then to **3.01 meters per second sustained (median
  3.25) at 600,000 steps** after the resume. Not solved, since this is an
  open-ended metric and MuJoCo-solved sits around 5 to 6 meters per
  second, but real, verified further progress.
- **Ant** got SAC tried as the literature-suggested fix for the
  stand-still local optimum; it does not fix it (960 reward, but verified
  0.00 meters per second, see the Ant entry above). This counts as a
  real, useful negative result: a reward-shaping problem, not an
  algorithm-choice one.
- **Reacher** was swept with real published starting points, both PPO and
  SAC; neither beat the existing best-known result (the fresh SAC sweep's
  best came to −9.9 against the existing −8.48). This reads as genuinely
  inconclusive, not a regression, since the existing −8.48 remains what's
  documented as best-known.
- Two real `sweep.py` bugs were found and fixed along the way, both
  concrete and generally applicable rather than agent-specific. First,
  the orchestrator's winner-selection sentinel was `best = -1.0`, which
  silently picks no winner and then deletes every trial's model whenever
  every score reads more negative than −1.0, exactly Reacher's normal
  reward range; fixed with `float("-inf")`. Second, the per-trial
  temporary model and CSV filenames (`.sweep_trial_{idx}.zip`) were never
  namespaced by the sweep's own `--out` name, only by trial index, so
  running multiple sweeps concurrently, six at once in this case, let one
  sweep's real trial-0 winner get silently overwritten by another sweep's
  own trial-0, losing real training work with no error; fixed by
  namespacing on `--out`'s filename stem.

**RL library integration beyond SB3** (skrl, rl_games, and rsl_rl, the
same four Isaac Lab advertises) trained all four end to end on
`cartpole_continuous` against the real engine, not stand-ins:
`training_scripts/train_skrl.py`, `train_rl_games.py`, and
`train_rsl_rl.py` (a new `rl-libs` pixi environment, kept out of the
default environment). skrl needed zero adapter code. rl_games needed a
small custom `IVecEnv`, the same mechanism Isaac Lab's own
`RlGamesVecEnvWrapper` uses. rsl_rl needed a real `TensorDict`/torch-native
adapter. Real, hard-won bugs were found and fixed in each: a
memory/rollout-size mismatch for skrl; dict-wrapped observations, legacy
`gym.spaces`, and per-agent episode-boundary semantics for rl_games; and
a missing `distribution_cfg` for rsl_rl. See `docs/rl_libraries.md` for
the full story, code, and real training curves (reward, loss, wall-clock
time). `cartpole_continuous` also got a real PPO sweep along the way and
now reads as genuinely solved (500.0 of 500, 4 of 4 perfect trials), not
just working; see `docs/examples/cartpole.md`. All three new scripts now
take `--agent <name>` like `train.py` and `sweep.py` do, verified against
`hopper`, with different observation and action dimensions, real
training, plus regression checks on the `cartpole_continuous` default,
instead of staying hardcoded to one environment; each also validates the
specification up front (Box actions, no image observations) with a clear
error otherwise.

**TensorBoard and Weights and Biases integration in `train.py`** added
`--tensorboard` (SB3's standard `tensorboard_log=`) and `--wandb`
(mirroring through `sync_tensorboard`, a soft dependency like `sweep.py`'s
own `--wandb`, degrading cleanly if `wandb` is not installed or
authenticated). Along the way, `train.py` was found never wrapping with
`VecMonitor`, so `rollout/ep_rew_mean`, the actual reward-over-training
curve and the main thing a dashboard exists for, never populated in the
console table or in either logger; fixed by wrapping the vector
environment with `VecMonitor` before the observation wrappers, matching
`sweep.py`'s existing pattern. `tensorboard` became a real, non-optional
pixi dependency, since it stays pure local logging with no external
account needed, unlike Weights and Biases.

**Harness image transport** means `line_follower` now works over the
launched-simulation (`harness`) backend, not just in-process:
`HarnessVecEnv` subscribes to each agent's `/rl/camera_i` topic directly,
the same way the in-process backend does, so GUI and vision now work
together. See `docs/examples/line_follower.md`.

**Other completed infrastructure** includes a single-Python environment
through Pixi and RoboStack, ending the dual-Python split; a generalized,
parameterized `MultiAgentGazeboVecEnv` and `make_multi`; the batched
in-sim harness, `HarnessCore` (ECM), the `MultiAgentHarness` plugin (O(1)
transport), and the `HarnessVecEnv` client, all with in-place reset; the
in-process backend (`InProcessHarnessVecEnv`), headless training with no
launch, where `<real_time_factor>0</real_time_factor>` unthrottles it to
roughly 88 times up to roughly 4,600 agent-steps per second at N=16; a
hyperparameter sweep (`training_scripts/sweep.py`, CSV plus optional
Weights and Biases) verifying cartpole to the 500-step cap; per-agent
independent same-step autoreset on the in-process backend, giving correct
vectorized-reinforcement-learning episode boundaries rather than a shared
group reset; standard `gymnasium.Env` registration
(`gymnasium.make("GazeboCartPole-v0")`) passing
`gymnasium.utils.env_checker` for universal tool interoperability; a
native `gymnasium.vector.VectorEnv` (`gymnasium.make_vec(...)` through a
`vector_entry_point`), the efficient N-in-one simulation as a standard
vector environment with same-step autoreset and `final_obs` info; verified
compatibility with ecosystem wrappers, SB3's `VecNormalize` and
Gymnasium's `RecordEpisodeStatistics`/`NormalizeObservation`; verified
determinism, where the same seed and actions produce bit-identical
trajectories, since DART stays deterministic and the seed threads through
reset randomization; population-based domain randomization
(`AgentSpec.mass_randomization`, `action_gain_randomization`), where each
of the N agents draws a different dynamics sample, seed-reproducible, and
a plus-or-minus-20-percent gain collapses a nominal policy from 500 down
to roughly 18; force-controlled CartPole, the textbook actuation, where
`set_force` was never actually broken, since the cart's collision box
rested on the ground plane at the old `spawn_z=0.10` and contact friction
pinned it (velocity control silently overrode the contact, but force
could not), fixed by spawning clear of the ground (`spawn_z=0.60`), after
which a random policy survives roughly 7 steps, PPO solves to the 500
cap, and mass DR now genuinely bites, since force equals mass times
acceleration; and a flake8 and pep257 clean state across the package
under the project configuration.
