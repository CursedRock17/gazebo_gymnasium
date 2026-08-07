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

- **Hugging Face Hub integration.** _Status: NOT STARTED — no Hub code exists
  in the repo today._ Uploading a trained policy currently means reaching for
  the `huggingface-cli` by hand, outside the repo; nothing in `train.py` /
  `deploy.py` / `sweep.py` pushes or pulls from the Hub. The goal is to make
  that a first-class, in-repo capability so *anyone* running this project can
  share and fetch models without hand-rolling it.

  Planned work (all to be built):
  - Add `--push-to-hub <repo_id>` to `train.py`: after training, upload the
    model and auto-generate a **model card** documenting the algorithm, the
    exact hyperparameters, the environment id, and the evaluated mean episode
    reward against the [solved bar](docs/examples/README.md).
  - Add `--from-hub <repo_id>` to `deploy.py`: download and run a Hub model.
  - Extend `sweep.py` to push the **best** config with its learning curve, so
    the hosted model carries the precise hyperparameters that produced it.
  - Add `huggingface_sb3` + `huggingface_hub` to the pixi env (a `pixi.lock`
    re-solve); auth via a Hub token. SB3's official path is
    `huggingface_sb3.package_to_hub` / `load_from_hub`.
  - Insertion points already identified (but not yet wired): models are saved
    at `train.py` `final_path` and loaded at `deploy.py`; the sweep's
    hyperparameters live in its `CONFIGS` + result CSV.
  - Honest scope: the Hub *stores the SB3 `.zip`* — that is the model format —
    so this hosts/versions/documents the zip rather than eliminating it;
    downloading still yields a `.zip`. `package_to_hub`'s replay video needs
    rendering (the camera/GUI path, subject to the one-camera-env-per-process
    limit), so the first slice should push model + card + eval metrics and make
    the video opt-in.


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
- **Port the remaining MuJoCo envs to `AgentSpec`s.** Done so far:
  `inverted_double_pendulum`, `hopper` (both learnability-verified headless;
  `cartpole_continuous` covers InvertedPendulum). The hopper established the
  planar-joint-root recipe + MuJoCo armature/damping emulation (see the
  porting notes in docs/creating_your_own_agent.md), so `walker2d`,
  `half_cheetah` (after its SDF frame fix), and `swimmer` are mechanical next
  steps. `reacher`/`pusher` need target bodies (target-as-joints trick);
  `ant`/`humanoid` need a link-state obs extension for their 3D free bases.
- **`half_cheetah`** — hand-rename the duplicate frame names in the converted
  SDF so it loads, then give it a spec.
- **Line follower: reach the solved bar.** The camera extension is BUILT
  (`image_obs`, per-agent camera topics, pose-restoring resets — see the
  guide) and the vision pipeline learns decisively, but policies so far master
  the straight and fail the first 90° corner. Tried: frame stacking
  (VecFrameStack 4) — at a 100k budget it did NOT beat the single-frame
  plateau (ep_len 66 vs 80), and CPU PPO+CNN on a 12-channel stack runs at
  only ~4-8 env-steps/s (gradient updates dominate). Next levers, in order:
  GPU training on an NVIDIA machine (GAZEBO_GYM_DEVICE=cuda is wired; pixi.lock
  ports the env), corner-aware reward shaping or a rounded-corner track
  variant, grayscale/2-frame stacks to cut compute.

## GPU acceleration — the full story (audited 2026-07)

| Layer | GPU possible? | Our status |
|---|---|---|
| Physics (DART) | No — upstream: gz-physics has no GPU engine | CPU by design; mitigated by N-agents-in-one-sim + real_time_factor=0 |
| Camera rendering (ogre2) | Yes | **Already on the GPU** — verified `GL_RENDERER = Mesa Intel Graphics (MTL)` via headless EGL; the ~180 ticks/s vision ceiling is GPU-rendered (readback + sensor pipeline overhead, not raster) |
| Learner (PyTorch/SB3) | Yes, NVIDIA only | Wired (`GAZEBO_GYM_DEVICE=cuda`); dev laptop has no NVIDIA device, so CNN training is CPU-bound here |
- **Harness image transport** — cameras currently flow through the in-process
  backend only; carrying frames to the launched-sim client would enable
  GUI + vision together (per-agent topic rewrite in the spawner + image
  subscription in HarnessVecEnv).

- **peragent backend dynamics.** The legacy per-agent backend still drives the
  controller-equipped `cartpole` model with velocity commands (JointController),
  so its dynamics differ from the force-based spec the ECM backends
  (inprocess/harness) use — policies don't transfer to it. Either port it to
  `gz-sim-apply-joint-force-system` or retire it once the harness backend is
  live-validated.

## Quality / infrastructure

- **CI**: `.github/workflows/ci.yml` runs the pixi build + full suite +
  linters on Ubuntu Noble (the REP-2000 Tier 1 platform), on every push/PR,
  nightly, and on demand. Remaining: its first green run on GitHub — the last
  gating item for [Quality Level 3](QUALITY_DECLARATION.md).
- **Performance benchmarks**: `training_scripts/benchmark.py` publishes
  steps/second per environment and scaling curves across `n_agents`
  (`--scale 1,4,16,32`, CSV output). Still TODO: a comparison run against
  canonical MuJoCo, and the same numbers for the launched-sim backends.
- **Read the Docs**: `.readthedocs.yaml` + `docs/requirements.txt` are in
  place and the Sphinx API reference builds warning-free. Remaining: connect
  the repository on readthedocs.org and replace the CursedRock17/gazebo_gymnasium placeholders.

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
- **Force-controlled CartPole** — the textbook actuation. `set_force` was never
  broken: the cart's collision box rested on the ground plane at the old
  `spawn_z=0.10` and contact friction pinned it (velocity control silently
  overrode the contact; force could not). Spawning clear of the ground
  (`spawn_z=0.60`) fixed it. Random policy ~7 steps, PPO solves to the 500 cap
  — and mass DR now genuinely bites (F = ma).
- flake8 + pep257 clean across the package under the project config.
