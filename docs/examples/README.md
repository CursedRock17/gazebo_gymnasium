# Examples

Each tutorial here shows a full RL training pipeline: launch a Gazebo world,
run a `gym.Env` against it, train an agent. Pick whichever example matches the
problem you want to learn from.

| Example | Algorithm | Action space | Observation space | Status |
|---------|-----------|--------------|-------------------|--------|
| [CartPole](cartpole.md) | PPO (SB3 + project's custom impl) | Discrete(2) | Box(4,) float32 | ✅ Solved |
| [CartPole — CleanRL style](cartpole_cleanrl.md) | PPO (single-file CleanRL impl) | Discrete(2) | Box(4,) float32 | ✅ Working — same env, different trainer to demo library-agnostic design |
| [InvertedPendulum](inverted_pendulum.md) | SAC | Box(1,) in [-3, 3] | Box(4,) float32 | ✅ Working — first continuous-action env, ported from MuJoCo MJCF |
| [InvertedDoublePendulum](inverted_double_pendulum.md) | A2C | Box(1,) in [-1, 1] | Box(8,) float32 | ✅ Working — two-link variant, auto-converted from MJCF |
| (planned) Reacher | TD3 | Box(2,) | Box(8,) | Stub plugin + launch live; env class TODO |
| (planned) HalfCheetah | SAC | Box(6,) | Box(17,) | Blocked — duplicate frame names in auto-converted SDF need manual rename |
| (planned) Ant / Hopper / Walker2d / Humanoid | SAC or TD3 | Box(n,) | Box(m,) | Stub plugins + launches live; env classes TODO |

## What's common across examples

All examples follow the same architecture pattern:

1. **The world SDF** lives in
   [`gazebo_gymnasium_examples/gazebo_gymnasium_resources/worlds/`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/worlds/)
   and references a **sync-gate plugin** via `PythonSystemLoader`.

2. **The sync-gate plugin** (in
   [`gazebo_gymnasium_examples/gazebo_gymnasium_resources/plugins/`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/plugins/))
   is the only Python code that runs inside `gz sim`. Its only job: subscribe to
   `/env/action`, count `frame_skip` ticks, publish `/env/state` with the new
   sensor values. It does NOT contain training logic.

3. **The env class** (in
   [`gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/`](../../gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/))
   is a plain `gym.Env` that communicates with the plugin over gz-transport.
   Any RL library that speaks Gymnasium (SB3, RLlib, CleanRL, custom) drives
   this env without modification.

4. **The training script** is a normal Python file under
   [`training_scripts/`](../../training_scripts/)
   (`train_cartpole_sb3.py`, `train_cartpole_custom.py`, etc.). It instantiates
   the env and calls the agent library's `.learn(...)` method.

If you're adding a new environment, copy the CartPole files and adjust:
- Model SDF (the robot you're controlling)
- World SDF (which loads the model + sync-gate plugin)
- Sync-gate plugin (how an action becomes a cmd_vel/force/torque, which links
  to read state from)
- Env class (observation/action spaces, reward function, terminated condition)
- Training script (algorithm + hyperparameters)

## Saving and deploying a trained policy

All three SB3 training scripts now save checkpoints automatically:

- Periodic snapshots every N steps via `CheckpointCallback` to
  `models/<env>/<algo>_<env>_<steps>_steps.zip`
- A final `models/<env>/final.zip` saved on clean exit OR on Ctrl-C (the
  `try/finally` block protects long runs)
- Re-running the training script auto-resumes from `final.zip` if it
  exists. Delete the file to start over from scratch.

Once you have a checkpoint, evaluate it with `training_scripts/deploy_policy.py`:

```bash
# Default: deterministic rollout for 3 episodes against the saved final model
python training_scripts/deploy_policy.py --env inverted_pendulum

# Specific checkpoint, more episodes, sample from policy distribution
python training_scripts/deploy_policy.py --env inverted_pendulum \
    --checkpoint models/inverted_pendulum/sac_inverted_pendulum_30000_steps.zip \
    --episodes 10 --stochastic
```

`deploy_policy.py` is one parameterized script — adding a new env means
appending one row to its `ENVS` dict.

## Watching the sim in Foxglove (FPS, RTF, agent metrics)

Each launch file bridges five gz topics to ROS when `use_foxglove:=true`:
`/tf`, `/joint_states`, the world `/stats` (RTF + iterations),
`/clock`, and `/env/metrics`. The metrics array carries
`[steps_per_sec, mean_step_ms, episode_reward, current_step, total_steps,
current_episode]` — plot `data[2]` against `data[4]` to see the canonical
**reward-vs-training-steps learning curve** straight in Foxglove. See any
individual example's "Watching in Foxglove" section for the full panel
layout.

## Why two terminals?

The recommended flow is:

```bash
# Terminal 1
ros2 launch gazebo_gymnasium_bringup cartpole.launch.py

# Terminal 2
python training_scripts/train_cartpole_sb3.py
```

Two terminals make it easy to swap the trainer without restarting the sim,
which keeps a long-running window of training and visualization. Use
`cartpole_train.launch.py` if you prefer the one-command flow — it starts both
in the same launch.

## Background docs (deeper dives)

- [`option_a_threading_bridge.md`](../option_a_threading_bridge.md) — the
  alternative architecture (plugin owns the loop). Documented for completeness;
  the working examples use the "external env" pattern instead.
- [`gymnasium_api_reference.md`](../gymnasium_api_reference.md) — extracted
  Gymnasium API surface. Useful for porting envs from other simulators.
- [`sb3_api_reference.md`](../sb3_api_reference.md) — what SB3 actually does to
  your env (Monitor wrapping, DummyVecEnv auto-reset, etc.). Read this before
  debugging SB3 weirdness.
- [`verbose_sb3_coexistence.md`](../verbose_sb3_coexistence.md) — how
  per-step logging interacts with SB3's verbose table.

## Screenshots

Drop reference images in [`../resources/`](../resources/) and link them inline
in each tutorial.
