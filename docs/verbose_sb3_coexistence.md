# Verbose Debug + SB3 Logging — Coexistence Audit

## What each side prints

**Our env (`cartpole_env.py`):**

- Per step (when `GAZEBO_GYM_VERBOSE=true`):
  ```
  [CartPole][ep 5 step 67] action=0 reward=1.0 cum=67 cart=(p=+0.024, v=-0.122) pole=(a=-0.211, av=-0.803)
  ```
- Per reset (always):
  ```
  [EpisodeSummary] ep=5 steps=67 reward=67
  ```

**Our PPO agent (`PPO_agent.py`):**

- Per training batch (when buffer reaches `batch_size`):
  ```
  [PPOTrain] buffer=256 episodes_in_buffer=4 actor_loss=-0.0014 value_loss=1100.4049 entropy=0.692
  ```

**SB3 PPO with `verbose=1`:**

- Every `n_steps` (default 256) rollout, a multi-line table:
  ```
  -------------------------------------
  | rollout/                |          |
  |    ep_len_mean          | 67.5     |
  |    ep_rew_mean          | 67.5     |
  | time/                   |          |
  |    fps                  | 145      |
  |    iterations           | 5        |
  |    total_timesteps      | 1280     |
  | train/                  |          |
  |    approx_kl            | 0.024    |
  |    clip_fraction        | 0.103    |
  |    entropy_loss         | -0.66    |
  |    explained_variance   | -0.0123  |
  |    policy_gradient_loss | -0.0028  |
  |    value_loss           | 28.4     |
  -------------------------------------
  ```

## Interleaving behavior

With **default `GAZEBO_GYM_VERBOSE=true`**, an SB3 run produces roughly:

| Frequency | Source | Lines/event |
|-----------|--------|-------------|
| Every step (~145/sec) | `[CartPole]` | 1 |
| Every ~70 steps | `[EpisodeSummary]` | 1 |
| Every 256 steps | SB3 table | ~14 |

So per second you get ~145 `[CartPole]` lines and the SB3 table about every 2 seconds. The table is buried in the stream — not readable in real time, but `grep '|'` on a log file still finds it.

## The fix: throttle env var

Added `GAZEBO_GYM_VERBOSE_EVERY=N` (default 1). Set it to a larger N to skip per-step prints:

```bash
# Use case A: maximum detail (custom PPO debugging) — same as before
ros2 launch gazebo_gymnasium_bringup cartpole.launch.py verbose:=true

# Use case B: SB3 run, keep SB3's table readable, sample env state every 50 steps
GAZEBO_GYM_VERBOSE_EVERY=50 ros2 launch gazebo_gymnasium_bringup cartpole_sb3.launch.py verbose:=true

# Use case C: silence env per-step prints entirely, SB3 table is the source of truth
ros2 launch gazebo_gymnasium_bringup cartpole_sb3.launch.py verbose:=false
```

`[EpisodeSummary]` and `[PPOTrain]` lines are NOT throttled — they're already at the right cadence (one per episode and one per training step respectively).

## Recommendations by mode

- **Custom PPO + local interactive debug** → `verbose:=true` (the default). All streams useful.
- **SB3 + watching it learn** → `verbose:=true` with `GAZEBO_GYM_VERBOSE_EVERY=50` or `100`. Keeps SB3's table the focal point, drops occasional state samples.
- **SB3 + headless training run** → `verbose:=false`. SB3's table + `[EpisodeSummary]` is enough for offline analysis.
- **Anything piped through `tee` for later grep-analysis** → `verbose:=true`. Disk is cheap, you'll want the data.

## Conflict check: do any lines collide on the same prefix?

| Prefix | Owner |
|--------|-------|
| `[CartPole]` | env |
| `[EpisodeSummary]` | base env in `gazebo_env.py` |
| `[PPOTrain]` | custom PPO in `PPO_agent.py` |
| `[GazeboEnv]` | env init message |
| `[GazeboEnvRunner]` | plugin shell |
| `[CartPoleSB3]` | SB3 trainer worker |
| `\|` table lines | SB3 |
| `Logging to ...` | SB3 monitor wrapper |

No collisions. All our lines start with `[X]` brackets; SB3's table lines start with `|` or `-`. Grep-able independently.
