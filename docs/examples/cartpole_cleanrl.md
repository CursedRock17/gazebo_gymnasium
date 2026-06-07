# CartPole with CleanRL-style PPO

Same Gazebo CartPole environment used in [the SB3 example](cartpole.md) and
[the project's custom PPO](cartpole.md#custom-ppo-vs-sb3-ppo), driven this
time by a **CleanRL-style single-file PPO**. This example exists to show
that the `GazeboCartPoleEnv` class is library-agnostic — it works with any
trainer that consumes the Gymnasium API.

## What is CleanRL?

[CleanRL](https://github.com/vwxyzjn/cleanrl) is an RL library with a strict
"one file per algorithm" rule. There are no abstractions to learn — every
hyperparameter is visible in one place and every line of the algorithm is
right next to its caller. It's the opposite end of the design spectrum from
Stable Baselines3's class hierarchy.

If you've ever found yourself printing class internals to figure out what
SB3 is doing, this style is for you. The training script
[`train_cartpole_cleanrl.py`](../../training_scripts/train_cartpole_cleanrl.py) is ~150
lines, all in one place, and adapts the canonical
[`cleanrl/ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
to drive our env directly (no `gym.make` ceremony, no `SyncVectorEnv` —
just `GazeboCartPoleEnv()`).

## Running it

```bash
# Terminal 1 — simulator
ros2 launch gazebo_gymnasium_bringup cartpole.launch.py

# Terminal 2 — CleanRL-style PPO
python training_scripts/train_cartpole_cleanrl.py
```

## What's different from the SB3 version

| Aspect | SB3 PPO | CleanRL PPO (this script) |
|--------|---------|--------------------------|
| Lines of code | ~5 (caller) | ~150 (entire algorithm) |
| Where is the loss computed? | Hidden inside `PPO.train()` | Inline in `main()` |
| How to tweak the policy? | Subclass `ActorCriticPolicy` | Edit the `Agent` class |
| Vectorization | Auto-wrapped in `DummyVecEnv` | None — single env |
| Logger | TensorBoard via callback | `print()` per update |

For a class assignment or research where you need to **understand every
update step**, CleanRL wins. For shipping production policies on multiple
envs, SB3 wins. Same gym.Env serves both.

## Hyperparameter notes

The defaults in [`train_cartpole_cleanrl.py`](../../training_scripts/train_cartpole_cleanrl.py)
come from CleanRL's canonical CartPole tuning:

| Param | Value | Rationale |
|-------|-------|-----------|
| `TOTAL_TIMESTEPS` | 200,000 | More than enough for CartPole — typically solved by step 20k |
| `NUM_STEPS` | 128 | Rollout length per update |
| `GAMMA` | 0.99 | Discount |
| `GAE_LAMBDA` | 0.95 | GAE smoothing |
| `LEARNING_RATE` | 2.5e-4 | Adam LR |
| `CLIP_COEF` | 0.2 | PPO clipping |
| `ENT_COEF` | 0.01 | Entropy bonus (CleanRL default — note our custom PPO uses 0.001) |
| `NUM_UPDATE_EPOCHS` | 4 | Epochs over each rollout |
| `MINIBATCH_SIZE` | 32 | SGD minibatch size |

If you tune any of these, edit the constants at the top of the file
directly — there's no `--lr` flag, by CleanRL design.

## Why this matters

A well-designed Gymnasium env should not care which trainer is driving it.
The same `GazeboCartPoleEnv` class is exercised by:

- `train_cartpole_sb3.py` — SB3 PPO
- `train_cartpole_custom.py` — the project's own from-scratch PPO
- `train_cartpole_cleanrl.py` — CleanRL-style single-file PPO

If any of those three fail while the others pass, the bug is in the trainer
— not in the env. That property is the value of writing a proper `gym.Env`
in the first place.

## Reference files

- Training script:
  [`train_cartpole_cleanrl.py`](../../training_scripts/train_cartpole_cleanrl.py)
- Env class (shared with SB3 + custom):
  [`bridge/envs/cartpole.py`](../../gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/cartpole.py)
- Original CleanRL source:
  [`vwxyzjn/cleanrl/cleanrl/ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
