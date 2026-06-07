# Pipelined inference / action chunking

User asked in TASKS.md: "In an attempt to squeeze more compute during
the process, maybe it would be best execute inference after the physics
steps, or try to do some sort of queueing, similar to how action chunking
works."

Here's the analysis and what we built.

## The current synchronous pattern

```
   trainer wall-clock                 gz sim wall-clock
   ──────────────                      ─────────────
   t=0   env.step(a_t)  ──publish──>  t=0   /env/action received
                                            ┃  run frame_skip ticks
                                            ┃  (~10 ms physics)
   t=10  <──/env/state── env.step returns   t=10
   t=10  policy.predict (~1 ms)
   t=11  optional gradient (~10–50 ms)
   t=61  env.step(a_{t+1}) ──publish─>  t=61 /env/action received
                                             ┃
```

Total wall time per step: ~60 ms. Trainer is BUSY for ~51 ms, gz sim is
BUSY for ~10 ms, but they don't overlap. So roughly 51 + 10 = 61 ms.

## What overlap could buy us

If the trainer's policy + gradient compute could run DURING the physics
tick, total per-step wall time would drop toward `max(51, 10) = 51 ms`
— ~17% wall-clock win. Real, but smaller than the 5–10× headless win or
the 4–16× multi-agent win.

## What "action chunking" actually means

Two distinct interpretations of the user's note:

**A. Pipeline policy & gradient with physics** (this doc's main thread):

Compute `a_{t+1}` *immediately* after step_async publishes `a_t`. While
gz sim runs the physics for `a_t`, the trainer is already computing the
gradient update for the buffered transitions OR predicting `a_{t+1}` so
it can publish without waiting. Action published at t=0, predict happens
during t=0–10, state received at t=10, step returns, publish `a_{t+1}`
immediately at t=10 (already computed). Saves the predict latency
(~1 ms) per step. Saves more if a long gradient update can run
concurrently with physics — but PyTorch isn't really async; the GIL
+ a synchronous SAC training loop hold the train thread.

**B. Action chunking (the Diffusion Policy / RT-1 style)** — predict
N future actions at once and execute them open-loop:

```
   env.step(a_t, a_{t+1}, a_{t+2}) ──publish a_t─>  gz runs physics
                                   ──publish a_{t+1}─>  gz runs physics
                                   ──publish a_{t+2}─>  gz runs physics
   <── /env/state at t+3 ──         (one return covers 3 ticks)
```

Now the trainer pays ONE policy-inference cost for 3 env steps. Big win
on policy-inference latency, but: the policy is trained per-step (the
gradient still wants single-step transitions), so we'd need an actor
that outputs an N-step trajectory (Decision Transformer / Diffusion
Policy territory).

## What we actually built

Approach **A** as a thin wrapper. Approach **B** is a bigger architectural
shift (different policy class, training objective changes); deferred.

See `gazebo_gymnasium_bridge/backend/pipelined_step.py` for the helper.
Pattern: it wraps an existing single-env's `step()` to do `step_async`
+ `step_wait`-style separation. Inside the wait, callers can interleave
gradient work via a callback.

For SB3 vec envs, this is already baked in — `VecEnv.step_async/step_wait`
is the standard pattern. The `MultiCartPoleVecEnv` we built earlier
uses it. So the SAC / PPO scripts get partial overlap for free when
they're running against any VecEnv (vs a single Env).

## Recommended order of operations for max throughput

1. **Run headless** (`headless:=true` in the launch file). 5–10× win.
2. **Use the multi-agent path** when available
   (`train_cartpole_multi_agent.py`). 4–16× more steps/wall-sec.
3. *Then* consider pipelining + chunking. ~20% on top of the above.

Items 1+2 dominate; item 3 is polish.
