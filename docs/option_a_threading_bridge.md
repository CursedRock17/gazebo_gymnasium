# Option A — Threading Bridge (Advanced)

> **For most users, use Option B (the default).** Option B is documented in the
> top-level `README.md` and exemplified by `cartpole_learner.py`. It works with
> any RL library out of the box. Option A exists for users who need lower
> per-step overhead or want the simulator to drive the training loop.

## TL;DR

There are two ways the plugin can connect a Gazebo simulation to a Gymnasium-
style agent:

| | Option B (default) | Option A (this doc) |
|---|---|---|
| Who owns the loop | Agent / RL library | Gazebo |
| Per-step overhead | ~1 ms (gz-transport service round-trip) | ~µs (in-process event signal) |
| User writes | A plain `gym.Env` | A Gazebo system plugin |
| Threading | None (agent thread only) | Required for libraries that own the loop (e.g. SB3) |
| Works with any RL library | Yes, trivially | Yes for inline (custom code); requires care for libraries like SB3 |
| Hardest part | Nothing | Race conditions if you do it wrong |

Pick Option A only if you've measured per-step overhead and it actually matters
for your problem. For CartPole-scale problems, the per-step overhead is dwarfed
by training updates and rendering — the speed argument is theoretical.

## When Option A is the right call

- You have a high-frequency control loop (>1 kHz) where 1 ms transport latency
  per step is genuinely meaningful.
- You're embedding a custom agent that's already structured to run inline (no
  external "owner of the loop" — just a forward pass per tick).
- You're prototyping inside a Gazebo system plugin and don't want to spin out
  a separate process for the agent.

## The two sub-patterns

### A.1 — Inline (custom synchronous agent)

The simplest form. Agent code lives inside `select_action()` and is called
synchronously from `pre_update()`. No threads. Works only if your agent step is
"fast" (no blocking calls, no internal training loop).

```python
import gazebo_gymnasium_bridge.plugins.gazebo_single_agent as gazebo_single_agent

class MyAgent(gazebo_single_agent.GazeboSingleAgent):
    def __init__(self):
        # ... obs_space, action_space, transport pubs/subs ...
        super().__init__(self.observation_space, self.action_space)

    def select_action(self):
        # Called by base_env.pre_update() each tick. Must return quickly.
        obs_tensor = torch.from_numpy(np.asarray(self.observation, dtype=np.float32)).unsqueeze(0)
        probs = self.actor_net(obs_tensor)
        action = torch.distributions.Categorical(probs).sample().item()
        return [action]

    def apply_action(self, action):
        # Publish the action via transport.
        ...

    def get_observation(self): ...
    def get_reward(self, action): return 1
    def is_terminated(self): ...
    def is_truncated(self): return False
    def get_info(self): return None
    def set_default_observation(self): return [0.0] * 4


def get_system():
    return MyAgent()
```

The base class (`GazeboBaseEnv`) handles the rest: it calls `select_action()`
in `pre_update`, applies the action, runs physics, then calls `step()` in
`post_update` which collects observation + reward + termination via your
`get_observation()` / `get_reward()` / `is_terminated()` methods.

This is what the original `cartpole_learner.py` looked like before the switch
to Option B. See the git history for the full file.

### A.2 — Threading bridge (library that owns the loop)

Required when the agent library — SB3, RLlib, Tianshou — expects to *call*
`env.step(action)` itself in its own loop. The library's loop and Gazebo's
loop are two separate control flows; you bridge them with `threading.Event`s.

The core idea:

```
SB3 thread                          Gazebo main thread
─────────                          ──────────────────
env.step(action) called
  ├─ self._pending_action = action
  ├─ self._tick_event.clear()
  └─ self._tick_event.wait()  ←──╮
                                  │  pre_update fires
                                  │    └─ reads self._pending_action,
                                  │       publishes it
                                  │  physics runs
                                  │  post_update fires
                                  │    └─ self._tick_event.set()  ─╮
                                  │                                │
  ←──────────────────────────────────────────────────────────────  ╯
  ├─ read self.observation
  └─ return (obs, reward, term, trunc, info)
```

Important: **for correctness under variable agent speed**, you need a *second*
event (`_action_ready`) and have `post_update` wait on it before continuing.
Otherwise Gazebo keeps ticking with stale actions if SB3's policy is slower
than physics. The single-event version below is the simplest form; promote
to two events if you observe stale-action behavior.

Reference implementation skeleton:

```python
import threading
import numpy as np

import gazebo_gymnasium_bridge.plugins.gazebo_single_agent as gazebo_single_agent

class MyAgentSB3(gazebo_single_agent.GazeboSingleAgent):
    def __init__(self):
        # ... obs_space, action_space, transport pubs/subs ...
        self._tick_event = threading.Event()
        self._reset_event = threading.Event()
        self._pending_action = None
        self._sb3_thread = None
        super().__init__(self.observation_space, self.action_space)

    def configure(self, entity, element, ecm, eventManager):
        super().configure(entity, element, ecm, eventManager)
        # Spawn SB3 once the sim is configured.
        self._sb3_thread = threading.Thread(target=self._train, daemon=True)
        self._sb3_thread.start()

    def _train(self):
        from stable_baselines3 import PPO
        PPO("MlpPolicy", self).learn(total_timesteps=200_000)

    # === Gazebo lifecycle (drives the sim, signals SB3) ===

    def pre_update(self, info, ecm):
        # Do NOT call super().pre_update() — we drive actions from SB3.
        if info.paused or self.needs_reset:
            return
        if self._pending_action is not None:
            self.apply_action([self._pending_action])

    def post_update(self, info, ecm):
        if info.paused:
            return
        self.current_step += 1
        # Set termination flags BEFORE signaling SB3.
        self._terminated = self.is_terminated()
        self._truncated = self.current_step >= self.max_steps_per_episode
        self._tick_event.set()

    def gazebo_reset(self, info, ecm):
        # Base class does the episode bookkeeping; we just unblock SB3's reset().
        super().gazebo_reset(info, ecm)
        self._reset_event.set()

    # === Gymnasium Env API (called by SB3.learn() from worker thread) ===

    def reset(self, seed=None, options=None):
        # base_env.reset() dual-dispatches by argument types — preserve that.
        if seed is not None and not isinstance(seed, int):
            return super().reset(seed, options)  # Gazebo Reset event path
        # SB3 path: trigger a world reset, wait for it to land.
        self._reset_event.clear()
        self.world_control.reset()
        self._reset_event.wait(timeout=10.0)
        return np.array(self.get_observation(), dtype=np.float32), {}

    def step(self, action):
        self._pending_action = int(action)
        self._tick_event.clear()
        self._tick_event.wait(timeout=10.0)
        return (np.array(self.get_observation(), dtype=np.float32),
                1.0, bool(self._terminated), bool(self._truncated), {})
```

## Pitfalls

1. **Don't call `super().pre_update()` in A.2** — it invokes `select_action()`
   and the base-env step pipeline, which is what your SB3 thread is supposed to
   own.
2. **The `reset()` dual-dispatch is fragile.** `GazeboBaseEnv.reset()` checks
   argument types to distinguish "Gazebo Reset event" (`reset(info, ecm)`) from
   "Gymnasium agent call" (`reset(seed, options)`). If you forget to forward
   the Gazebo path to `super().reset()`, your episode counters break silently.
3. **Stale actions when the agent is slower than Gazebo.** With
   `real_time_update_rate=0`, Gazebo runs as fast as it can. If your policy
   forward pass takes 20 ms and one physics step is 0.1 ms, Gazebo will tick
   200 times with the same action between SB3 calls. The two-event version
   fixes this by making `post_update` wait for the next action before
   returning.
4. **Daemon threads die silently.** If SB3 crashes in the worker thread, you
   get one traceback in stdout and then nothing — Gazebo keeps running. Wrap
   your trainer body in `try/except` and log on exit.
5. **gz-transport callbacks are async.** Even with the threading bridge, your
   pose/joint_state subscribers fire on transport's own thread. The data you
   read in `step()` was captured at most one tick ago, sometimes two. If
   determinism matters, query the ECM directly from `post_update` instead of
   subscribing to topics — that's synchronous with physics.

## Why Option B is the default

Option B treats Gazebo as a "stepped backend" — the agent calls
`world_control.step()` to advance the sim explicitly. The result is a plain
`gym.Env` subclass that any RL library accepts unchanged. No threading, no
lifecycle override, no dual-dispatch reset.

You give up the in-process event-signaling speed advantage, but at CartPole
scale (and most realistic robotics scales) the per-step transport latency is a
small fraction of total wall-clock time. The simplicity wins.

If you measure your workload and Option B's overhead is actually limiting,
come back here.

## Reference

- `gazebo_gymnasium_bridge/plugins/gazebo_single_agent.py` — base class for
  Option A plugins (provides `pre_update` / `post_update` / `select_action`
  contracts).
- `gazebo_gymnasium_bridge/plugins/gazebo_base_environment.py` — the shared
  state machine + `GazeboBaseEnv.reset()` dual-dispatch + `gazebo_reset` hook.
- For a full working A.1 example, check out the git history of
  `cartpole_learner.py` before the Option B refactor.
