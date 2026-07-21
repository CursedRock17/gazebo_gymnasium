# Stable-Baselines3 API Reference

Extracted from `DLR-RM/stable-baselines3` @ commit `10dda86` (2026-05-11, current `master`),
version `2.9.0a2`. Local installed version: `2.7.0`.
Source: https://github.com/DLR-RM/stable-baselines3.git

The goal of this doc is to trace **exactly** what happens between
`PPO("MlpPolicy", our_env).learn(...)` and our env's `step` / `reset` methods,
so when something misbehaves we know whose contract to suspect.

---

## What `PPO("MlpPolicy", env)` actually does to our env

`PPO.__init__` → `BaseAlgorithm.__init__` → `_wrap_env(env)` at
`base_class.py:204-249`. The wrap pipeline:

```
our env (gym.Env)
    │
    ▼   _patch_env  (handles legacy gym → gymnasium API translation)
    │
    ▼   Monitor(env)        ← unless monitor_wrapper=False
    │
    ▼   DummyVecEnv([lambda: env])
    │
    ▼   (optional) VecTransposeImage  ← only for image observations
    │
    ▼
final self.env  (always a VecEnv)
```

So after `PPO(...)`, our env is buried inside:

```
DummyVecEnv ─▶ Monitor ─▶ CartPoleEnv (us)
```

Two wrappers we don't directly see but absolutely interact with at runtime.

---

## How rollouts drive `step()` / `reset()`

### Initial reset (`base_class.py:421-423`, in `_setup_learn`)

Called **once** at the start of `learn()`, only if `self._last_obs is None`:

```python
if reset_num_timesteps or self._last_obs is None:
    self._last_obs = self.env.reset()
```

This is `VecEnv.reset()`, which calls each underlying `env.reset(seed=...)`
exactly once. After this, SB3 **never** calls reset externally — done.

### The rollout loop (`on_policy_algorithm.py:194-256`)

```python
while n_steps < n_rollout_steps:           # n_rollout_steps = our n_steps=256
    with th.no_grad():
        actions, values, log_probs = self.policy(obs_tensor)
    actions = actions.cpu().numpy()         # shape (num_envs,) for Discrete

    # ... potential clipping for Box action spaces — N/A for Discrete ...

    new_obs, rewards, dones, infos = env.step(clipped_actions)   # ← our step
    self.num_timesteps += env.num_envs

    # Truncation bootstrap (the "TimeLimit handling")
    for idx, done in enumerate(dones):
        if done and infos[idx].get("terminal_observation") is not None \
           and infos[idx].get("TimeLimit.truncated", False):
            terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
            with th.no_grad():
                terminal_value = self.policy.predict_values(terminal_obs)[0]
            rewards[idx] += self.gamma * terminal_value
            # ↑ correctly bootstraps V(s_T) into the reward when truncated

    rollout_buffer.add(self._last_obs, actions, rewards,
                       self._last_episode_starts, values, log_probs)
    self._last_obs = new_obs
    self._last_episode_starts = dones
```

**Note the action flow:** `actions` is a numpy ndarray of shape `(num_envs,)`,
`dtype=int64` for Discrete. With `num_envs=1` (DummyVecEnv around our single
env), `actions` is shape `(1,)` containing one `np.int64`. SB3 passes the whole
array `clipped_actions` to `env.step()` — but `env` here is the `VecEnv`, not
our env. The `VecEnv` does the per-env indexing internally.

### `DummyVecEnv.step_wait` (`dummy_vec_env.py:56-73`) — the auto-reset

```python
def step_wait(self):
    for env_idx in range(self.num_envs):
        obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = \
            self.envs[env_idx].step(self.actions[env_idx])      # ← our step
        self.buf_dones[env_idx] = terminated or truncated
        self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

        if self.buf_dones[env_idx]:
            self.buf_infos[env_idx]["terminal_observation"] = obs
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()  # ← auto-reset
        self._save_obs(env_idx, obs)
    return (self._obs_from_buf(), np.copy(self.buf_rews),
            np.copy(self.buf_dones), deepcopy(self.buf_infos))
```

Critical implications:

1. **SB3 NEVER calls our reset() externally during training.** Auto-reset
   happens inside `step_wait` whenever our `step` returns `terminated or
   truncated`. This is why the 1-step-per-episode bug was so painful — the
   auto-reset was using an env that thought it was already terminated.

2. **`info["TimeLimit.truncated"]` is set by DummyVecEnv, NOT us.** We set
   it ourselves in `GazeboEnv.step()`, but DummyVecEnv overwrites that line
   unconditionally. Harmless duplication — we can stop setting it, or keep it
   for when the env is used outside SB3.

3. **`info["terminal_observation"]` is added on done.** This is what powers
   PPO's truncation bootstrap (above). If we ever start producing
   `terminated=True` (real termination), this still gets set but the bootstrap
   only fires when `TimeLimit.truncated=True`.

4. **Action passed to our step is `self.actions[env_idx]`** — for our single-env
   case, that's an `np.int64` scalar (not an ndarray). Our `apply_action` handles
   this: `isinstance(action, np.ndarray)` is False for `np.int64`, so we fall
   into `int(action)`, which works.

### `Monitor.step` (`monitor.py:87-113`)

Wraps `step` to track episode reward / length / wall time and inject
`info["episode"] = {"r": ep_rew, "l": ep_len, "t": time}` on the terminal step.

```python
def step(self, action):
    if self.needs_reset:
        raise RuntimeError("Tried to step environment that needs reset")
    observation, reward, terminated, truncated, info = self.env.step(action)
    self.rewards.append(float(reward))
    if terminated or truncated:
        self.needs_reset = True
        ep_info = {"r": sum(self.rewards), "l": len(self.rewards), "t": time.time() - self.t_start}
        info["episode"] = ep_info
    self.total_steps += 1
    return observation, reward, terminated, truncated, info
```

And `Monitor.reset` resets `self.rewards = []` and `self.needs_reset = False`.

Implications:

1. **Monitor enforces "step after done → must reset first"** via
   `self.needs_reset`. DummyVecEnv satisfies this by auto-resetting on done.
   If we ever step our env outside SB3's vec wrapper and forget the reset,
   Monitor will raise `RuntimeError("Tried to step environment that needs reset")`.

2. **Monitor's `allow_early_resets=True` by default** — calling reset before
   done is fine.

3. **The episode stats SB3 logs** (the `ep_rew_mean`, `ep_len_mean` in the
   `rollout/` section of the verbose table) come from `info["episode"]` set
   here by Monitor, NOT from our env's `[EpisodeSummary]` lines. They will
   match in steady state, but Monitor's count is the canonical one for SB3.

---

## SB3's `env_checker.py` — what it validates

`check_env(env)` (line 467, ~80 lines of asserts) is the explicit validator
users can run via `from stable_baselines3.common.env_checker import check_env`.
Worth running once on `CartPoleEnv` as a smoke test.

`_check_returned_values` (line 331) asserts:

| Check | Severity | Our env |
|-------|----------|---------|
| `reset()` returns a tuple | assert | ✓ |
| `reset()` returns a 2-tuple | assert | ✓ |
| `info` from reset is a dict | assert | ✓ (we return `{}`) |
| `step()` returns 5 values | assert | ✓ |
| `reward` is `float` or `int` | assert | ✓ (`1.0`) |
| `terminated` is `bool` (strict) | assert | ✓ Python `bool` from `is_terminated()` |
| `truncated` is `bool` (strict) | assert | ✓ |
| `info` from step is a dict | assert | ✓ |
| Discrete starts at 0 | warning | ✓ (`Discrete(2)`) |
| Obs shape/dtype matches space | warning | ✓ (`np.float32` matches `Box(dtype=np.float32)`) |

**Verdict:** all SB3 env_checker assertions should pass on our `CartPoleEnv`.

---

## Cross-reference: where things can still go wrong

After the bug fixes earlier this session (stale sensor cache + WorldController
retry), our env satisfies all of SB3's static checks. Where could "SB3 still
has problems" come from?

1. **Speed.** PPO's `n_steps=256` in our config × ~10 ms per env.step (Gazebo
   service round-trip + physics step) = ~2.5 seconds per rollout, plus
   `n_epochs=10` × `batch_size=64` SGD passes = another ~1 second. Each PPO
   iteration is ~3 s wall-clock. 200k timesteps ≈ 200,000/256 = 781 iterations
   ≈ 40 minutes. **This is the expected ballpark — not a bug, but worth
   knowing if you thought it was hung.**

2. **Auto-reset hides errors.** When our `env.step` raises an exception inside
   `DummyVecEnv.step_wait`, the loop stops — but if our step *silently* returns
   garbage (e.g. NaN observations from a missed callback), SB3 will train on
   garbage with no error. **Mitigation:** SB3 has `VecCheckNan` wrapper —
   wrap the env once for debugging if you suspect NaN obs:
   ```python
   from stable_baselines3.common.vec_env import VecCheckNan
   env = VecCheckNan(env, raise_exception=True)
   ```

3. **Monitor.needs_reset → silent re-step.** If our flow somehow steps without
   resetting first (e.g. via a custom callback that calls `env.step` directly),
   Monitor raises `RuntimeError("Tried to step environment that needs reset")`.
   You'd see a traceback, not silent breakage.

4. **`_last_obs is not None` cache poisoning.** If you re-train the same
   `model` after manual env interaction, `_last_obs` may be stale. Solution:
   pass `reset_num_timesteps=True` to `model.learn()` (the default).

5. **Vec-env env_method routing.** If a wrapper-aware callback tries to call
   `env.get_attr("max_steps_per_episode")`, that goes through
   `DummyVecEnv.get_attr` which uses `get_wrapper_attr`. Our env has the
   attribute, so this works — but for any future attribute we add, make sure
   it's actually on the env instance (not just a closure).

6. **`reset` keyword-only.** `DummyVecEnv.reset` calls
   `env.reset(seed=self._seeds[env_idx], **maybe_options)` — always with
   keyword. So our positional-allowed signature works today, but the strict-API
   keyword-only fix from the Gymnasium doc would shake out any other code that
   tries to pass them positionally.

7. **VecEnv.observation_space.dtype mismatch with policy.** SB3's `MlpPolicy`
   builds its first layer from `observation_space.shape`. If `dtype` were
   `np.float64` (it isn't, we use float32), the policy would still build but
   tensor casting would slow things down. Worth keeping `float32` going
   forward.

---

## Recommended next debugging steps if SB3 still misbehaves

1. **Run SB3's own `check_env` against our env once.**
   ```python
   from stable_baselines3.common.env_checker import check_env
   from cartpole_env import CartPoleEnv
   check_env(CartPoleEnv(world_name="cartpole"))
   ```
   This won't catch the runtime async-callback bugs we've been hitting, but it
   verifies the static API contract. We have our own version of this in
   `test/test_env_contract.py`, but SB3's check is more thorough.

2. **Wrap with `VecCheckNan` to surface NaN/inf observations.** Cheap to add,
   loud failure mode.

3. **Compare our `[EpisodeSummary]` count with SB3's `ep_len_mean`** in the
   rollout table. If they diverge, the auto-reset path is doing something
   different than expected.

4. **If SB3's iterations/sec is suspiciously low**, look at
   `WorldController.step` — it takes ~10 ms per call from the worker thread.
   Bumping `steps_per_action` in CartPoleEnv from 1 to e.g. 5 means each
   `env.step` is one transport round-trip but 5 physics ticks. Faster
   throughput, coarser control resolution. Common trade.

---

## Source reference (commit `10dda86`)

- `stable_baselines3/common/base_class.py` — `_wrap_env`, `_setup_learn`,
  `learn` entry point
- `stable_baselines3/common/on_policy_algorithm.py` — `collect_rollouts`
  (lines 162–290)
- `stable_baselines3/common/vec_env/dummy_vec_env.py` — `DummyVecEnv.step_wait`
  with auto-reset
- `stable_baselines3/common/monitor.py` — `Monitor.step`, `Monitor.reset`
- `stable_baselines3/common/env_checker.py` — `check_env` and
  `_check_returned_values`
- `stable_baselines3/ppo/ppo.py` — PPO-specific logic (gae_lambda computation,
  policy update). Doesn't touch the env.
