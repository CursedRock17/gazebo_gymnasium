# Gymnasium API Reference

Extracted from `Farama-Foundation/Gymnasium` @ commit `0f6b0db` (2026-05-15, current `main`).
Source: https://github.com/Farama-Foundation/Gymnasium.git

The goal of this doc is to give us a single place to check our `GazeboEnv` /
`CartPoleEnv` against the canonical Gymnasium contract, so when SB3 (or any
other library) complains, we know whether the bug is on our side or theirs.

---

## The `gymnasium.Env` contract

From `gymnasium/core.py` lines 25-284.

```python
class Env(Generic[ObsType, ActType]):
    # Set in ALL subclasses:
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]

    # Set in SOME subclasses (defaults exist):
    metadata: dict[str, Any] = {"render_modes": []}
    render_mode: str | None = None
    spec: EnvSpec | None = None
```

### `step(action)`: must return a **5-tuple**

```python
def step(self, action: ActType) -> tuple[
    ObsType,           # observation — element of observation_space
    SupportsFloat,     # reward — float, int, np.integer, np.floating; NOT NaN/inf
    bool,              # terminated — MDP terminal state
    bool,              # truncated — outside MDP (typically time limit)
    dict[str, Any],    # info
]:
```

**Critical:** `terminated` and `truncated` are **separate booleans**. Pre-v0.26
APIs used a single `done`; that's deprecated. SB3 ≥ 2.0 expects the 5-tuple.

**Passive checker warnings to avoid** (from `gymnasium/utils/passive_env_checker.py`):

- `terminated` must be `bool` or `np.bool_` (not `int`, not `np.bool8`)
- `truncated` must be `bool` or `np.bool_`
- `reward` must be `int | float | np.integer | np.floating`: **NaN or inf logs a warning**
- `info` must be a `dict`
- `obs.dtype` must match `observation_space.dtype` (if Box): **this is where `np.float64` vs `np.float32` bites**

### `reset(*, seed=None, options=None)`: must return a **2-tuple**

```python
def reset(
    self,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[ObsType, dict[str, Any]]:
```

**Critical points:**

1. `seed` and `options` are **keyword-only** (`*` in the signature). The passive
   checker uses `inspect.signature(env.reset)` and warns if these aren't named.
2. **First line must be `super().reset(seed=seed)`**: this initializes
   `self._np_random` so `env.action_space.sample()` is reproducible.
3. Default seed must be `None` (not 0, not -1). The passive checker warns otherwise.
4. The returned `obs` must match `observation_space` just like `step()`.

### Required attributes

| Attribute | Type | Required? | Notes |
|-----------|------|-----------|-------|
| `observation_space` | `Space` | Yes | Set in `__init__` |
| `action_space` | `Space` | Yes | Set in `__init__` |
| `metadata` | `dict` | Has default `{"render_modes": []}` | Override if you implement `render` |
| `render_mode` | `str \| None` | Has default `None` | Set if render is supported |
| `spec` | `EnvSpec \| None` | Optional | `gym.make` sets it; SB3 reads it for logging |

### Other methods

- `render() -> RenderFrame | list[RenderFrame] | None`: optional; supports modes `"human"`, `"rgb_array"`, `"ansi"`. Default raises `NotImplementedError`.
- `close()`: cleanup. Idempotent. Default is a `pass`.
- `unwrapped` property: returns base env; `self` by default.

---

## Spaces: what we use

### `Box`

`gymnasium/spaces/box.py` lines 67-~700.

- Cartesian product of n closed intervals. `low.shape == high.shape == shape`.
- `dtype` defaults to `np.float32` (matters; see SB3 section).
- Bounds can be ±inf for unbounded dimensions. The space tracks `bounded_below`/`bounded_above` arrays.

Our usage in `cartpole_env.py` is correct:
```python
Box(low=np.array([-4.8, -inf, -0.42, -inf], dtype=np.float32),
    high=np.array([4.8, inf, 0.42, inf], dtype=np.float32),
    shape=(4,), dtype=np.float32)
```

### `Discrete`

`gymnasium/spaces/discrete.py` lines 27-~200.

- Set `{0, 1, ..., n-1}` (or `{start, ..., start+n-1}`).
- Default sample dtype `np.int64`. **`env.action_space.sample()` returns `np.int64`**, not Python `int`. Code that takes the action must handle this; our `apply_action` does (`int(action.item()) if isinstance(action, np.ndarray) else int(action)`).

---

## Wrappers: what `gym.make` adds automatically

When you `gym.make("CartPole-v1")`, the returned env is:

```
TimeLimit < OrderEnforcing < PassiveEnvChecker < CartPoleEnv
```

The standalone path (what we use: instantiating our env class directly and feeding
it to SB3) **skips all three**. That's both a feature and a footgun.

### `TimeLimit`: `gymnasium/wrappers/common.py:42-140`

Wraps `env.step` to also return `truncated=True` once
`elapsed_steps >= max_episode_steps`. Reset zeros the counter.

**We re-implement this inside our env** via `is_truncated()` returning
`self.current_step >= self.max_steps_per_episode`. That's fine, but note the
ordering difference from the wrapper version:

- Wrapper increments `_elapsed_steps` then checks `>= max`.
- Our env checks BEFORE incrementing `current_step` in `step()`.

Net behavior is identical for `max=500` (both halt after 500 steps). Worth keeping
in mind if you ever set `max_steps_per_episode = 1`.

### `OrderEnforcing`: same file, ~line 250

Raises `gymnasium.error.ResetNeeded` if `step()` is called before `reset()`. We
don't have this guard; SB3 calls `reset()` first internally so it never trips.

### `PassiveEnvChecker`: same file, ~line 350

Wraps `step`/`reset`/`render` with the validation functions in `passive_env_checker.py`.
It logs warnings (not errors) for type mismatches. Useful to mentally run through
when an env behaves strangely.

---

## What this means for our SB3 integration

Cross-checking `cartpole_env.py` against the contract above:

| Check | Status | Notes |
|-------|--------|-------|
| `observation_space` set | ✓ | Box(4,) float32 |
| `action_space` set | ✓ | Discrete(2) |
| `step` returns 5-tuple | ✓ | (obs, reward, terminated, truncated, info) |
| `reset` returns 2-tuple | ✓ | (obs, {}) |
| `reset` signature has kw-only `seed`, `options` | ⚠ | Our signature is `def reset(self, seed=None, options=None)`, positional-allowed. **Passive checker warns this could trigger a deprecation log**. Fix: add `*,` to make them keyword-only. |
| First line of `reset` calls `super().reset(seed=seed)` | ✓ | Via `GazeboEnv.reset → super().reset(seed=seed, options=options)` (the gym.Env one), which initializes `_np_random` |
| `obs.dtype == observation_space.dtype` | ✓ | Both `np.float32` |
| `terminated`/`truncated` are `bool` | ✓ | Python bool from `is_terminated()` / `is_truncated()` |
| `reward` is numeric, not NaN/inf | ✓ | `1.0` constant |
| `info` is a dict | ✓ | |
| `metadata` declared | ✓ | `{"render_modes": ["human"], "render_fps": 30}` from `GazeboEnv` |
| `spec` set | n/a | Not set; SB3 handles `None` gracefully but logging may show "Unknown" |
| `render` callable | ✓ | No-op in `GazeboEnv` |
| `close` callable | ✓ | No-op |

### Likely SB3-specific gotchas (read these when SB3 misbehaves)

1. **`reset(seed=)` keyword-only.** Make the signature `def reset(self, *, seed=None, options=None)`. SB3's `DummyVecEnv` calls it with keywords, so it works today, but the passive-checker warning is noise. Surgical 1-char fix in two files.
2. **`action.dtype` on SB3 side.** SB3's `MlpPolicy` for `Discrete` action spaces returns a `np.ndarray` of shape `(n_envs,)` with `dtype=np.int64`. Our `apply_action` handles `numpy.ndarray` via `.item()`. ✓
3. **VecEnv wrapping.** `PPO("MlpPolicy", env)` wraps our env in a `DummyVecEnv` automatically. `DummyVecEnv.step_wait()` calls `env.step(action)` where `action` is a numpy array, same as above. ✓
4. **Auto-reset on termination.** `DummyVecEnv.step_wait()` calls `env.reset()` automatically when `terminated or truncated`. So our env doesn't need to auto-reset (and shouldn't: that'd skip a frame).
5. **`Monitor` wrapper inside SB3.** `PPO(..., env)` may add `Monitor` automatically. `Monitor` records `episode_reward` and `episode_length` and emits them in the `info["episode"]` dict on the terminal step. We don't need to do anything; just don't fight it.

---

## Concrete next step

The reset keyword-only fix is the only place where our env diverges from the strict
Gymnasium spec. Worth applying when we touch this code next; not load-bearing.

For deeper SB3 debugging, the next research pass is SB3 itself, specifically
`stable_baselines3/common/base_class.py` and `vec_env/dummy_vec_env.py` to see how
exactly our env is being driven. That's a future roadmap item (see ROADMAP.md).

---

## Source reference (commit `0f6b0db`)

- `gymnasium/core.py`: `Env` + `Wrapper` base classes
- `gymnasium/spaces/box.py`: `Box`
- `gymnasium/spaces/discrete.py`: `Discrete`
- `gymnasium/wrappers/common.py`: `TimeLimit`, `OrderEnforcing`, `PassiveEnvChecker`, `Autoreset`, etc.
- `gymnasium/utils/passive_env_checker.py`: `env_step_passive_checker`, `env_reset_passive_checker`, `check_obs`, `_check_box_observation_space`
- `gymnasium/envs/registration.py`: `EnvSpec`, `make`, `register`
