# gazebo_gymnasium

Core library package. Provides the `GazeboEnv` base class and `WorldController` used by all Gazebo-backed Gymnasium environments.

## Modules

**`gazebo_env.py`** — `GazeboEnv(gymnasium.Env)`

Abstract base class. Manages the simulation step/reset loop and exposes a standard Gymnasium interface. Subclass this for each robot/task.

Abstract methods to implement:

| Method | Called by | Purpose |
|---|---|---|
| `apply_action(action)` | `step()` | Send action to simulation |
| `get_observation()` | `step()` | Read current state |
| `get_reward(action)` | `step()` | Compute step reward |
| `is_terminated()` | `step()` | Episode end condition |
| `is_truncated()` | `step()` | Time-limit condition |
| `set_default_observation()` | `reset()` | Initial state on episode start |

Optional override:

| Method | Default | Purpose |
|---|---|---|
| `get_info()` | `{}` | Auxiliary info dict |

Constructor parameters:

| Parameter | Type | Description |
|---|---|---|
| `world_name` | `str` | Gazebo world name (must match world SDF `<world name="...">`) |
| `observation_space` | `gym.Space` | Observation space definition |
| `action_space` | `gym.Space` | Action space definition |
| `steps_per_action` | `int` | Physics ticks per `step()` call (default: 10) |

**`world_control.py`** — `WorldController`

Thin wrapper around the Gazebo `WorldControl` service over `gz.transport13`. Used internally by `GazeboEnv`.

| Method | Gazebo service effect |
|---|---|
| `step()` | Run exactly `steps_per_action` physics ticks, then pause |
| `reset()` | Reset all entities to initial state |
| `pause()` | Pause simulation |
| `unpause()` | Resume simulation |

## Dependencies

- `gz.transport13` (Gazebo Harmonic system package)
- `gz.msgs10` (Gazebo Harmonic system package)
- `gymnasium`
