# Per-step training metrics emitted by the env class for visualization
# (e.g. Foxglove plots) and diagnostics. Replaces the older Float_V-as-
# array layout — every value is a named field here.

# Time-stamped on the env-loop process clock.
std_msgs/Header header

# Env-loop throughput over a 1 s sliding window (env.step() calls/sec).
float32 steps_per_sec

# Mean wall-clock duration of env.step() over the same window, in ms.
float32 mean_step_ms

# Cumulative reward of the CURRENT episode. Resets to 0 on env.reset().
float32 episode_reward

# Step index within the CURRENT episode (1-based). Resets on reset().
uint32 current_step

# Total env.step() calls across ALL episodes since the trainer started.
# Plot reward vs this to get the canonical learning curve.
uint32 total_steps

# Current episode index (1-based). Increments on reset().
uint32 current_episode
