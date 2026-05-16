# Import our stablebaselines3 env
from stable_baselines3.common.env_checker import check_env

# Create our custom environment - **Change** to your desired environments
env = CustomEnv(arg1, ...)
# It will check your custom environment and output additional warnings if needed
check_env(env)

# Original Reset
obs, info = env.reset()

# Low Number of steps for testing
n_steps = 10
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    # Check if the simulation should be finished
    if done:
        obs, info = env.reset()
