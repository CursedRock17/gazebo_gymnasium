# Must import gymnasium as a top level repository
import gymnasium as gym

# Import our desired algorithm from stable_baselines3
from stable_baselines3 import A2C

# Create our environment *note* This doesn't have to come straight from gymnasium, any Gymnasium Custom ENV will work
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Create our desired model and learn from some given timesteps within our environment
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Vectorize the environment if possible
vec_env = model.get_env()

# Call a reset on the overall environment before we begin, collect original observations based on the observation space
obs = vec_env.reset()

# Loop for a total of 1000 episodes, each around 10,000 steps a piece
for i in range(1000):
    # Make a prediction which just gives us an action
    action, _state = model.predict(obs, deterministic=True)
    # Step through the environment and underneath the hood, increase the current step number by 1.
    obs, reward, done, info = vec_env.step(action)
    # Render with default Gymnasium environment (not necessary if you have a separate backend)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
