---
name: stable-baselines-context
description: Provide additional context and information on the StableBaselines3 Library
---
## What is StableBaselines3
StableBaselines3 (SB3) is a repository which conviently implements popular reinforcement learning algorithms as an overlay to Farama's Gymnaisum.

Since it serves as a wrapper environments are often set up in a strict manner, even a simple run seen in [sb3_simple.py](sb3_simple.py)

## Tips and Tricks when creating a custom environment

If you want to learn about how to create a custom environment, we recommend you read this page. We also provide a colab notebook for a concrete example of creating a custom gym environment.

Some basic advice:

- always normalize your observation space when you can, i.e., when you know the boundaries

- normalize your action space and make it symmetric when continuous (cf potential issue below) A good practice is to rescale your actions to lie in [-1, 1]. This does not limit you as you can easily rescale the action inside the environment

- start with shaped reward (i.e. informative reward) and simplified version of your problem

- debug with random actions to check that your environment works and follows the gym interface:

Two important things to keep in mind when creating a custom environment is to avoid breaking Markov assumption and properly handle termination due to a timeout (maximum number of steps in an episode). For instance, if there is some time delay between action and observation (e.g. due to wifi communication), you should give a history of observations as input.

Termination due to timeout (max number of steps per episode) needs to be handled separately. You should fill the key in the info dict: info["TimeLimit.truncated"] = True. If you are using the gym TimeLimit wrapper, this will be done automatically. You can read Time Limit in RL or take a look at the RL Tips and Tricks video for more details.

If you want to do quickly sample a random agent in a custom environment you can run the [quick_check.py](quick_check.py) script

## Using Custom Environments
To use the RL baselines with custom environments, they just need to follow the gymnasium interface. That is to say, your environment must implement the following methods (and inherits from Gymnasium Class):
The overall class must inherhit the `gymnasium.Env` class as the base class, take a took a [custom_env.py](custom_env.py) shows a barebones implementation of the class.

## Algorithms Available:
- A2C : Advantage Actor Critic - [SB3 HTML Page](https://stable-baselines3.readthedocs.io/en/v2.1.0/modules/a2c.html)
- DDPG : Deep Deterministic Policy Gradient (DDPG) - [SB3 HTML Page](https://stable-baselines3.readthedocs.io/en/v2.1.0/modules/ddpg.html)
- DQN : Deep Q Network - [SB3 HTML Page](https://stable-baselines3.readthedocs.io/en/v2.1.0/modules/dqn.html)
- HER : Hindsight Experience Replay - [SB3 HTML Page](https://stable-baselines3.readthedocs.io/en/v2.1.0/modules/her.html)
- PPO : Proximal Policy Optimization - [SB3 HTML Page](https://stable-baselines3.readthedocs.io/en/v2.1.0/modules/ppo.html)
- SAC : Soft Actor Critic - [SB3 HTML Page](https://stable-baselines3.readthedocs.io/en/v2.1.0/modules/sac.html)
- TD3 : Twin Delayed DDPG - [SB3 HTML Page](https://stable-baselines3.readthedocs.io/en/v2.1.0/modules/td3.html)

