---
name: custom-gymnasium
description: Explains how the Farama Gymnasium API works, by extending the API to custom environments.
disable-model-invocation: false
allowed-tools: Bash(python3)
---

## The Context
Reinforcement Learning uses a classc "agent-environment loop" in which an agent observes the current scene, chooses an action based on what it sees, the environment responds with a new situation (through physics and other agent reactions) and a reward, repeat until episode ends.
Code is written in python which provides easy access to virtual environments and popular machine learning libraries


## Creating a Custom Environment
1) Import the gymnasium library: ```import gymnasium as gym```
2) Create a class which takes a gymnasium environment (```gym.Env```) as a base class, such that we can extend the default Gymnasium class.
3) Create the base class' ```__init__``` function which will create all necessary information for the environment, the observation space, the action space.
4) Observations need to be acquired in both ```Env.reset()``` and ```Env.step()``` (more on that later) as those are the two instances in which the scene will be altered, so we need to get our code aligned with DRY (Don't Repeat Yourself) principles. Create a ```_get_obs(self)``` function with no args (other than self) and returns a dict with our observations mimicing the observation space.
5) Create a ```_get_info(self)``` function which returns auxiliary information at the same time as ```_get_obs```, this is only for debugging and shouldn't affect the scene itself, returns any formatting of information.
6) Create a ```reset(self, seed: Optional[int] = None, optiions: Optional[dict] = None)``` function. The ```reset()``` function will start a new episode and takes two opitional parameters: seed, options. This will randomly setup parts of the environment in a manner that makes the most sense for the scene. Returns both ```observation, info``` as a tuple reflecting our two previous functions
7) Create a ```step(self, action)``` fucntion which contains the core environment logic. It takes an action, updates the environment state, and returns the results. This is where the physics, game rules, and reward logic live. Should take an action (in the format of our action space) as a parameter, along with ```self``` to satisfy as a class method. It should return a tuple with ```(observation, reward, terminated, truncated, info)```

## Common Environment Design Pitfalls
**Problem**: Incorrectly setting up the reward function (sparse rewards)
**Problem**: Incorrectly representing the state by including irrelevant information or missing crucial details
**Problem**: Incorrectly adding actions that aren't relevant or impossible to execute
**Problem**: Incorrectly handling boundaries, invalid states, or unclear boundary behavior

## Action and Observation Spaces
Every environment specifies the format of valid actions and observations with the action_space and observation_space attributes. This is helpful for knowing both the expected input and output of the environment, as all valid actions and observations should be contained within their respective spaces. In the example above, we sampled random actions via env.action_space.sample() instead of using an intelligent agent policy that maps observations to actions (which is what you’ll learn to build).

Understanding these spaces is crucial for building agents: - Action Space: What can your agent do? (discrete choices, continuous values, etc.) - Observation Space: What can your agent see? (images, numbers, structured data, etc.)

Importantly, Env.action_space and Env.observation_space are instances of Space, a high-level python class that provides key functions: Space.contains() and Space.sample(). Gymnasium supports a wide range of spaces:

## Real-World Environment Design Tips
### Start Simple, Add Complexity Gradually

1) First: Get basic movement and goal-reaching working

2) Then: Add obstacles, multiple goals, or time pressure

3) Finally: Add complex dynamics, partial observability, or multi-agent interactions

### Design for Learning

- Clear Success Criteria: Agent should know when it’s doing well

- Reasonable Difficulty: Not too easy (trivial) or too hard (impossible)

- Consistent Rules: Same action in same state should have same effect

- Informative Observations: Include everything needed for optimal decisions

### Think About Your Research Question

- Navigation: Focus on spatial reasoning and path planning

- Control: Emphasize dynamics, stability, and continuous actions

- Strategy: Include partial information, opponent modeling, or long-term planning

- Optimization: Design clear trade-offs and resource constraints

## Additional Resources
- As a template of a custom environment see [custom_envs/cartpole_2d.py](custom_envs/cartpole_2d.py)
- To see all available spaces (both actions & observations) check [spaces.py](spaces.py)
