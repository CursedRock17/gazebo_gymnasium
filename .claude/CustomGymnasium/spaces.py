# Available Spaces for Gymnasium Actions & Observations
# Each space has an example asssociated with it:

# Add each of the spaces from the gymnasium space section
from gymnasium.space import Box, Discrete, MultiBinary, MultiDiscrete, Text

# Need numpy types
import numpy as np

# Box - describes bounded space with upper and lower limits of any n-dimensional shape (like continuous control or image pixels).
identical_box = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
independent_bounded_box = Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)

# Discrete -  describes a discrete space where {0, 1, ..., n-1} are the possible values (like button presses or menu choices).
two_element_discrete = Discrete(2, seed=42)  # {0, 1}
three_element_discrete = Discrete(3, start=-1, seed=42)  # {-1, 0, 1}

# MultiBinary -  describes a binary space of any n-dimensional shape (like multiple on/off switches).
five_element_multi = MultiBinary(5, seed=42)  # array([1, 0, 1, 0, 1], dtype=int8)
three_two_element_multi = MultiBinary([3, 2], seed=42)  # array([[1, 0], [1, 0], [1, 0]], dtype=int8)

# MultiDiscrete
two_two_element_multi_discrete = MultiDiscrete(np.array([[0, 0], [3, 4]]), seed=42)
