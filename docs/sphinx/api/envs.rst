gazebo_gymnasium_bridge.envs
=============================

The environment layer. An :class:`~gazebo_gymnasium_bridge.envs.agent_spec.AgentSpec`
describes one agent; the backends below run *N* copies of it in a single Gazebo
world as a vectorized environment.

Agent specification
-------------------

The spec layer is deliberately free of any ``gz.*`` import, so it is
unit-testable without a simulator.

.. automodule:: gazebo_gymnasium_bridge.envs.agent_spec
   :members:
   :undoc-members:
   :show-inheritance:

Backends
--------

In-process (default) — hosts the simulator inside the training process.

.. automodule:: gazebo_gymnasium_bridge.envs.inprocess_vec_env
   :members:
   :undoc-members:
   :show-inheritance:

Batched harness — O(1) transport to a launched ``gz sim``.

.. automodule:: gazebo_gymnasium_bridge.envs.harness_vec_env
   :members:
   :undoc-members:
   :show-inheritance:

Per-agent backend.

.. automodule:: gazebo_gymnasium_bridge.envs.multi_agent_env
   :members:
   :undoc-members:
   :show-inheritance:

Standard Gymnasium interfaces
-----------------------------

.. automodule:: gazebo_gymnasium_bridge.envs.gym_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: gazebo_gymnasium_bridge.envs.gym_vector_env
   :members:
   :undoc-members:
   :show-inheritance:

Observation wrapping
--------------------

.. automodule:: gazebo_gymnasium_bridge.envs.obs_wrap
   :members:
   :undoc-members:
   :show-inheritance:
