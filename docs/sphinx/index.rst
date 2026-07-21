Gazebo Gymnasium
================

Train reinforcement-learning agents in `Gazebo <https://gazebosim.org/>`_ with
the standard `Gymnasium <https://gymnasium.farama.org/>`_ API.

You describe one agent as a single ``AgentSpec`` — its model, observation,
action, reward, and termination — and the framework runs *N* copies of it in
one Gazebo world as a vectorized environment trainable with
`Stable-Baselines3 <https://stable-baselines3.readthedocs.io/>`_.

.. code-block:: bash

   pixi install && pixi run build
   pixi run train --agent hopper --n_agents 16

Guides
------

The narrative documentation lives in the repository (Markdown):

* `Creating your own agent <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/creating_your_own_agent.md>`_ — the reference guide.
* `Porting Hopper, annotated <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/examples/porting_hopper.md>`_ — a narrated real port.
* `CartPole walkthrough <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/examples/cartpole.md>`_ — the reference environment.
* `Environment status and solved bars <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/examples/README.md>`_.

API reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API

   api/envs
   api/harness
   api/world_control

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
