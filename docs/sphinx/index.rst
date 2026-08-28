Gazebo Gymnasium
================

Train reinforcement-learning agents in `Gazebo <https://gazebosim.org/>`_ with
the standard `Gymnasium <https://gymnasium.farama.org/>`_ API.

You describe one agent as a single ``AgentSpec``, its model, observation,
action, reward, and termination, and the framework runs *N* copies of it in
one Gazebo world as a vectorized environment trainable with
`Stable-Baselines3 <https://stable-baselines3.readthedocs.io/>`_.

.. code-block:: bash

   pixi install && pixi run build
   pixi run train --agent hopper --n_agents 16

Guides
------

The narrative documentation lives in the repository as Markdown; this is
the recommended reading order, not just a link list. The full,
up-to-date index is
`docs/README.md <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/README.md>`_.

**Your first environment**

* `CartPole walkthrough <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/examples/cartpole.md>`_, the reference environment.

**Training and reviewing results**

* `Reviewing training data <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/reviewing_data.md>`_: TensorBoard, the Hugging Face Hub, and watching a live simulation through Foxglove or PlotJuggler.
* `Integrating Stable-Baselines3, skrl, rl_games, and rsl_rl <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/rl_libraries.md>`_, plus bringing a fully custom algorithm.

**Building your own environment**

* `Creating your own agent <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/creating_your_own_agent.md>`_, the reference guide.
* `Importing CAD models <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/importing_cad_models.md>`_: exporting a URDF and converting it to SDF.
* `Porting Hopper, annotated <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/examples/porting_hopper.md>`_, a narrated real port.
* `Domain randomization <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/domain_randomization.md>`_: concept, current support, and a feasibility survey.

**Environment reference**

* `Environment status and solved bars <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/examples/README.md>`_, with a page per environment.

**Framework internals**

* `How this compares to MuJoCo, PyBullet, Isaac Lab, Brax, and other ROS/Gazebo RL tooling <https://github.com/CursedRock17/gazebo_gymnasium/blob/main/docs/comparison.md>`_.

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
