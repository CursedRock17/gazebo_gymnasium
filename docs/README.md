# Documentation

This is the reading order, not just a link dump. Each tier assumes the
one above it; skipping ahead works too, everything below stays
individually readable, but this is the path from zero to a trained
policy and, eventually, a new environment of your own.

## Quickstart

Not a document, a command. [The root README](../README.md#installation)
covers the one-time `pixi install && pixi run build`, and
[Seeing It Run](../README.md#seeing-it-run-cartpole-in-the-gazebo-gui)
gets a trained CartPole balancing in the Gazebo GUI in two terminals,
no code written yet.

## Your First Environment

[**CartPole**](examples/cartpole.md) is the reference environment every
other doc points back to: observation and action spaces, how the spec
is built, training and evaluation commands, what a solved run looks
like. Read this before anything else here.

## Training And Reviewing Results

- [**Reviewing training data**](reviewing_data.md): TensorBoard for the
  scalar curves already logged, the Hugging Face Hub for published
  checkpoints, and Foxglove or PlotJuggler for watching a live
  simulation's raw signals over gz-transport.
- [**RL libraries**](rl_libraries.md): training with skrl, rl_games, or
  rsl_rl instead of Stable-Baselines3, real adapters and the bugs each
  one surfaced, plus what it takes to bring a fully custom algorithm
  with no library at all.

## Building Your Own Environment

- [**Creating your own agent**](creating_your_own_agent.md): the
  reference guide. One `AgentSpec` dataclass, the spec toolkit that
  removes most of the boilerplate, the three backends, and the
  checklist to follow end to end.
- [**Importing CAD models**](importing_cad_models.md): starting from an
  existing CAD model instead of writing SDF by hand, exporting a URDF
  and converting it, the real gotchas this repository's own rover model
  hit along the way.
- [**Porting Hopper, annotated**](examples/porting_hopper.md): a second
  worked tutorial, narrating a real MuJoCo port stage by stage,
  including the three physics traps and how each was diagnosed.
- [**Domain randomization**](domain_randomization.md): the population-
  based and software-level mechanism classes, what each costs to build,
  useful before adding sim-to-real robustness to a new agent.

## Environment Reference

[**Status and solved bars**](examples/README.md) tracks every
environment's verified results and what counts as solved, with a page
per environment:
[CartPole](examples/cartpole.md),
[InvertedDoublePendulum](examples/inverted_double_pendulum.md),
[Hopper](examples/hopper.md),
[Walker2d](examples/walker2d.md),
[HalfCheetah](examples/half_cheetah.md),
[Reacher](examples/reacher.md),
[Ant](examples/ant.md), and
[Line Follower](examples/line_follower.md), the one vision-based,
sim-to-real environment.

## Framework Internals

Deeper reference, useful once something needs debugging rather than
building.

- [**How this compares**](comparison.md) to MuJoCo, PyBullet, Isaac Lab,
  Brax, and other ROS/Gazebo RL tooling, with real throughput numbers.
- [**Gymnasium API reference**](gymnasium_api_reference.md) and
  [**SB3 API reference**](sb3_api_reference.md): what each library
  actually requires from an environment, useful when a contract check
  fails.
- [**Verbose logging and SB3 coexistence**](verbose_sb3_coexistence.md):
  how this project's own per-step logging interacts with SB3's verbose
  table.
- [**PyTorch JIT analysis**](pytorch_jit_analysis.md): why
  `torch.compile`/TorchScript are not worth it for this project's model
  sizes today.
- [**ROS 2 Composable Nodes analysis**](composable_nodes_analysis.md)
  and [**ROS 2 REPs compliance**](ros2_reps_compliance.md): infrastructure
  decisions and where this project stands against the ROS 2 REPs.

See also [`ROADMAP.md`](../ROADMAP.md) for where the framework is
headed and [`QUALITY_DECLARATION.md`](../QUALITY_DECLARATION.md) for
its REP-2004 quality level.
