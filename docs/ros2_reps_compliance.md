# ROS 2 REPs Compliance

This document maps the official ROS 2 REPs (REP-2000 through REP-2009,
plus REP-2014) to what `gazebo_gymnasium` actually does. Each section
states the policy from the REP, then how we comply or where we
deliberately diverge.

REPs are at <https://ros.org/reps/>. The TL;DR per REP:

| REP | Topic | Our status |
|-----|-------|-----------|
| 2000 | Target Platforms | ✅ Compliant (Jazzy, Ubuntu Noble Tier 1) |
| 2001 | Variants | N/A: research package, not a distro variant |
| 2002 | Rolling Release | ⚠️ Partial: version numbering aligned, no rolling sync |
| 2003 | Sensor + Map QoS | ⚠️ Mostly N/A: RL data flows over gz-transport, not ROS topics |
| 2004 | Package Quality | ✅ Declared at **Level 4** (see `QUALITY_DECLARATION.md`) |
| 2005 | Common Packages | N/A: not seeking inclusion |
| 2006 | Vulnerability Disclosure | ✅ Policy at `SECURITY.md` |
| 2007 | Type Adapters | N/A: no custom adapters |
| 2008 | Hardware Acceleration | ⚠️ Partial: `GAZEBO_GYM_DEVICE` flag for CPU/CUDA selection |
| 2009 | Type Negotiation | N/A: fixed message types only |
| 2014 | Benchmarking | ⚠️ Partial: throughput benchmark + per-step metrics; LTTng tracing not wired |

## REP-2000: Releases & Target Platforms

**Policy:** Each ROS 2 distribution targets one Ubuntu LTS as Tier 1.
LTS releases supported 5 years; non-LTS 1.5 years. Distro-specific
language version mandates (C++17 / Python 3.8+ for recent releases).

**Our status: ✅ Compliant.**
- Target distribution: **ROS 2 Jazzy Jalisco** (LTS, supports 2024-05 → 2029-05).
- Target platform: **Ubuntu Noble 24.04** (Tier 1).
- Python version: 3.12 (exceeds 3.8+ minimum). A single conda Python (via Pixi + RoboStack) serves ROS, Gazebo, and the RL libraries; no dual-interpreter split.
- Gazebo version: Harmonic (paired with Jazzy per <https://gazebosim.org/docs/harmonic/ros_installation>).
- C++ standard: C++17 (set in `gazebo_gymnasium_msgs/CMakeLists.txt`).

Documented in `CONTRIBUTING.md`.

## REP-2001: Variants

**Policy:** ROS 2 distributions ship variants: `ros_core`, `ros_base`,
`desktop`, `desktop_full`, etc. Maintainers should keep lower-tier
variants free of GUI dependencies.

**Our status: N/A.** `gazebo_gymnasium` is a research / examples
package, not a candidate for inclusion in any variant. We depend on
`ros_gz_*` and `foxglove_bridge` (visualization-tier deps), so if we
were ever to register a variant placement it would be at the
`desktop_full` / `simulation` tier, not `ros_core` or `ros_base`.

## REP-2002: Rolling Release

**Policy:** Rolling distro is unsupported. Maintainers should follow
semver-style version numbering: patch for backports, minor/major for
features. Release repos in `ros2-gbp` for automated bloom releases.

**Our status: ⚠️ Partial.**
- ✅ Semver-style numbering: all `package.xml` files at `0.1.0`. Bumping
  to `0.2.0` for next feature batch, `0.1.1` for bugfixes only.
- ❌ Not yet released through `bloom` / `ros2-gbp`: that step happens
  when the package matures past Level 4. Local install + colcon build
  only for now.

## REP-2003: Sensor + Map QoS

**Policy:** Sensor drivers should publish with `SystemDefaultsQoS`;
subscribers use `SensorDataQoS` (BEST_EFFORT, VOLATILE). Map publishers
use RELIABLE + TRANSIENT_LOCAL.

**Our status: ⚠️ Largely not applicable, with one reference config.**
- **The RL data path does not use ROS topics.** Observations, actions, and
  camera frames move over gz-transport, in-process for the default backend
  (no IPC at all), and over `/rl/*` gz topics for the launched-simulator
  harness. REP-2003 governs ROS 2 sensor topics, so it does not bind the path
  that actually carries our sensor data.
- **Where it does apply**, we follow it: the `ros_gz_bridge` config at
  `gazebo_gymnasium_bringup/config/line_follower_bridge.yaml` sets the camera
  image topic to `ros_qos: {reliability: best_effort, durability: volatile}`,
  matching REP-2003's `SensorDataQoS` recommendation. It is kept as a **reference
  template** for users bridging sensor data into ROS 2; no shipped launch
  wires it today.
- We publish no map topics, so the map-QoS half doesn't apply.

## REP-2004: Package Quality

**Policy:** Five-level quality framework. Maintainers self-declare a
level by documenting compliance with the level's policies in a
`QUALITY_DECLARATION.md`.

**Our status: ✅ Declared at Level 4.** See `QUALITY_DECLARATION.md`
at the repo root for the full declaration with the REP-2004
sub-checklist. Summary:
- Level 4 ("demos / tutorials / experiments") is the appropriate
  starting point for a research-flavored package.
- We exceed Level 4's minimums on several axes:
  - Testing: 260+ pytest tests across 8 environments incl. stress and vision suites (Level 4 requires none).
  - CI lint: ruff (lint + format) / ament_copyright all green.
  - Public API documented in `docs/sphinx/` + per-env tutorials.
- Path to Level 3 (introspection tools): change control and nightly Tier 1
  CI are now in place; the remaining gate is the first green run on GitHub.
  Documented in the QUALITY_DECLARATION.

## REP-2005: Common Packages

**Policy:** Curated list of packages "integral to" ROS 2. Inclusion
requires Quality Level 3+, two maintainers, ROS-specific scope.

**Our status: N/A.** Not currently seeking inclusion. Listed here for
completeness.

## REP-2006: Vulnerability Disclosure

**Policy:** Maintainers should declare a security disclosure policy:
contact email, response SLA, safe-harbor language for researchers.

**Our status: ✅ Policy at `SECURITY.md`.** Contact:
`lwendlan@umd.edu`. Response SLA: best-effort within 5 business days
(slower than ROS 2 core's 2 days; we're a single maintainer and don't
want to over-promise). Safe-harbor mirrors REP-2006's language.

## REP-2007: Type Adapters

**Policy:** Standardizes the `rclcpp` mechanism for adapting custom
C++ types into ROS messages (`is_specialized` template + two
`convert_*` static functions).

**Our status: N/A.** We use stock ROS 2 message types
(`gazebo_gymnasium_msgs/EnvMetrics`, `std_msgs/Header`, etc.) and
gz-transport's native Protobuf messages. No custom type adaptation.

## REP-2008: Hardware Acceleration

**Policy:** Vendor-neutral CMake macros + workspace-level firmware
swap for GPUs / FPGAs / DPUs. Conditional compilation via
`ROS_ACCELERATION` and friends.

**Our status: ⚠️ Partial.**
- We expose **`GAZEBO_GYM_DEVICE`** env var (`cpu` default, `cuda` or
  `auto` opt-in) which routes PyTorch / SB3 onto a GPU when requested.
  See `docs/pytorch_jit_analysis.md` for the rationale.
- We don't use the `ament_acceleration` CMake macros because our policy
  network sizes (~5-130k params) don't currently benefit from GPU
  on-device kernels. Would revisit if we add CNN policies for the
  vision-from-pixels rover variant.

## REP-2009: Type Negotiation

**Policy:** Publishers/subscribers can negotiate message types
dynamically at connection time (e.g., RGB8 vs YUV420 video).

**Our status: N/A.** Our topics use fixed types
(`gz.msgs.Float_V → ros_gz_interfaces/Float32Array`,
`gz.msgs.Image → sensor_msgs/Image`, etc.). No need for runtime
negotiation.

## REP-2014: Benchmarking

**Policy:** Grey-box benchmarking via LTTng tracing. Metrics: latency,
throughput, memory, power, real-time capability. Reproducible methodology
with realistic data.

**Our status: ⚠️ Partial.**
- ✅ Reproducible throughput benchmark: `training_scripts/benchmark.py`
  reports env-steps/s and agent-steps/s per environment with a random policy
  (no learner in the loop), plus scaling curves across `n_agents`
  (`--scale 1,4,16,32`) and CSV output.
- ✅ Per-step env metrics published via
  `gazebo_gymnasium_msgs/EnvMetrics` on `/env/metrics`: `steps_per_sec`,
  `mean_step_ms`, `episode_reward`, `current_step`, `total_steps`,
  `current_episode`. Lets Foxglove plot the canonical learning curve
  (reward vs total steps) and the throughput curve (steps_per_sec).
- ❌ Not yet LTTng-instrumented. Would benefit from `ros2_tracing`
  probes on the harness plugin to measure the
  publish→physics→subscribe latency end-to-end. Tracked in ROADMAP.md.

## Summary

We've explicitly aimed at **REP-2004 Level 4** (demos/tutorials), which
sets the bar low enough that our 275-test pytest suite, lint gates, and
sphinx docs all comfortably clear it. The other REPs that apply
(REP-2000, REP-2003, REP-2006) are fully met. The remaining REPs are
either non-applicable to a research-style env package (REP-2001, 2005,
2007, 2009) or partial in ways that match where the project actually is
in its lifecycle (REP-2002, 2008, 2014).
