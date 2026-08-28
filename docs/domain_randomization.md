# Domain Randomization

**Domain Randomization (DR)** varies the physical and environmental
properties of an agent and its world across simulated training instances,
instead of training against one fixed, maximally faithful configuration.
The technique injects artificial noise into the simulation, covering
dynamics, actuation, and sensing, so the policy learns to handle a range of
conditions rather than overfitting to the exact instance it observed. The
underlying bet is straightforward: a policy that stays robust across a wide
simulated distribution transfers to the real robot better than one trained
against a single high fidelity model. The real robot's true parameters stay
unknown, and DR removes the need to guess them correctly, since the trained
range only needs to contain them.

This document covers the general framework support for DR: which
mechanisms exist as `AgentSpec` fields, how they're classified, and what
each one costs to build. It intentionally stays framework-level.
`line_follower` is the only vision-based, mobile-base specification in
the framework today, and it uses DR to improve robustness against
physical disruptions such as varying lighting, motor jitter, and battery
discharge, so it has the most real mileage on these mechanisms. Every
concrete value, range, tuning result, and candidate mechanism considered
for it lives in
[`examples/line_follower.md`'s Domain Randomization section](examples/line_follower.md#domain-randomization)
instead of here, so this document doesn't drift out of sync with one
example's fast-moving tuning work. The `AgentSpec` field reference lives
in
[`creating_your_own_agent.md`](creating_your_own_agent.md#domain-randomization-for-sim-to-real).

## Mechanism Classes

The framework's world-construction model splits every DR mechanism into one
of two classes, and the class a candidate falls into determines how
expensive it becomes to build.

- **Population-based, fixed per agent.** Physics and geometry parameters,
  including mass, friction, and model geometry, are baked into each agent's
  Simulation Description Format (SDF) file at world-build time. Each of the
  N agents in one world draws its own fixed value from the distribution and
  keeps it for that agent's whole lifetime, so the population itself
  becomes the randomization, rather than any single agent's experience
  changing over time. Changing one of these mid-training would need a
  respawn path, which does not exist yet, a limitation
  `creating_your_own_agent.md` already notes. `mass_randomization` is the
  existing example.
- **Software-level, resampled per step or episode.** Action and observation
  transforms applied in Python at read or write time need no SDF change and
  no respawn, so they resample cheaply at every step or every reset.
  `action_gain_randomization` and `visual_randomization` are the existing
  examples.

New mechanisms in the population-based class carry a bigger lift, since
they touch world construction. New mechanisms in the software-level class
usually arrive as a same-shaped addition next to the existing two.

## Current Support

| AgentSpec Field | Class | Backend | Effect |
|---|---|---|---|
| `mass_randomization` | Population | Both | Scales mass and inertia by `U(1-x, 1+x)` |
| `action_gain_randomization` | Software | In-process | Scales the actuator command by `U(1-x, 1+x)`, fixed per agent |
| `visual_randomization` | Software | In-process | Applies brightness, noise, and JPEG jitter at read time |
| `track_color_randomization` | Software | In-process | Lightens dark line pixels toward a random target color |
| `action_noise_randomization` | Software | In-process | Adds per-step Gaussian jitter to the raw action |
| `battery_discharge_randomization` | Software | In-process | Decays actuator authority across an episode |

Every field except `mass_randomization` currently has exactly one real
user, `line_follower`, so its shipped values, ranges, and full tuning
history live in
[`examples/line_follower.md`'s Domain Randomization section](examples/line_follower.md#domain-randomization)
rather than duplicated here.

```python
from dataclasses import replace
from gazebo_gymnasium_bridge.envs.agent_spec import get_spec

# Strength ranges from 0.0 (off) up to roughly 0.3-0.8, depending on the field.
spec = replace(get_spec("line_follower"),
               track_color_randomization=0.2,
               action_noise_randomization=0.1,
               battery_discharge_randomization=0.3)
```

Passing this specification into `make_inprocess` builds a world where every
agent draws its own strength from each distribution, the same
population-based pattern `mass_randomization` already uses for physics
parameters.

The harness backend, which runs a launched Robot Operating System 2 (ROS 2)
simulation, implements none of the software-level DR mechanisms above.
Only `mass_randomization` runs on both backends, since both build the world
the same way.

## Glossary

| Term | Definition |
|---|---|
| **Domain Randomization (DR)** | Varying physical or environmental parameters across simulated training runs so a policy generalizes past one fixed configuration |
| **Population-Based Mechanism** | A DR mechanism sampled once per agent at world-build time and held fixed for that agent's lifetime |
| **Software-Level Mechanism** | A DR mechanism applied as an action or observation transform in Python, resampled cheaply at any point during training |
| **AgentSpec** | The dataclass describing one agent's model, observation, action, reward, and DR configuration |
| **Robot Operating System 2 (ROS 2)** | The middleware the harness backend uses to launch and communicate with a standalone simulation |
