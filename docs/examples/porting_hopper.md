# Porting Hopper, An Annotated Walkthrough

This is the second worked tutorial, after [CartPole](cartpole.md), and it
narrates the real port of MuJoCo's Hopper, including the three physics
traps encountered, how each was diagnosed from headless probes, and the
actual numbers at every stage. Reading it before porting any articulated
robot is worthwhile; the
[create-your-own-agent guide](../creating_your_own_agent.md) stays the
reference, while this document tells the story of using it.

Hopper matters because it was the first ground-contact locomotion
environment: if the pattern held here, Walker2d and HalfCheetah would
follow mechanically, and they did, with Walker2d's physics working on the
first try in about 30 minutes of work.

Six stages carry a port from MuJoCo source to a trained policy: extracting
the source of truth, modeling the root as joints, building the bare model
through three traps, writing the declarative specification, running the
headless verification loop, and finally confirming learnability.

## Extracting The Source Of Truth From MuJoCo

Before touching SDF, pulling the following facts from `hopper.xml` and the
Gymnasium documentation matters most.

- **Kinematics.** Torso connects to thigh, thigh to leg, leg to foot,
  through three actuated hinges.
- **The root.** Not a free joint, but a slide-x, slide-z, hinge-y chain,
  the key structural fact behind the first stage below.
- **Masses**, derived from capsule densities: torso 3.53, thigh 3.93, leg
  2.71, foot 5.32 kilograms.
- **Actuators.** A gear of 200 on each hinge, with control values in the
  range −1 to 1.
- **Joint defaults**, the part everyone misses: `armature="1.0"
  damping="1.0"`, plus joint ranges of −150 to 0 degrees for the thigh and
  knee and plus or minus 45 degrees for the ankle.
- **Observation**, 11 values: all positions except the forward slide,
  followed by all 6 velocities.
- **Reward.** Forward velocity, plus 1.0 for staying alive, minus 0.001
  times the summed squared actions.
- **Health.** Height above 0.7, pitch magnitude below 0.2 radians, and
  every state value below 100 in magnitude.

## Modeling The Root As Joints

Because MuJoCo's planar root really is just three joints, modeling it
identically follows directly.

```
world = anchor -[root_fwd: prismatic +Y]- d1 -[root_up: prismatic +Z]- d2
        -[root_pitch: revolute X]- torso - thigh - leg - foot
```

Once modeled this way, observation, reset, and forward velocity all flow
through the standard joint-based pattern, needing no extensions. Two
conventions matter here. Forward means positive Y, since agents get spaced
along X by the world builder, so N hoppers hop in parallel lanes and never
collide. The dummy links, `d1` and `d2`, need small but real mass, gaining
more in the next stage.

Spawning the root at height 1.25, matching MuJoCo's own initial Z, and
walking the chain down through −0.2 at the thigh joint, −0.45 at the knee,
and −0.5 at the ankle, puts the ankle at 0.10 and the foot's bottom
roughly 0.02 above the ground, so contact engages on the very first hop.

## Building The Bare Model Through Three Traps

The first version carries geometry and masses only, following the
[bare-model rules](../creating_your_own_agent.md#writing-the-model-sdf):
no plugins, no effort limits, and visuals on every collision link. The
probe loop then began, and every iteration used the same three headless
probes described further below, at a total wall time per iteration of
roughly 30 seconds.

**Trap one, torque explosion.** Random actions ended episodes in just 1.1
steps. Plus or minus 200 Newton-meters on a 0.068 kilogram-meter-squared
thigh works out to roughly 3,000 radians per second squared, and one
frame-skip later the velocity blows past the magnitude-100 health bound.
The diagnosis traced to MuJoCo's `damping="1.0"` default being missing
entirely. Adding `<dynamics><damping>` on every joint fixed it, and random
actions then survived 3.1 steps, better but still wrong.

**Trap two, the hidden armature.** MuJoCo's `armature="1.0"` adds roughly 1
kilogram-meter-squared of reflected rotor inertia per joint, about 15
times these links' physical inertia, and the real reason gear-200 torques
stay tame there. SDF has no armature tag, so emulating it meant adding the
armature value to each link's inertia about its hinge axis, plus roughly 1
kilogram to the root dummies for the translational degrees of freedom. One
subtlety bit later on Reacher: inertia must satisfy the triangle
inequality, meaning ixx plus iyy is at least izz in every permutation, or
`sdformat` rejects the link and the world silently fails to load. The
in-process environment now raises loudly at construction for exactly this
reason.

**Trap three, unbounded configurations.** The model still twitched, since
without MuJoCo's joint ranges, random torques folded the leg into
impossible poses. Position limits proved safe to add, since it is
`<effort>` limits specifically that break ECM actuation, never position
limits. With ranges in place, a constant knee torque made the leg lock
against its limit and support the body, the first visible sign the
contact physics was right.

The final probe numbers for the shipped model landed at roughly 35 steps
for passive collapse, expected since an unactuated monopod must buckle,
and roughly 4 steps for random survival, short of MuJoCo's own roughly 15
steps since the pitch band stays tight, but stable, controllable, and, as
the learnability stage shows, learnable.

## Writing The Declarative Specification

With the
[spec toolkit](../creating_your_own_agent.md#the-spec-toolkit), the entire
behavioral definition runs about 15 declarative lines.

```python
_HOPPER_ACT = ("thigh_joint", "leg_joint", "foot_joint")
_ROOT = ("root_fwd", "root_up", "root_pitch")

joint_obs=pos_then_vel_obs(("root_up", "root_pitch") + _HOPPER_ACT,
                           _ROOT + _HOPPER_ACT),      # MuJoCo's 5+6 layout
action_to_commands=proportional_forces(_HOPPER_ACT, 200.0),
reward_fn=forward_progress_reward(vel_index=5),        # v_fwd + 1 - 1e-3*sum(a^2)
terminated_fn=planar_health_termination(spawn_z=1.25, min_z=0.7,
                                        max_pitch=0.2),
reset_joint_state=uniform_reset(_ROOT + _HOPPER_ACT, 0.005),
```

`vel_index=5` matters specifically: the forward-slide position gets
excluded from the observation, matching MuJoCo convention, but its
velocity lands at `obs[5]`, and that one index is the entire
"forward progress" plumbing.

## Running The Headless Verification Loop

Three probes run after every model tweak, all real physics, with no
launch needed.

1. **Passive instability.** Zero actions, a single agent, must fall
   unhealthy within roughly 2 seconds. Probing with `n_agents=1` matters,
   since with per-agent autoreset, multiple agents almost never finish on
   the same step, so `dones.all()` would mislead.
2. **Finiteness under abuse.** Random torques, asserting `np.isfinite` on
   both observations and rewards every step.
3. **Determinism.** The same seed and the same actions across two fresh
   environments must produce bit-identical results.

These probes became
`test/test_mujoco_specs.py::test_locomotor_passive_collapse_terminates`
and its siblings; copying them fits any new port well.

## Confirming Learnability

```bash
python training_scripts/train.py --agent hopper --n_agents 16 --timesteps 400000
```

The actual probe curve, PPO plus `VecNormalize`, 16 agents, roughly 2,300
steps per second, showed steady improvement.

| Steps | Mean Episode Reward | Mean Episode Length |
|---|---|---|
| Random | ~4 | ~4 |
| 100,000 | 57 | 32 |
| 200,000 | ~120 | ~75 |
| 300,000 | 234 | 124 |
| 400,000 | **272** | **134**, still climbing |

That 68 times improvement establishes learnability, the environment's
green check. The stricter solved bar,
[defined here](README.md#solved-bars-explained), asks for 1,000-step
episodes with reward at or above 1,000, and this environment now clears
it. Research-informed hyperparameters, published rather than guessed, see
[README.md's training-curve section](README.md#what-solved-policies-look-like),
plus more training closed the remaining gap, confirming the shortfall really
was budget and tuning, not architecture, all along.

![Hopper standing balanced upright, solved policy, real spectator-camera capture](../images/hopper_solved_gui.png)

That pose is the honest result, not a lucky frame. The solved policy stays
almost perfectly stationary, for reasons the training-curve document above
explains: `alive_bonus` dominates Hopper's reward the way it does Ant's,
just not severely enough to prevent solving the stated bar.

## What Transferred

Walker2d reused this document's recipe wholesale, with two legs instead of
one; coplanar legs work fine because `self_collide` defaults to false
within a model. Its physics were right on the first probe run, with random
actions surviving 10.9 steps, PPO reaching 1013 reward across 710-step
episodes, and it has since been solved too, at 1522.9 reward across
999 of 1000 steps, the same way Hopper was. HalfCheetah followed with
per-joint gears and truncation-only episodes. The total marginal cost of
ports two and three landed around an hour combined, the real payoff of the
probe-driven loop this tutorial narrates.
