# Porting Hopper — an annotated walkthrough

This is the second worked tutorial (after [cartpole](cartpole.md)), narrating
the real port of MuJoCo's Hopper — including the three physics traps we hit,
how each was diagnosed from headless probes, and the actual numbers at every
stage. Read it before porting any articulated robot; the
[create-your-own-agent guide](../creating_your_own_agent.md) is the reference,
this is the *story of using it*.

Hopper matters because it was the first ground-contact locomotion env: if the
pattern held here, walker2d and half_cheetah would be mechanical. (They were —
walker2d's physics worked on the first try, ~30 minutes of work.)

## Step 0 — Extract the source of truth from MuJoCo

Before touching SDF, pull from `hopper.xml` and the Gymnasium docs:

- **Kinematics**: torso→thigh→leg→foot, three actuated hinges.
- **The root**: *not* a free joint — a slide-x, slide-z, hinge-y chain. This is
  the key structural fact (Step 1).
- **Masses** (from capsule densities): torso 3.53, thigh 3.93, leg 2.71,
  foot 5.32 kg.
- **Actuators**: gear 200 on each hinge, ctrl ∈ [-1, 1].
- **Joint defaults** — the part everyone misses: `armature="1.0"
  damping="1.0"` and joint ranges (thigh/knee −150..0°, ankle ±45°).
- **Obs (11)**: all positions *except* forward-slide, then all 6 velocities.
- **Reward**: forward velocity + 1.0 alive − 1e-3·Σa².
- **Health**: z > 0.7, |pitch| < 0.2, |state| < 100.

## Step 1 — The root is just joints

Because MuJoCo's planar root is literally three joints, model it identically:

```
world ═ anchor ─[root_fwd: prismatic +Y]─ d1 ─[root_up: prismatic +Z]─ d2
        ─[root_pitch: revolute X]─ torso ─ thigh ─ leg ─ foot
```

Everything — observation, reset, forward velocity — now flows through the
standard joint-based pattern. No extensions needed. Two conventions:

- **Forward = +Y.** Agents are spaced along X by the world builder, so N
  hoppers hop in parallel lanes and never collide.
- The dummy links (`d1`, `d2`) need small-but-real mass (they'll get more in
  Step 2).

Spawn height: root at 1.25 (MuJoCo's initial z); walking the chain down
(−0.2 thigh joint, −0.45 knee, −0.5 ankle) puts the ankle at 0.10 and the
foot's bottom ~0.02 above ground — contact engages on the first hop.

## Step 2 — The bare model, and the three traps

First version: geometry + masses only, per the
[bare-model rules](../creating_your_own_agent.md#step-1--write-the-model-sdf)
(no plugins, **no effort limits**, visuals on every collision link). Then the
probe loop began. Every iteration used the same three headless probes
(Step 4) — total wall time per iteration: ~30 seconds.

**Trap 1 — torque explosion.** Random actions ended episodes in **1.1 steps**:
±200 N·m on a 0.068 kg·m² thigh is ~3000 rad/s²; one frame-skip later the
velocity blows past the |obs| < 100 health bound. Diagnosis: MuJoCo's
`damping="1.0"` default was missing. Fix: `<dynamics><damping>` on every
joint. Result: random survives 3.1 steps. Better, still wrong.

**Trap 2 — the hidden armature.** MuJoCo's `armature="1.0"` adds ~1 kg·m² of
reflected rotor inertia *per joint* — ~15× these links' physical inertia, and
the real reason gear-200 torques are tame there. SDF has no armature tag.
Emulation: add the armature value to each link's inertia **about its hinge
axis** (and ~1 kg to the root dummies for the translational dofs). One
subtlety that bit us later on reacher: inertia must satisfy the triangle
inequality (ixx + iyy ≥ izz in every permutation) or sdformat rejects the
link and **the world silently fails to load** — the in-process env now raises
loudly at construction for exactly this reason.

**Trap 3 — unbounded configurations.** Still twitchy: without MuJoCo's joint
*ranges*, random torques fold the leg into impossible poses. Position limits
are safe to add (it's `<effort>` limits that break ECM actuation — never those).
With ranges in place, a constant knee torque made the leg lock against its
limit and *support the body* — the first visible sign the contact physics was
right.

Final probe numbers for the shipped model: passive collapse terminates at
~35 steps (an unactuated monopod must buckle); random survives ~4 steps.
Short by MuJoCo standards (~15) — the pitch band is tight — but stable,
controllable, and as Step 5 shows, learnable.

## Step 3 — The spec is ~15 declarative lines

With the [spec toolkit](../creating_your_own_agent.md#the-spec-toolkit--dont-hand-write-the-plumbing),
the entire behavioral definition:

```python
_HOPPER_ACT = ("thigh_joint", "leg_joint", "foot_joint")
_ROOT = ("root_fwd", "root_up", "root_pitch")

joint_obs=pos_then_vel_obs(("root_up", "root_pitch") + _HOPPER_ACT,
                           _ROOT + _HOPPER_ACT),      # MuJoCo's 5+6 layout
action_to_commands=proportional_forces(_HOPPER_ACT, 200.0),
reward_fn=forward_progress_reward(vel_index=5),        # v_fwd + 1 - 1e-3·Σa²
terminated_fn=planar_health_termination(spawn_z=1.25, min_z=0.7,
                                        max_pitch=0.2),
reset_joint_state=uniform_reset(_ROOT + _HOPPER_ACT, 0.005),
```

Note `vel_index=5`: the forward-slide *position* is excluded from obs (MuJoCo
convention) but its *velocity* is obs[5] — that one index is the entire
"forward progress" plumbing.

## Step 4 — The headless verification loop

Three probes, run after every model tweak (all real physics, no launch):

1. **Passive instability** — zero actions, single agent: must fall unhealthy
   within ~2 s. *Probe with `n_agents=1`*: with per-agent autoreset, multiple
   agents almost never finish on the same step, so `dones.all()` misleads.
2. **Finiteness under abuse** — random torques, assert `np.isfinite` on obs
   and rewards every step.
3. **Determinism** — same seed + same actions across two fresh envs must be
   bit-identical.

These became `test/test_mujoco_specs.py::test_locomotor_passive_collapse_terminates`
etc. — copy them for your port.

## Step 5 — Learnability, and what "passing" means

```bash
python training_scripts/train.py --agent hopper --n_agents 16 --timesteps 400000
```

The actual probe curve (PPO + VecNormalize, 16 agents, ~2,300 steps/s):

| steps | mean ep reward | mean ep length |
|---|---|---|
| random | ~4 | ~4 |
| 100k | 57 | 32 |
| 200k | ~120 | ~75 |
| 300k | 234 | 124 |
| 400k | **272** | **134** — still climbing |

That 68× improvement establishes *learnability* — the env's green check. The
stricter **solved bar** ([defined here](README.md#solved-bars--what-passing-means-per-environment))
is 1000-step episodes with reward ≥ 1000; the remaining gap is training
budget/tuning, not architecture (the cartpole sweep demonstrated that closure).

## What transferred

Walker2d reused this file's recipe wholesale (two legs instead of one; coplanar
legs work because `self_collide` defaults false within a model) and its physics
were right on the **first** probe run — random 10.9 steps, PPO to 1013 reward /
710-step episodes. Half_cheetah followed with per-joint gears and
truncation-only episodes. Total marginal cost of ports two and three: about an
hour combined. That is the payoff of the probe-driven loop this tutorial
narrates.
