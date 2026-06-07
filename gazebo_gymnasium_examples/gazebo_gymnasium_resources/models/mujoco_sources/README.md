# MuJoCo source models — staging area

These are the upstream MJCF (MuJoCo XML) files from
[Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
at `gymnasium/envs/mujoco/assets/`. Cloned 2026-05-25 from the `main` branch.

Licensed under MIT (see `LICENSE_GYMNASIUM`). Copyright OpenAI 2016 + Farama
Foundation 2022.

## Status

**These files are NOT directly loadable by Gazebo Harmonic.** They are MJCF —
MuJoCo's own format — not URDF or SDF. They live here so the conversion to SDF
can happen in this repo without re-cloning Gymnasium.

| File | Env | Lines | Likely conversion difficulty |
|------|-----|-------|------------------------------|
| `inverted_pendulum.xml` | InvertedPendulum-v5 | 27 | **Easy** (1 link, 1 joint — similar to our CartPole) |
| `inverted_double_pendulum.xml` | InvertedDoublePendulum-v5 | 47 | Easy (2 links) |
| `reacher.xml` | Reacher-v5 | 39 | Easy (2 links, planar) |
| `swimmer.xml` | Swimmer-v5 | 39 | Medium (planar, fluid-like joints) |
| `hopper.xml` | Hopper-v5 | 53 | Medium (1 leg, balance required) |
| `walker2d.xml` / `walker2d_v5.xml` | Walker2d-v5 | 68 | Medium (2 legs, planar) |
| `ant.xml` | Ant-v5 | 81 | Medium-hard (4 legs, 8 joints) |
| `half_cheetah.xml` | HalfCheetah-v5 | 96 | Medium (planar, 6 joints) |
| `pusher.xml` / `pusher_v5.xml` | Pusher-v5 | 91/97 | Medium-hard (arm + object dynamics) |
| `humanoid.xml` | Humanoid-v5 | 121 | **Hard** (17 joints, contact-rich) |
| `humanoidstandup.xml` | HumanoidStandup-v5 | 121 | Hard (same model, different reward) |
| `point.xml` | (point mass for tests) | 31 | Trivial but not a benchmark env |

## What conversion involves

For each env you actually want to port, the steps are:

1. **MJCF → URDF.** Two paths:
   - Quick & dirty: write a small Python script using `mujoco-py` / `mujoco`
     to walk the model and emit URDF. Sufficient for simple models.
   - More robust: existing tools like
     [`mjcf2urdf`](https://github.com/Yasunori-Hirakawa/mjcf2urdf) or
     [`obj2mjcf`](https://github.com/kevinzakka/obj2mjcf) workflows. None of
     them handle everything (tendons, equality constraints, sites).

2. **URDF → SDF.** One liner:
   ```bash
   gz sdf -p model.urdf > model.sdf
   ```
   Inertias and visuals usually carry over fine. Collisions sometimes need
   manual tuning for Gazebo's physics engines (ODE / DART / Bullet) which use
   different solvers than MuJoCo.

3. **Add actuators.** MJCF uses `<motor>` tags; SDF uses plugin instances like
   `<plugin filename="gz-sim-joint-controller-system">`. One per actuated DOF.
   See `models/cartpole/cartpole.sdf` for the pattern.

4. **Tune for Gazebo's solver.** Things that often need manual fixup:
   - Joint damping / friction values (MuJoCo and Gazebo interpret these
     differently)
   - Contact stiffness / friction coefficients
   - Mass scaling (some MuJoCo models use unusually low masses)
   - Step size — MuJoCo defaults to 2 ms; Gazebo's `max_step_size` is usually
     10 ms. Sub-step the controller if you want apples-to-apples dynamics.

5. **Pick the right physics engine.** For contact-rich locomotion (Humanoid,
   Ant), Bullet or DART tends to be more stable than ODE in Gazebo.

## Suggested porting order

If you actually want to do this work, my recommended sequence:

1. **InvertedPendulum** — almost identical to our CartPole; mostly a copy of
   the existing plugin with the action mapping switched from velocity to torque.
   Validates that the pattern works for a second env.
2. **Reacher** — 2D arm, planar, no balance. Tests the multi-joint observation
   path.
3. **Pendulum-v1** (note: this is *classic control*, not MuJoCo — pure analytic
   physics). Worth doing in pure Python with no Gazebo overhead as a sanity
   benchmark.
4. **HalfCheetah** — flagship locomotion benchmark, planar (so no falling
   sideways issues).
5. **Ant** — 4-legged, well-studied. Significant jump in joint count.
6. **Humanoid / HumanoidStandup** — flagship hard problem. Defer until the
   rest of the pipeline is bulletproof.

The plumbing on the **env / training-script / launch-file** side is already
in place (see `docs/examples/cartpole.md` for the template). The new work is
basically all model conversion + a per-env sync-gate plugin.

## Why these aren't pre-converted

A real conversion of even one env (say Ant) is a multi-hour task that requires
hands-on tuning in Gazebo to get the physics looking right. Doing all 14 blind
would produce 14 broken models. I've staged the source XML so the conversion
work isn't blocked on "find the upstream files," but the conversion itself is
left to a focused future session.
