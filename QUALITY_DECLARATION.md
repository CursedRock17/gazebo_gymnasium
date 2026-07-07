# Quality Declaration

This package declares Quality Level **4** under
[REP-2004](https://ros.org/reps/rep-2004.html).

Level 4 covers "Demos, Tutorials, and Experiments." It's the right
self-classification for `gazebo_gymnasium`: the package's primary value
is as a reference implementation for running OpenAI Gymnasium-style RL
training against Gazebo Harmonic, not as a production-critical library.

Below is the REP-2004 checklist for Level 4 plus notes on areas where we
exceed the minimums (and so could move toward Level 3 if we wanted to
take on the formal change-control overhead).

## Level 4 — Required Criteria

| Criterion | REP-2004 requires | Our status |
|-----------|--------------------|-----------|
| **License** | LICENSE file present | ✅ `LICENSE` — Apache-2.0 |
| **Copyright** | Copyright statements in source | ✅ Apache-2.0 header in every `.py` file, enforced by `test_copyright` (ament_copyright) |
| **Versioning** | Not required | ✅ Semver in every `package.xml` (`0.1.0`) |
| **Change control** | Not required | ⚠️ GitHub PRs encouraged but not strictly enforced |
| **Documentation** | Not required | ✅ Per-env tutorial in `docs/examples/`; Sphinx skeleton in `docs/sphinx/`; `CONTRIBUTING.md` |
| **Testing** | Not required | ✅ 178 pytest tests — spec/VecEnv contract, library integration, in-sim harness (ECM core + plugin), stress (scale/endurance/robustness), SDF validity + gz-check, and flake8/pep257/copyright lint |
| **Platform Support** | Tier 1 platforms | ✅ Tested on Ubuntu Noble 24.04 + ROS 2 Jazzy + gz Harmonic |
| **Security** | Not required | ✅ `SECURITY.md` with disclosure contact |

## Areas Where We Exceed Level 4

These would be the **minimum** under higher tiers; we've already done
the work, just not the change-control / Tier-1-CI requirements that
gate Level 3 and above.

- **Test coverage**: 178 tests across the spec layer, the SB3 VecEnv contract,
  library-integration smoke tests (PPO/A2C on the cartpole spec), the in-sim
  harness (ECM core + plugin round-trip), a stress suite (16–64 agents,
  thousands of steps, malformed-input / timeout robustness), SDF validity
  (static XML + `gz sdf --check`), and lint conformance. Level 1-2 want ≥90%
  line coverage with enforcement — not measured, but the bridge layer is
  heavily exercised.
- **Lint & static analysis**: `ament_flake8`, `ament_pep257`, and
  `ament_copyright` run as pytest tests and pass; `pixi run lint` runs the
  three. CI workflow pending (tracked in ROADMAP.md).
- **Public API documentation**: The reference walkthrough
  (`docs/examples/cartpole.md`) plus the "create your own agent" guide
  (`docs/creating_your_own_agent.md`). Sphinx-renderable autodoc skeleton in
  `docs/sphinx/`.
- **Architecture docs**: `docs/pytorch_jit_analysis.md`,
  `docs/composable_nodes_analysis.md`, `docs/ros2_reps_compliance.md`. Each
  captures a design decision + rationale for future readers.

## Path to Level 3

Level 3 ("development / introspection tools" — same tier as `rviz`,
`rqt`) would require:

- [ ] **Change control**: enforce PR-based workflow with at least CI
      gating (already mostly true; need to formalize a
      `.github/workflows/colcon-test.yml`).
- [ ] **Documented Tier 1 platform CI**: nightly build on Ubuntu Noble
      + ROS 2 Jazzy at minimum.
- [ ] **Quality declaration linked from README** (this file once it's
      in the repo root).

We've staged most of the work; the gating items are CI infrastructure
(GitHub Actions workflow) and a stable public API declaration. Both are
tracked in ROADMAP.md.

## Path to Level 2 / 1

Not currently a project goal. Levels 1–2 are appropriate for
production-critical libraries (rclcpp, tf2, urdf) that other packages
depend on for safety-critical behavior. `gazebo_gymnasium` is upstream
of nothing other than our own example training scripts; investing in
the formal peer-review / vulnerability-SLA / coverage-enforcement
machinery would be wasted effort at this point in the project's
lifecycle. If the package ever ships a stable API that external robots
in the field rely on, we'd revisit.

## How to Verify

```bash
# Full suite (functional + stress + lint), one command:
pixi run test

# Just the three ament linters (flake8 / pep257 / copyright):
pixi run lint

# Build the workspace:
pixi run build
```

All three of those should pass clean. If they don't, the package has
slipped below its declared Level 4 and the declaration should be
adjusted before publishing the next release.
