# Contributing to gazebo_gymnasium

Thanks for considering a contribution. This repo aims to align with the
[ROS 2 Developer Guide](https://docs.ros.org/en/jazzy/Contributing/Developer-Guide.html)
and the
[ROS 2 Code Style Guide](https://docs.ros.org/en/jazzy/The-ROS2-Project/Contributing/Code-Style-Language-Versions.html);
this document only lists what's project-specific.

## Quick checks before opening a PR

```bash
# Build the workspace
pixi run build

# Auto-format every project .py file in place (ruff format, config in
# pyproject.toml; run this before the checks below, not after — it's a
# formatter, not a checker). Also applies auto-fixable lint findings
# (import order, etc.).
pixi run format

# Full test suite — functional + stress + lint
pixi run test

# Just the lint checks (ruff check + ruff format --check + copyright headers)
pixi run lint

# Static type check (ty, config in pyproject.toml's [tool.ty.rules] --
# not wired into `pixi run test` yet, run it separately)
pixi run typecheck
```

Python style/lint runs through [Ruff](https://docs.astral.sh/ruff/) (`test_ruff.py`),
configured in `pyproject.toml`'s `[tool.ruff]` to mirror
[Google's Python style guide](https://google.github.io/styleguide/pyguide.html):
99-character lines, double quotes, one imported name per `from` line,
alphabetized import groups (stdlib → third-party → first-party), and the
Google docstring convention. It replaces the former `flake8` + `ament_pep257`
+ `yapf` trio (and `scripts/lint.sh`, which wrapped the old ament linters
directly — no longer the canonical entry point; use `pixi run lint`).
[ty](https://docs.astral.sh/ty/) adds static type checking on top,
currently permissive (see `pyproject.toml`'s comments for why — this
codebase has no type annotations yet and its heaviest dependencies are
stub-incomplete) rather than blocking on every finding.

`ament_copyright` still checks every project Python file has an
Apache-2.0 header (`test_copyright.py`) — unrelated to style/lint, no
Ruff equivalent, left as-is.

If you add a new Python file, run `pixi run python
scripts/add_license_headers.py` once to prepend the standard header (the
helper is idempotent — already-headered files are skipped).

All tests must pass before review. CI runs `colcon test` on every PR.

## Code style

### Python (most of the codebase)

- **PEP 8 / PEP 257** plus [Google's Python style guide](https://google.github.io/styleguide/pyguide.html),
  both enforced by Ruff (`pyproject.toml`'s `[tool.ruff]`). `test_ruff.py`
  enforces this — don't disable it.
- **99-character line limit**, matching the ROS 2 Developer Guide default
  (`pyproject.toml`'s `[tool.ruff] line-length` is the source of truth).
- **Imports**: standard library → third-party → first-party, each group
  separated by a blank line, alphabetized within each group (Google's
  import-order style — `gz.*` and `gymnasium` are third-party).
- **Type hints**: encouraged on public APIs (env constructors, plugin
  configure methods); not required everywhere.
- **Docstrings**: every public class and module gets one. Single-line for
  utility functions is fine.

### XML (SDF, URDF, package.xml, launch)

- 2-space indent (not 4, not tabs).
- Self-closing tags where applicable.
- One attribute per line for elements with 3+ attributes.

### Naming

- Python: `snake_case` for variables/functions, `PascalCase` for classes,
  `UPPER_SNAKE_CASE` for module-level constants.
- Topics: `/lower_snake_case_with_slashes` — match canonical ROS 2 / gz
  conventions.
- SDF model and link names: `snake_case`, ASCII only.

## Package conventions

This repo has four top-level ROS 2 packages:

| Package | Build type | Purpose |
|---------|------------|---------|
| `gazebo_gymnasium_bridge` | `ament_python` | The `gym.Env` classes + `WorldController` |
| `gazebo_gymnasium_examples/gazebo_gymnasium_bringup` | `ament_cmake` | Launch files only |
| `gazebo_gymnasium_examples/gazebo_gymnasium_resources` | `ament_cmake` | SDF worlds, models, plugins (installed assets) |
| `gazebo_gymnasium_msgs` | `ament_cmake` | Custom ROS 2 message types |

When adding new resources (models, worlds, plugins), drop them in the
appropriate `gazebo_gymnasium_resources/<category>/` directory; install
rules in its `CMakeLists.txt` glob them automatically.

When adding a new launch file, put it in
`gazebo_gymnasium_bringup/launch/<env_name>.launch.py` and make sure
`gazebo_gymnasium_bringup/package.xml` lists `gazebo_gymnasium_resources`
as an `<exec_depend>` (NOT `<build_depend>` — the bringup package only
runtime-depends on the resources).

## Adding a new environment

An environment is **one `AgentSpec`**, not a per-env class — the framework runs
N copies of it in a single world as an SB3 `VecEnv`. The full walkthrough is in
[`docs/creating_your_own_agent.md`](docs/creating_your_own_agent.md). Briefly:

1. Add or convert the model (`gazebo_gymnasium_resources/models/<env>_bare/`) —
   geometry only for the harness backend (no controllers, no effort limit on
   actuated joints).
2. Define an `AgentSpec` (obs/action/reward/termination + `action_to_commands`
   and `reset_joint_state` for the harness) and `register_spec("<env>", …)` in
   `gazebo_gymnasium_bridge/envs/agent_spec.py`.
3. Add a world SDF loading the `MultiAgentHarness` plugin and a launch file that
   spawns N models (copy the `cartpole_harness` pair).
4. Add offline coverage: a spec test in `test/test_agent_spec.py` and, for ECM
   actuation, a `TestFixture` test like `test/test_harness_core.py`.
5. Add a doc page in `docs/examples/<env>.md` and link it from the
   `docs/examples/README.md` status table.

## Commit messages

- Use the imperative mood: "Add InvertedDoublePendulum env" not
  "Added ...".
- First line ≤ 70 characters.
- Reference the issue/PR in the body, not the subject.

## Reporting bugs

Open a GitHub issue with:
- ROS 2 distribution + Gazebo version (`ros2 --version` and `gz sim --version`)
- Python version (`pixi run python --version`)
- Exact reproducer (the `ros2 launch` command + the trainer invocation)
- Full stack trace if any

## License

This project is licensed under Apache-2.0 — see [`LICENSE`](LICENSE). By
submitting a contribution you agree your code is released under the same
license.
