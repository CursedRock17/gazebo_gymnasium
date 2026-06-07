# Contributing to gazebo_gymnasium

Thanks for considering a contribution. This repo aims to align with the
[ROS 2 Developer Guide](https://docs.ros.org/en/jazzy/Contributing/Developer-Guide.html)
and the
[ROS 2 Code Style Guide](https://docs.ros.org/en/jazzy/The-ROS2-Project/Contributing/Code-Style-Language-Versions.html);
this document only lists what's project-specific.

## Quick checks before opening a PR

```bash
# Build the workspace (uses the in-repo venv)
colcon build --symlink-install

# Lint
./venv/bin/python -m pytest gazebo_gymnasium_bridge/test/ -v
ament_lint_auto  # if you have ROS 2 sourced

# Smoke-test all envs (no Gazebo needed — transport is mocked)
./venv/bin/python -m pytest gazebo_gymnasium_bridge/test/test_library_integration.py -v

# Static SDF checks (no Gazebo needed — pure XML inspection)
./venv/bin/python -m pytest gazebo_gymnasium_bridge/test/test_sdf_validity.py -v

# Semantic SDF checks via `gz sdf --check` (skips cleanly if `gz` not on PATH)
./venv/bin/python -m pytest gazebo_gymnasium_bridge/test/test_sdf_loads_in_gazebo.py -v

# Full ROS 2 ament lint suite (flake8 + pep257 + copyright headers)
scripts/lint.sh
```

`scripts/lint.sh` wraps the three ament linters with project-specific
config:

- `ament_flake8.ini` overrides the bundled config to permit double quotes
  (modern Python convention; ament defaults to single).
- `ament_pep257` is invoked with `--add-ignore` for D212/D213 (we use
  first-line docstring summaries, per PEP 257) and the D406-D415 section
  rules (we don't use Google/NumPy section docstrings).
- `ament_copyright` checks every project Python file has an Apache-2.0
  header.

If you add a new Python file, run `./venv/bin/python
scripts/add_license_headers.py` once to prepend the standard header (the
helper is idempotent — already-headered files are skipped).

All tests must pass before review. CI runs `colcon test` on every PR.

## Code style

### Python (most of the codebase)

- **PEP 8 / PEP 257** via `flake8` and `ament_pep257`. The existing
  `test_flake8.py` and `test_pep257.py` enforce this — don't disable them.
- **88-character line limit** (matches black's default; ROS 2 default is 99
  but we use 88 to keep the code readable in side-by-side diffs).
- **Imports**: standard library → third-party → first-party, each group
  separated by a blank line. `gz.*` and `gymnasium` are third-party.
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

This repo has three top-level ROS 2 packages plus a learning library:

| Package | Build type | Purpose |
|---------|------------|---------|
| `gazebo_gymnasium_bridge` | `ament_python` | The `gym.Env` classes + `WorldController` |
| `gazebo_gymnasium_examples/gazebo_gymnasium_bringup` | `ament_cmake` | Launch files only |
| `gazebo_gymnasium_examples/gazebo_gymnasium_resources` | `ament_cmake` | SDF worlds, models, plugins (installed assets) |
| `gazebo_gymnasium_msgs` | `ament_cmake` | Custom ROS 2 message types |
| `gazebo_gymnasium_reinforcement_learning` | (pure Python) | Algorithm implementations (custom PPO, etc.) |

When adding new resources (models, worlds, plugins), drop them in the
appropriate `gazebo_gymnasium_resources/<category>/` directory; install
rules in its `CMakeLists.txt` glob them automatically.

When adding a new launch file, put it in
`gazebo_gymnasium_bringup/launch/<env_name>.launch.py` and make sure
`gazebo_gymnasium_bringup/package.xml` lists `gazebo_gymnasium_resources`
as an `<exec_depend>` (NOT `<build_depend>` — the bringup package only
runtime-depends on the resources).

## Adding a new environment

The canonical pattern lives in [`docs/examples/`](docs/examples/README.md).
Briefly:

1. Add or convert the model (`gazebo_gymnasium_resources/models/<env>/`).
2. Add a world that loads it (`gazebo_gymnasium_resources/worlds/<env>.sdf`).
3. Add a sync-gate plugin
   (`gazebo_gymnasium_resources/plugins/<env>_learner.py`) following the
   pattern in `cartpole_learner.py` or `inverted_pendulum_learner.py`.
4. Add a `gym.Env` class in `gazebo_gymnasium_bridge/envs/<env>.py` and
   export it from `envs/__init__.py`.
5. Add a launch file in `gazebo_gymnasium_bringup/launch/`.
6. Add an integration test in
   `gazebo_gymnasium_bridge/test/test_library_integration.py`.
7. Add a doc page in `docs/examples/<env>.md` and link it from the
   `docs/examples/README.md` status table.

## Commit messages

- Use the imperative mood: "Add InvertedDoublePendulum env" not
  "Added ...".
- First line ≤ 70 characters.
- Reference the issue/PR in the body, not the subject.

## Reporting bugs

Open a GitHub issue with:
- ROS 2 distribution + Gazebo version (`ros2 --version` and `gz sim --version`)
- Python version (`./venv/bin/python --version`)
- Exact reproducer (the `ros2 launch` command + the trainer invocation)
- Full stack trace if any

## License

This project is licensed under Apache-2.0 — see [`LICENSE`](LICENSE). By
submitting a contribution you agree your code is released under the same
license.
