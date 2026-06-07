## Tasks to Execute
I'll be away from the keyboard, thus you have permissions to execute any program given it's not destructive. If you cannot make a decisino, simple make a note in this file and move on. Don't execute anything that coudl be harmful, but don't ever stop to ask for validation, as I cannot give it.
You have permissions to serach through files on the device, access the internet, run shell comamands.
Make sure we're using Opus4.7 Model (\model)

## Current asks
- [~] Run /loop for a 10-agent CartPole vectorized environment, mean reward ≥ 480 over the last 20 episodes within 4 iterations — **BLOCKED FROM CLAUDE'S SIDE**:
    - Generated `worlds/cartpole_multi_10.sdf` for the requested 10-agent count.
    - Tried to spin up `gz sim -s -r cartpole_multi_10.sdf` from the agent's non-interactive shell to bootstrap the loop. Result: **deterministic segfault inside `libpython3.12.so.1.0 :: PyImport_ImportModule → PyUnicode_New`** when gz sim's `python-system-loader` plugin loads the sync-gate Python module. Reproduced with multiple env configurations (full ROS source, `env -i`, custom `PYTHONHOME`, custom `PYTHONPATH`) — all crash at the same address.
    - Same gz sim launches work in YOUR interactive shell. Some environmental difference (likely libpython allocator state) makes the embedded interpreter unusable from the agent's session.
    - **Fallback path**: kick off the iteration yourself with one terminal:
        ```
        ros2 launch gazebo_gymnasium_bringup cartpole_multi.launch.py \
            n_agents:=10 headless:=true
        # in a second terminal:
        python training_scripts/train_cartpole_multi_agent.py \
            --n_agents 10 --timesteps 100000
        ```
      After it finishes, paste the last 20–40 `[EpisodeSummary]` lines back to me and I'll diagnose / propose the next hyperparameter delta. That gives us the same outer loop, just with you as the launcher.
    - Generated 10-agent world is staged and validated (`scripts/validate_sdfs.py` passes); resources package rebuilt with `--symlink-install` so it's already installed.

## Completion Notes
- [x]  Process ROS 2's developer guide — **DONE**: added `CONTRIBUTING.md` covering quick-checks-before-PR, code style, naming, package conventions (ament_python vs ament_cmake roles), the "adding a new environment" walkthrough, commit message conventions, bug-reporting template, license clarification. Aligns with the Developer Guide rather than copying it verbatim.
- [x]  Process ROS 2's Code style language guide — **DONE**: documented Python (PEP 8 / PEP 257, 88-char line limit, import ordering, naming conventions) and XML (2-space indent, attribute layout) rules in `CONTRIBUTING.md`. Added `[flake8]` (max-line-length=88, ignores E203/W503) and `[pep257]` blocks to `gazebo_gymnasium_bridge/setup.cfg` so the existing `test_flake8.py` / `test_pep257.py` use the right config.
- [x]  Process ROS 2's Code quality guide — **DONE** at the test-coverage level: every env class now has contract tests (`test_env_contract.py`) AND library-integration smoke tests (`test_library_integration.py`). Total 66 tests passing. The Quality Guide's other axes (deterministic builds, documented public API surface, performance benchmarks) are partially covered by the existing Sphinx setup but would need explicit benchmark scripts for the "performance" axis — left for future work.


### Combined train+sim launch
- New `cartpole_train.launch.py` runs both gz sim AND the training script via `ExecuteProcess`:
    ```
    ros2 launch gazebo_gymnasium_bringup cartpole_train.launch.py
    ros2 launch gazebo_gymnasium_bringup cartpole_train.launch.py trainer:=custom
    ros2 launch gazebo_gymnasium_bringup cartpole_train.launch.py timesteps:=500000
    ```
- Trainer subprocess is delayed by `delay_train:=5.0` seconds so the sim has time to advertise services; `WorldController.reset` retries internally too, so this is mostly cosmetic.

### Documentation
- `docs/examples/README.md` — index of tutorials with status table.
- `docs/examples/cartpole.md` — full walk-through: run instructions, algorithm explanation, observation/action spaces, reward function, expected training trajectory, Foxglove panel setup.
- `docs/resources/README.md` — placeholder for screenshots and reference media.
- `docs/sphinx/` — Sphinx project skeleton. Build with:
    ```
    venv/bin/python -m sphinx -b html docs/sphinx docs/sphinx/_build
    ```
    Then open `docs/sphinx/_build/index.html`. Pages auto-generated for `gazebo_gymnasium_bridge.envs`, `world_control`, and `PPO_agent`. Mock imports configured for gz/torch so the build doesn't require those at doc-build time.


## Future Goals - DO NOT Complete
- Hand-rename `half_cheetah` SDF's duplicate frame names so it can be wired up alongside the other locomotion envs.
- Wire env classes + training scripts for the 9 envs that currently have only stub plugins: `ant`, `hopper`, `humanoid`, `humanoidstandup`, `point`, `pusher`, `pusher_v5`, `reacher`, `swimmer`, `walker2d`, `walker2d_v5`. Reacher (TD3) is the natural next slot.
- Add a `.github/workflows/colcon-test.yml` so the test suite actually runs in CI.
- Performance benchmarks (Quality Guide axis): wall-clock steps/second per env vs canonical MuJoCo.

- [ ] ROS 2 Reps
- [ ] Setup [Read-the-docs](https://about.readthedocs.com/?ref=app.readthedocs.org) for easy hosting of documentation when time is ready

