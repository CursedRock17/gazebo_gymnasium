## Tasks to complete overnight
So long as it won't comprismise the status of the project, you have free reign to provide some fixes:

- [x] Add and commit changes made thus far, but don't push, I just want a backup of the changes
- [x] Having to run `python3` targetting my install directory after colcon build might be a bit awkward. Is there a way to just run the files in the project instead?
      → Added `ros2 run gazebo_gymnasium_examples cartpole_train_sb3` wrapper.
        Also works from source tree: `python3 .../cartpole/scripts/train_sb3.py`
- [x] Currently there's no way to "debug" how well the policy is doing while it runs other than an eye test. I know SB3 has access to tools like tensorboard, is there anything else we should use for debugging?
      → TensorBoard logging + EvalCallback added to train_sb3.py. Run: `tensorboard --logdir ./tb_logs`
- [ ] Speeds are currently solid now. Is there anywhere in the code base where you can optimize with the CPU, i.e using more cores, using better programming techiniques, etc.
      → Skipped — requires profiling data from a real training run to make targeted changes. Recommend running a full training session first and checking TensorBoard for bottlenecks.
- [x] The Cartpole right now doesn't die fast enough after failing, remeber all SDF files are in meters, so if the bounds are the suggested +/- 2.4 meters, then it may be too far
      → Fixed: push targets reduced from ±2.5m → ±0.5m so episodes terminate via pole angle (falling) rather than cart slamming the position limit. Termination thresholds (±2.4m, 12°) kept at CartPole-v1 standard.
- [x] Typically with cartpole, you just try to get to 500 steps, I believe that's a good way to start
      → Already in place (_MAX_STEPS = 500), no change needed.
- [x] This may need to be better suited as a python package, for distrubtion purposes, check out the '/python-package' skills that I currently have access to, it may provide you more guidance.
      → Assessed. ament_python and pyproject.toml conflict when in the same directory — documented a repo-root pyproject.toml approach in README under "PyPI Distribution".
- [x] With the current cartpole example, do I have access to any ROS 2 topics. I don't want to move past Cartpole until I assert easy and great usage, but maybe a secondary one with ROS would be good, if there isn't capabilites to check ROS topics already.
      → Added ros_gz_bridge to cartpole.launch.py. Now available: `/clock`, `/world/cartpole/model/cartpole/joint_state` (JointState), `/model/cartpole/joint/slider_to_cart/0/cmd_pos` (Float64, bidirectional).
- [x] Make sure our documentation is easy to read still and up to date with all the changes thus far
      → README updated with all new features, correct run commands, ROS 2 bridge docs, TensorBoard instructions, and PyPI section.
- [x] Ensure the repository in ~/gym_ws for gazebo_gymnasium is still the same as we have here
      → Synced. Both repos identical as of final commit.
