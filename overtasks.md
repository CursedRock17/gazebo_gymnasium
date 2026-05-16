## Tasks to complete overnight
So long as it won't comprismise the status of the project, you have free reign to provide some fixes:

- [ ] Add and commit changes made thus far, but don't push, I just want a backup of the changes
- [ ] Having to run `python3` targetting my install directory after colcon build might be a bit awkward. Is there a way to just run the files in the project instead?
- [ ] Currently there's no way to "debug" how well the policy is doing while it runs other than an eye test. I know SB3 has access to tools like tensorboard, is there anything else we should use for debugging?
- [ ] Speeds are currently solid now. Is there anywhere in the code base where you can optimize with the CPU, i.e using more cores, using better programming techiniques, etc.
- [ ] The Cartpole right now doesn't die fast enough after failing, remeber all SDF files are in meters, so if the bounds are the suggested +/- 2.4 meters, then it may be too far
- [ ] Typically with cartpole, you just try to get to 500 steps, I believe that's a good way to start
- [ ] This may need to be better suited as a python package, for distrubtion purposes, check out the '/python-package' skills that I currently have access to, it may provide you more guidance.
- [ ] With the current cartpole example, do I have access to any ROS 2 topics. I don't want to move past Cartpole until I assert easy and great usage, but maybe a secondary one with ROS would be good, if there isn't capabilites to check ROS topics already.
- [ ] Make sure our documentation is easy to read still and up to date with all the changes thus far
- [ ] Ensure the repository in ~/gym_ws for gazebo_gymnasium is still the same as we have here

So again, if you can't solve a task due to permissions/risk, skip it and leave it unchecked for me to deal with in the morning
