# Gazebo Gymnasium
------------------------
This repository reserves as integration between OpenAI's [Gymnasium](https://gymnasium.farama.org/)
package along with ROS 2 & [Gazebo](https://gazebosim.org/docs/latest/getstarted/).
It allows user to engage reinforcement learning through simulation, then port that data to real 
life, allowing for an easier integration process of a robot model.

## Installation Process
Ensure you have the correct version of ROS 2 & Gazebo for this template.

#### Prerequisites
Install [ROS 2](https://docs.ros.org/en/jazzy/Installation.html) Jazzy
Install [Gazebo](https://gazebosim.org/docs/harmonic/install/) Harmonic
Install [ROS-GZ](https://github.com/gazebosim/ros_gz/tree/jazzy) bridge

#### Create Workspace
```
mkdir -p ~/gym_ws/src
cd ~/gym_ws/src
git clone 
```

#### Install Depdencies
```
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -i -y --rosdistro jazzy
```

#### Build Workspace
```
source /opt/ros/jazzy/setup.bash
colcon build --packages-ignore gazebo_gymnasium_bringup
source install/setup.bash
colcon build --packages-select gazebo_gymnasium_bringup
source install/setup.bash
```

