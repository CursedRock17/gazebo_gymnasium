cmake_minimum_required(VERSION 3.10)
project(gazebo_gymnasium_examples)

find_package(ament_cmake REQUIRED)

# --- CartPole ---
install(DIRECTORY cartpole/models/
  DESTINATION share/${PROJECT_NAME}/cartpole/models
)
install(DIRECTORY cartpole/worlds/
  DESTINATION share/${PROJECT_NAME}/cartpole/worlds
)
install(DIRECTORY cartpole/launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(PROGRAMS
  cartpole/scripts/cartpole_env.py
  cartpole/scripts/train_sb3.py
  DESTINATION lib/${PROJECT_NAME}/cartpole
)
# ros2 run entry point — delegates to the scripts in the cartpole/ subdirectory
install(PROGRAMS
  cartpole/scripts/cartpole_train_sb3
  DESTINATION lib/${PROJECT_NAME}
)

# --- Add new examples above this line following the same pattern ---

ament_package()
