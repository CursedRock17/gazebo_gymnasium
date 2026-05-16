cmake_minimum_required(VERSION 3.8)
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
  DESTINATION share/${PROJECT_NAME}/cartpole/launch
)
install(PROGRAMS
  cartpole/scripts/cartpole_env.py
  cartpole/scripts/train_sb3.py
  DESTINATION lib/${PROJECT_NAME}/cartpole
)

# --- Add new examples above this line following the same pattern ---

ament_package()
