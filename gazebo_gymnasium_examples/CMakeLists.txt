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
  cartpole/scripts/cartpole_continuous_env.py
  cartpole/scripts/cartpole_fixture_env.py
  cartpole/scripts/train_sb3.py
  cartpole/scripts/train_a2c.py
  cartpole/scripts/train_custom_ppo.py
  cartpole/scripts/train_fixture.py
  DESTINATION lib/${PROJECT_NAME}/cartpole
)
# ros2 run entry points — delegate to the scripts in the cartpole/ subdirectory
install(PROGRAMS
  cartpole/scripts/cartpole_train_sb3
  DESTINATION lib/${PROJECT_NAME}
)

# --- Inverted Pendulum ---
install(DIRECTORY inverted_pendulum/models/
  DESTINATION share/${PROJECT_NAME}/inverted_pendulum/models
)
install(DIRECTORY inverted_pendulum/worlds/
  DESTINATION share/${PROJECT_NAME}/inverted_pendulum/worlds
)
install(DIRECTORY inverted_pendulum/launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(PROGRAMS
  inverted_pendulum/scripts/inverted_pendulum_env.py
  inverted_pendulum/scripts/train_ppo.py
  DESTINATION lib/${PROJECT_NAME}/inverted_pendulum
)

# --- Hopper ---
install(DIRECTORY hopper/models/
  DESTINATION share/${PROJECT_NAME}/hopper/models
)
install(DIRECTORY hopper/worlds/
  DESTINATION share/${PROJECT_NAME}/hopper/worlds
)
install(DIRECTORY hopper/launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(PROGRAMS
  hopper/scripts/hopper_env.py
  hopper/scripts/train_ppo.py
  DESTINATION lib/${PROJECT_NAME}/hopper
)

# --- Half Cheetah ---
install(DIRECTORY half_cheetah/worlds/
  DESTINATION share/${PROJECT_NAME}/half_cheetah/worlds
)
install(DIRECTORY half_cheetah/launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(PROGRAMS
  half_cheetah/scripts/half_cheetah_env.py
  half_cheetah/scripts/train_ppo.py
  DESTINATION lib/${PROJECT_NAME}/half_cheetah
)

# --- Ant ---
install(DIRECTORY ant/worlds/
  DESTINATION share/${PROJECT_NAME}/ant/worlds
)
install(DIRECTORY ant/launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(PROGRAMS
  ant/scripts/ant_env.py
  ant/scripts/train_ppo.py
  DESTINATION lib/${PROJECT_NAME}/ant
)

# --- Reacher ---
install(DIRECTORY reacher/worlds/
  DESTINATION share/${PROJECT_NAME}/reacher/worlds
)
install(DIRECTORY reacher/launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(PROGRAMS
  reacher/scripts/reacher_env.py
  reacher/scripts/train_ppo.py
  DESTINATION lib/${PROJECT_NAME}/reacher
)

ament_package()
