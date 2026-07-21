cmake_minimum_required(VERSION 3.8)
project(gazebo_gymnasium_bringup)

# External Dependencies
find_package(ament_cmake REQUIRED)

# NOTE: gazebo_gymnasium_resources is a RUNTIME dependency (launches use
# get_package_share_directory() to locate world / model files). It is declared
# as <exec_depend> in package.xml so colcon orders the build correctly, but we
# don't find_package it here because nothing at build time consumes it. The
# earlier `find_package(gazebo_gymnasium_resources REQUIRED)` was the cause of
# the "bringup builds before resources" failure when both packages were
# rebuilt from scratch.

# Install project launch files
install(
  DIRECTORY
    launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install project configuration files
install(
  DIRECTORY
    config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install helper scripts (e.g. spawn_multi_cartpoles.py invoked by launch).
install(
  DIRECTORY
    scripts/
  DESTINATION share/${PROJECT_NAME}/scripts
  USE_SOURCE_PERMISSIONS
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
