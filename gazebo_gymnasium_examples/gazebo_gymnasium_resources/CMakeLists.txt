cmake_minimum_required(VERSION 3.8)
project(gazebo_gymnasium_resources)

# Necessary Packages
find_package(ament_cmake REQUIRED)

## Gazebo Versioning ##
find_package(gz-cmake3 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
set(GZ_PLUGIN_VER ${gz-plugin2_VERSION_MAJOR})
find_package(gz-common5 REQUIRED COMPONENTS profiler)
set(GZ_COMMON_VER ${gz-common5_VERSION_MAJOR})
# Harmonic
find_package(gz-sim REQUIRED)
set(GZ_SIM_VER ${gz-sim8_VERSION_MAJOR})
# set(GZ_SIM_VER 8.9.0)
find_package(gz-sensors8 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-common REQUIRED)
find_package(gz-transport REQUIRED)
message(STATUS "Compiling against Gazebo Harmonic")

# Find Inner Packages
#find_package(gazebo_gymnasium_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Create Install Directories for our Resources
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)
install(DIRECTORY worlds/ DESTINATION share/${PROJECT_NAME}/worlds)
install(DIRECTORY plugins/ DESTINATION share/${PROJECT_NAME}/plugins)

# Add the necessary hooks for our resources to be found by the system
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.sh.in")

ament_package()
