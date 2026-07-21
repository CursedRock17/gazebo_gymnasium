cmake_minimum_required(VERSION 3.8)
project(gazebo_gymnasium_msgs)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# External dependencies
find_package(ament_cmake REQUIRED)
# Necessary to build the actual msg/srv files
find_package(rosidl_default_generators REQUIRED)
# Excess msg interfaces 
find_package(std_msgs REQUIRED)

# Group together all of our messages
set(msg_files
  "msg/TrainingEpisode.msg"
  "msg/EnvMetrics.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs
  ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
