cmake_minimum_required(VERSION 3.5)
project(lidar_processing)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find ROS2 and PCL dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)

# Find PCL with components needed for clustering + bounding boxes + visualization
find_package(PCL REQUIRED COMPONENTS common io filters segmentation visualization)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# --- Filter node ---
add_executable(filter_node
  ${CMAKE_CURRENT_SOURCE_DIR}/src/filter_node.cpp
)
ament_target_dependencies(filter_node
  rclcpp
  sensor_msgs
  pcl_conversions
)
target_link_libraries(filter_node ${PCL_LIBRARIES})

# --- Segmentation node ---
add_executable(segmentation_node
  ${CMAKE_CURRENT_SOURCE_DIR}/src/segmentation_node.cpp
)
ament_target_dependencies(segmentation_node
  rclcpp
  sensor_msgs
  pcl_conversions
)
target_link_libraries(segmentation_node ${PCL_LIBRARIES})

# --- Clustering node ---
add_executable(clustering_node
  ${CMAKE_CURRENT_SOURCE_DIR}/src/clustering_node.cpp
)
ament_target_dependencies(clustering_node
  rclcpp
  sensor_msgs
  pcl_conversions
  visualization_msgs
)
target_link_libraries(clustering_node ${PCL_LIBRARIES})

# Install executables
install(TARGETS
  filter_node
  segmentation_node
  clustering_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
