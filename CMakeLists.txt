cmake_minimum_required(VERSION 3.5)
project(lidar_processing)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation visualization)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Filter node
add_executable(filter_node src/filter_node.cpp)
ament_target_dependencies(filter_node rclcpp sensor_msgs pcl_conversions)
target_link_libraries(filter_node ${PCL_LIBRARIES})

# Segmentation node
add_executable(segmentation_node src/segmentation_node.cpp)
ament_target_dependencies(segmentation_node rclcpp sensor_msgs pcl_conversions visualization_msgs)
target_link_libraries(segmentation_node ${PCL_LIBRARIES})

# Clustering node
add_executable(clustering_node src/clustering_node.cpp)
ament_target_dependencies(clustering_node rclcpp sensor_msgs pcl_conversions visualization_msgs)
target_link_libraries(clustering_node ${PCL_LIBRARIES})

install(TARGETS filter_node segmentation_node clustering_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

