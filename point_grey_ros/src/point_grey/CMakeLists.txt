cmake_minimum_required(VERSION 2.8.3)
project(point_grey)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  sensor_msgs
  std_msgs
)
find_package(OpenCV REQUIRED)

# find_package(Boost REQUIRED COMPONENTS system)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES point_grey
#  CATKIN_DEPENDS cv_bridge roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  include/point_grey
  ${catkin_INCLUDE_DIRS}
  /usr/include/spinnaker
)
link_directories(
    /usr/lib)
link_libraries(
    Spinnaker)

 add_executable(${PROJECT_NAME}_node src/main.cpp src/pointgrey.cpp)

 target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
   )
