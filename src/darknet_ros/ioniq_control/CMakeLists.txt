cmake_minimum_required(VERSION 2.6)

project(ioniq_control)

find_package(serial)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  darknet_ros_msgs
  roscpp
  std_msgs
)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ioniq_control
  CATKIN_DEPENDS darknet_ros_msgs roscpp std_msgs
  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

include_directories(
	/usr/local/include
	include
	src
        ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/serial/include
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(
    /tmp/usr/local/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/serial/lib
    /usr/local/lib
    /usr/libs
    libs
)

link_libraries(
    serial
        #libPCBUSB.dylib
        #libserial.dylib
) # libPCBUSB.0.8.dylib

# Add OpenCV Libraries
link_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

add_executable(${PROJECT_NAME}_node
    #src/main.cpp
    src/ioniq_control_node.cpp
    src/parser_vercpp.cpp
    src/coord_map.cpp
    src/CAN_RW__thread__.cpp
    src/GNSS_receive__thread__.cpp
    src/dead_reckoning__thread__.cpp
    src/path_follow__thread__.cpp
    src/ROS_enable__thread__.cpp
    src/decision__thread__.cpp

    include/env_setting.h
    include/pd_control.h
    include/coord_map.h
    include/parser_vercpp.hpp
    include/TSR_receive.h
    include/ROS_header.hpp
)

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
)
