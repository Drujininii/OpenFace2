cmake_minimum_required(VERSION 3.5)
project(try_dlib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
#find_package(dlib_DIR REQUIRED)
include(./dlib/dlib/cmake)

add_definitions(-DSOME_IMPORTANT_DEFINITION)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3")

set(SOURCE_FILES main.cpp)
add_executable(try_dlib ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(try_dlib ${OpenCV_LIBS} dlib::dlib)