cmake_minimum_required(VERSION 3.17)
project(TinyDL)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/")

find_package(MKL)
include_directories(./include ${MKL_INCLUDE_DIR})

file(GLOB_RECURSE SOURCE_FILES ./src/*)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -fopenmp")

add_executable(TinyDL main.cpp ${SOURCE_FILES})

target_link_libraries(TinyDL ${MKL_LIBRARIES})