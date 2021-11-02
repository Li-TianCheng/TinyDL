cmake_minimum_required(VERSION 3.17)

project(TinyDL CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/")

find_package(MKL)
find_package(CUDA)
include_directories(./include ${MKL_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCE_FILES ./src/*)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -fopenmp")

add_subdirectory(cuda)

add_executable(TinyDL main.cpp ${SOURCE_FILES})

set_target_properties(TinyDL PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TinyDL PROPERTIES CUDA_ARCHITECTURES "35;50;72")

target_link_libraries(TinyDL ${MKL_LIBRARIES} cuda)