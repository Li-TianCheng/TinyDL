cmake_minimum_required(VERSION 3.17)
project(TinyDL)

set(CMAKE_CXX_STANDARD 14)

include_directories(./include)

file(GLOB_RECURSE SOURCE_FILES ./src/*)

add_executable(TinyDL main.cpp ${SOURCE_FILES})