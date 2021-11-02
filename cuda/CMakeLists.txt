cmake_minimum_required(VERSION 3.17)

project(cuda CUDA)
set(CMAKE_CUDA_STANDARD 14)

include_directories(./include)
aux_source_directory(src SOURCES_FILES)

add_library(cuda ${SOURCES_FILES})

set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda PROPERTIES CUDA_ARCHITECTURES "35;50;72")