cmake_minimum_required(VERSION 3.22.0)

project(CommonProject CUDA CXX)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_RELEASE
    "-std=c++17 -O3 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB"
)

find_package(CUDA 11 REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(TARGET1 common)
add_library(${TARGET1} INTERFACE)
target_include_directories(${TARGET1} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

set(TARGET2 cuda_wrapper)
add_library(${TARGET2} INTERFACE)
target_include_directories(${TARGET2} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
