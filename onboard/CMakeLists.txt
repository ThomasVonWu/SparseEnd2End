# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
cmake_minimum_required(VERSION 3.22.0)
project(SparseEnd2EndOnboard CUDA CXX)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE
    "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB"
)
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS_RELEASE
    "-Werror=all-warnings -Xcompiler -std=c++17,-Wextra,-Wall,-Wno-deprecated-declarations,-O3 -DENABLE_TEXT_BACKEND_STB"
)
set(CUDA_NVCC_FLAGS_DEBUG "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")

get_filename_component(CUDA_ROOT_PATH $ENV{ENV_CUDA_INC} DIRECTORY)
set(CMAKE_PREFIX_PATH ${CUDA_ROOT_PATH})
# or set(CUDA_TOOLKIT_ROOT_DIR "/path/to/cuda")
find_package(CUDA REQUIRED)

include_directories($ENV{ENV_TensorRT_INC} $ENV{ENV_CUDA_INC})
link_directories($ENV{ENV_TensorRT_LIB} $ENV{ENV_CUDA_LIB} build)

add_subdirectory(common preprocessor)
add_executable(main.bin main.cpp)
target_link_libraries(main.bin PRIVATE common)
