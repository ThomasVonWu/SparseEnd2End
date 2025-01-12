# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
cmake_minimum_required(VERSION 3.22.0)

project(Sparse4dHeadFirstFrameTrtInferUnitTest CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

set(CMAKE_BUILD_TYPE "Release")

# C++  setting
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_RELEASE
    "-std=c++17 -O3 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB"
)

# CUDA setting
find_package(CUDA 11 REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# TensorRT setting
set(ENV_TensorRT_INC /mnt/env/tensorrt/TensorRT-8.6.1.6/include)
set(ENV_TensorRT_LIB /mnt/env/tensorrt/TensorRT-8.6.1.6/lib)
set(ENV_cuDNN_LIB
    /mnt/env/tensorrt/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib)
include_directories(${ENV_TensorRT_INC})
link_directories(${ENV_TensorRT_LIB} ${ENV_cuDNN_LIB})
# If you run bin file failed, you migth run command manually : `export
# LD_LIBRARY_PATH=/mnt/env/tensorrt/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib:$LD_LIBRARY_PATH`

# cuDNN RPATH setting for linking during running, it's not valid!!!
# set(CMAKE_INSTALL_RPATH ${ENV_cuDNN_LIB}) set(CMAKE_BUILD_WITH_INSTALL_RPATH
# TRUE) set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
  ${CMAKE_SOURCE_DIR}/../../../common ${CMAKE_SOURCE_DIR}/../../../preprocessor
  ${CMAKE_SOURCE_DIR}/../../tensorrt)

set(Sparse4dFirstFrameTrtInferBin
    sparse4d_head_first_frame_trt_infer_unit_test.bin)

add_executable(
  ${Sparse4dFirstFrameTrtInferBin}
  sparse4d_head_first_frame_trt_infer_unit_test.cpp
  ${CMAKE_SOURCE_DIR}/../../tensorrt/tensorrt.cpp
  ${CMAKE_SOURCE_DIR}/../../tensorrt/logging.cpp
  ${CMAKE_SOURCE_DIR}/../../preprocessor/parameters_parser.cpp)

# Ensure libdl.so.2 is exists in system path :
# "/lib/x86_64-linux-gnu/libdl.so.2".
target_link_libraries(
  ${Sparse4dFirstFrameTrtInferBin}
  gtest
  gtest_main
  gmock_main
  pthread
  dl
  cudart
  cudnn
  yaml-cpp
  nvinfer
  nvinfer_plugin
  nvonnxparser)

# target_link_options(${Sparse4dFirstFrameTrtInferBin} PRIVATE
# -Wl,-rpath,${ENV_cuDNN_LIB})

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../third_party/googletest)
set(GTEST_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/googletest)
add_subdirectory(${GTEST_SOURCE_DIR} ${GTEST_BINARY_DIR})

set(YAMLCPP_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../third_party/yaml-cpp)
set(YAMLCPP_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/yaml-cpp)
add_subdirectory(${YAMLCPP_SOURCE_DIR} ${YAMLCPP_BINARY_DIR})
