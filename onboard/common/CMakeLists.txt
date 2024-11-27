cmake_minimum_required(VERSION 3.22.0)
project(CommonProject CUDA CXX)

find_package(CUDA REQUIRED)

set(COMMON_INC common.h cuda_wrapper.cuh)
set(TARGET common)

add_library(${TARGET} STATIC ${COMMON_INC})

# 设置 CUDA 架构
set_target_properties(
  ${TARGET} PROPERTIES CUDA_ARCHITECTURES 86 # 指定 CUDA 架构（例如 sm_86）
)

target_compile_options(${TARGET}
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86>)

target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CUDA)
