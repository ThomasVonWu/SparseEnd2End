cmake_minimum_required(VERSION 3.22.0)

project(CommonUnitTest CUDA CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_RELEASE
    "-std=c++17 -O3 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB"
)

find_package(CUDA 11 REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

include_directories(${CMAKE_SOURCE_DIR}/.. ${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(CudaWrapperBin cuda_wrapper_unit_test.bin)
add_executable(${CudaWrapperBin} cuda_wrapper_unit_test.cpp)

set(GTEST_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/googletest)
add_subdirectory(${CMAKE_SOURCE_DIR}/../../third_party/googletest
                 ${GTEST_BINARY_DIR})
set(GTEST_BUILD_TESTS
    OFF
    CACHE BOOL "Disable building Google Test's own tests")

target_link_libraries(${CudaWrapperBin} PRIVATE gtest gtest_main gmock_main
                                                pthread cudart)
