cmake_minimum_required(VERSION 3.22.0)

project(PreprocessorUnitTest CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_RELEASE
    "-std=c++17 -O3 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB"
)

#[[
# Build Eigen dependency if you make install `eigen3` manually.
# find_package: default search and find it in system path `/usr/include/eigen3` or `/usr/local/include/eigen3` if make install `eigen3` manually.
find_package(Eigen3 REQUIRED)
if((Eigen3_FOUND) AND (DEFINED EIGEN3_INCLUDE_DIR))
  message(STATUS "Found Eigen3: " ${EIGEN3_INCLUDE_DIR})
  include_directories(${EIGEN3_INCLUDE_DIR})
else()
  message(FATAL_ERROR "EIGEN3 not found.")
endif()
]]

set(EigenBin eigen.bin)
add_executable(${EigenBin} eigen_third_party_unit_test.cpp)

target_link_libraries(${EigenBin} gtest gtest_main gmock_main pthread eigen)

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../../third_party/googletest)
set(GTEST_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/googletest)
add_subdirectory(${GTEST_SOURCE_DIR} ${GTEST_BINARY_DIR})

set(EIGEN_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../../third_party/eigen)
set(EIGEN_INCLUDE_DIR ${CMAKE_BINARY_DIR}/third_party/eigen)
add_subdirectory(${EIGEN_SOURCE_DIR} ${EIGEN_INCLUDE_DIR})
