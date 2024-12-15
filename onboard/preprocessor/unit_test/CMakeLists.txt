cmake_minimum_required(VERSION 3.22.0)

project(PreprocessorUnitTest CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_RELEASE
    "-std=c++17 -O3 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB"
)

include_directories(${CMAKE_SOURCE_DIR}/../common ${CMAKE_SOURCE_DIR}/..)
set(ParametersParserBin parameters_parser_unit_test.bin)
add_executable(
  ${ParametersParserBin} parameters_parser_unit_test.cpp
                         ${CMAKE_SOURCE_DIR}/../parameters_parser.cpp)
target_link_libraries(
  ${ParametersParserBin}
  gtest
  gtest_main
  gmock_main
  glog
  pthread
  yaml-cpp)

set(YAMLCPP_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../third_party/yaml-cpp)
set(YAMLCPP_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/yaml-cpp)
add_subdirectory(${YAMLCPP_SOURCE_DIR} ${YAMLCPP_BINARY_DIR})

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../third_party/googletest)
set(GTEST_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/googletest)
add_subdirectory(${GTEST_SOURCE_DIR} ${GTEST_BINARY_DIR})
