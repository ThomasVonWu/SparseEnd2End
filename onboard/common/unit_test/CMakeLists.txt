cmake_minimum_required(VERSION 3.22.0)

# project(cuda_wrapper_unit_test) # Build target name.

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin) # Define saving folder of executable
                                        # file in cmake:build.

# 使用 add_subdirectory 引入父目录的 CMakeLists.txt 配置（确保父目录的静态库 common 已经创建）
# add_subdirectory(.. common)

add_executable(cuda_wrapper_unit_test.bin cuda_wrapper_unit_test.cpp)

target_link_libraries(
  cuda_wrapper_unit_test.bin
  common # 上一级静态库的依赖
  gtest
  gtest_main
  gmock_main
  glog
  pthread)

# 1.mkdir build 2.cd build 3.cmake .. 4.make
