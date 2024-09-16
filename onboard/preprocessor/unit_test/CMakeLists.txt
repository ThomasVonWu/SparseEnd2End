set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin) # Define saving folder ofexecutable file
                                        # in cmake:build.

cmake_minimum_required(VERSION 3.22.0)

project(preprocess_unit_test) # Build target name.

enable_testing()
add_executable(preprocess_unit_test.bin preprocess_unit_test.cpp)

# find_package(glog REQUIRED)

target_link_libraries(preprocess_unit_test.bin gtest gtest_main gmock_main glog
                      pthread)
