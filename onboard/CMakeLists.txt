# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
cmake_minimum_required(VERSION 3.10.0)
project(SparseEnd2End)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")


include_directories(
    common
    modules/image_preprocess
    )
add_library(common STATIC  common/image_preprocess.cpp)
add_library(nn_interpol_base   STATIC  modules/image_preprocess/nn_interpol_base.cpp)

add_subdirectory(third_party/googletest)

option(BUILD_TEST "Whether or not to build the tests" ON)
if (${BUILD_TEST})
    message(STATUS "Building tests")
    enable_testing()  
    add_subdirectory(modules/image_preprocess/unit_tests )
endif ()