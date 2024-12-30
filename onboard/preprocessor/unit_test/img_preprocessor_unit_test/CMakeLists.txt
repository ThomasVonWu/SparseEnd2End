cmake_minimum_required(VERSION 3.22.0)

project(PreprocessorUnitTest CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

set(CMAKE_BUILD_TYPE "Release")

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

# Including Setting
include_directories(${CMAKE_SOURCE_DIR}/../../../common
                    ${CMAKE_SOURCE_DIR}/../..)

# Build static library : libimg_preprocessor_cuda.a
set(IMGPREPROCESSORCUDA_SRC
    ${CMAKE_SOURCE_DIR}/../../img_preprocessor.cpp
    ${CMAKE_SOURCE_DIR}/../../img_aug_with_bilinearinterpolation_kernel.cu)
set(IMGPREPROCESSORCUDA_TARGET img_preprocessor_cuda)
add_library(${IMGPREPROCESSORCUDA_TARGET} STATIC ${IMGPREPROCESSORCUDA_SRC})

# Build Executable bin : img_preprocessor_unit_test.bin
set(ImgPreprocessorBin img_preprocessor_unit_test.bin)
add_executable(
  ${ImgPreprocessorBin} img_preprocessor_unit_test.cpp
                        ${CMAKE_SOURCE_DIR}/../../parameters_parser.cpp)

target_link_libraries(
  ${ImgPreprocessorBin}
  gtest
  gtest_main
  gmock_main
  glog
  pthread
  yaml-cpp
  cudart
  ${IMGPREPROCESSORCUDA_TARGET})

set(YAMLCPP_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../../third_party/yaml-cpp)
set(YAMLCPP_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/yaml-cpp)
add_subdirectory(${YAMLCPP_SOURCE_DIR} ${YAMLCPP_BINARY_DIR})

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../../third_party/googletest)
set(GTEST_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/googletest)
add_subdirectory(${GTEST_SOURCE_DIR} ${GTEST_BINARY_DIR})
