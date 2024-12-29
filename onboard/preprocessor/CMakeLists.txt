cmake_minimum_required(VERSION 3.22.0)

project(Preprocessor CXX CUDA)

set(IMGPREPROCESSCUDA_INC img_preprocessor.h
                          img_aug_with_bilinearinterpolation_kernel.h)
set(IMGPREPROCESSCUDA_SRC img_preprocessor.cpp
                          img_aug_with_bilinearinterpolation_kernel.cu)
set(IMGPREPROCESSCUDA_TARGET img_preprocessor_cuda)

include_directories(${CMAKE_SOURCE_DIR}/../common IMGPREPROCESSCUDA_INC)

# CUDA setting
find_package(CUDA 11 REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

add_library(${IMGPREPROCESSCUDA_TARGET} STATIC ${IMGPREPROCESSCUDA_SRC})

target_link_libraries(${IMGPREPROCESSCUDA_TARGET} cudart)
