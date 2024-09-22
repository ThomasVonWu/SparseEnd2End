set(PREPROCESSOR_INC img_preprocessor.h
                     img_aug_with_bilinearinterpolation_kernel.h)
set(PREPROCESSOR_SRC img_preprocessor.cpp
                     img_aug_with_bilinearinterpolation_kernel.cu)

set(TARGET preprocessor)
cuda_add_library(${TARGET} STATIC ${PREPROCESSOR_SRC})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
