// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_PREPROCESSOR_IMG_AUG_WITH_BILINEARINTERPOLATION_KERNEL_H
#define ONBOARD_PREPROCESSOR_IMG_AUG_WITH_BILINEARINTERPOLATION_KERNEL_H

#include <cuda_fp16.h>

#include <cstdint>

#include "../common/common.h"
namespace sparse_end2end {
namespace preprocessor {

/// @brief RGB format image prepocess CUDA launcher: resize(bilinear
/// interpolation) -> crop -> normalization.
/// @return image augmentation with output dtype: fp32.
common::Status imgPreprocessLauncher(const std::uint8_t* raw_imgs_cuda_ptr,
                                     const std::uint32_t& num_cams,
                                     const std::uint32_t& raw_img_c,
                                     const std::uint32_t& raw_img_h,
                                     const std::uint32_t& raw_img_w,
                                     const std::uint32_t& model_input_img_h,
                                     const std::uint32_t& model_input_img_w,
                                     const float& resize_ratio,
                                     const std::uint32_t& crop_height,
                                     const std::uint32_t& crop_width,
                                     const cudaStream_t& stream,
                                     float* model_input_imgs_cuda_ptr);

/// @brief RGB format image prepocess CUDA launcher: resize(bilinear
/// interpolation) -> crop -> normalization.
/// @return image augmentation with output dtype: fp16.
common::Status imgPreprocessLauncher(const std::uint8_t* raw_imgs_cuda_ptr,
                                     const std::uint32_t& num_cams,
                                     const std::uint32_t& raw_img_c,
                                     const std::uint32_t& raw_img_h,
                                     const std::uint32_t& raw_img_w,
                                     const std::uint32_t& model_input_img_h,
                                     const std::uint32_t& model_input_img_w,
                                     const float& resize_ratio,
                                     const std::uint32_t& crop_height,
                                     const std::uint32_t& crop_width,
                                     const cudaStream_t& stream,
                                     half* model_input_imgs_cuda_ptr);

}  // namespace preprocessor
}  // namespace sparse_end2end

#endif  // ONBOARD_PREPROCESSOR_IMG_AUG_WITH_BILINEARINTERPOLATION_KERNEL_H
