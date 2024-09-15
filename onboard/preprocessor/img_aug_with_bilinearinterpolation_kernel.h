// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_PREPROCESSOR_IMG_AUG_WITH_BILINEARINTERPOLATION_KERNEL_H
#define ONBOARD_PREPROCESSOR_IMG_AUG_WITH_BILINEARINTERPOLATION_KERNEL_H

#include <cuda_fp16.h>

namespace SparseEnd2End {
namespace image {

/// @brief RGB format image prepocess CUDA launcher: resize(bilinear
/// interpolation) -> crop -> normalization.
/// @return image augmentation with output dtype: fp32.
Status imgPreprocessLauncher(
    const std::uint8_t* raw_imgs_ptr, const std::uint32_t num_cams,
    const std::uint32_t raw_img_c, const std::uint32_t raw_img_h,
    const std::uint32_t raw_img_w, const std::uint32_t target_img_h,
    const std::uint32_t target_img_w, const std::uint32_t crop_h,
    const std::uint32_t crop_w, const float resize_ratio,
    const cudaStream_t& stream, float* output_img_ptr);

/// @brief RGB format image prepocess CUDA launcher: resize(bilinear
/// interpolation) -> crop -> normalization.
/// @return image augmentation with output dtype: fp16.
Status imgPreprocessLauncher(
    const std::uint8_t* raw_imgs_ptr, const std::uint32_t num_cams,
    const std::uint32_t raw_img_c, const std::uint32_t raw_img_h,
    const std::uint32_t raw_img_w, const std::uint32_t target_img_h,
    const std::uint32_t target_img_w, const std::uint32_t crop_h,
    const std::uint32_t crop_w, const float resize_ratio,
    const cudaStream_t& stream, half* output_img_ptr);

}  // namespace image
}  // namespace SparseEnd2End

#endif  // ONBOARD_PREPROCESSOR_IMG_AUG_WITH_BILINEARINTERPOLATION_KERNEL_H
