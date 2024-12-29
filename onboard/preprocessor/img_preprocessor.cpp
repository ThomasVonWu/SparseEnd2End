// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "img_preprocessor.h"

#include <iostream>

#include "img_aug_with_bilinearinterpolation_kernel.h"

namespace sparse_end2end {
namespace preprocessor {

ImagePreprocessor::ImagePreprocessor(const common::E2EParams& params) : params_(params) {}

common::Status ImagePreprocessor::forward(const common::CudaWrapper<std::uint8_t>& raw_imgs,
                                          const cudaStream_t& stream,
                                          const common::CudaWrapper<float>& model_input_imgs) {
  if (raw_imgs.getSize() != params_.preprocessor_params.num_cams * params_.preprocessor_params.raw_img_c *
                                params_.preprocessor_params.raw_img_h * params_.preprocessor_params.raw_img_w) {
    // LOG_ERROR(kLogContext, "Input raw imgs' size mismatches with params.");
    std::cout << "[ERROR] Input raw imgs' size mismatches with params!" << std::endl;

    return common::Status::kImgPreprocesSizeErr;
  }

  if (model_input_imgs.getSize() !=
      params_.preprocessor_params.num_cams *
          (params_.preprocessor_params.model_input_img_c * params_.preprocessor_params.model_input_img_h *
           params_.preprocessor_params.model_input_img_w)) {
    // LOG_ERROR(kLogContext, "Model input imgs' size mismatches with params.");
    std::cout << "[ERROR] Model input imgs' size mismatches with params!" << std::endl;

    return common::Status::kImgPreprocesSizeErr;
  }

  const common::Status Ret_Code = imgPreprocessLauncher(
      raw_imgs.getCudaPtr(), params_.preprocessor_params.num_cams, params_.preprocessor_params.raw_img_c,
      params_.preprocessor_params.raw_img_h, params_.preprocessor_params.raw_img_w,
      params_.preprocessor_params.model_input_img_h, params_.preprocessor_params.model_input_img_w,
      params_.preprocessor_params.resize_ratio, params_.preprocessor_params.crop_height,
      params_.preprocessor_params.crop_width, stream, model_input_imgs.getCudaPtr());

  return Ret_Code;
}

common::Status ImagePreprocessor::forward(const common::CudaWrapper<std::uint8_t>& raw_imgs,
                                          const cudaStream_t& stream,
                                          const common::CudaWrapper<half>& model_input_imgs) {
  if (raw_imgs.getSize() != params_.preprocessor_params.num_cams *
                                (params_.preprocessor_params.raw_img_c * params_.preprocessor_params.raw_img_h *
                                 params_.preprocessor_params.raw_img_w)) {
    // LOG_ERROR(kLogContext, "Input raw imgs' size mismatches with params.");
    std::cout << "[ERROR] Input raw imgs' size mismatches with params!" << std::endl;

    return common::Status::kImgPreprocesSizeErr;
  }

  if (model_input_imgs.getSize() !=
      params_.preprocessor_params.num_cams * params_.preprocessor_params.model_input_img_c *
          params_.preprocessor_params.model_input_img_h * params_.preprocessor_params.model_input_img_w) {
    // LOG_ERROR(kLogContext, "Model input imgs' size mismatches with params.");
    std::cout << "[ERROR] Model input imgs' size mismatches with params!" << std::endl;

    return common::Status::kImgPreprocesSizeErr;
  }

  common::Status Ret_Code = imgPreprocessLauncher(
      raw_imgs.getCudaPtr(), params_.preprocessor_params.num_cams, params_.preprocessor_params.raw_img_c,
      params_.preprocessor_params.raw_img_h, params_.preprocessor_params.raw_img_w,
      params_.preprocessor_params.model_input_img_h, params_.preprocessor_params.model_input_img_w,
      params_.preprocessor_params.resize_ratio, params_.preprocessor_params.crop_height,
      params_.preprocessor_params.crop_width, stream, model_input_imgs.getCudaPtr());

  return Ret_Code;
}

}  // namespace preprocessor
}  // namespace sparse_end2end