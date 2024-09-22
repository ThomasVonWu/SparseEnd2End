// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "img_preprocess.h"

namespace SparseEnd2End {
namespace image {

Preprocess::Preprocess(const Params& params) : params_(params) {}

common::Status Preprocess::forward(const CudaWrapper<std::uint8_t>& raw_imgs,
                                   const cudaStream_t& stream,
                                   const CudaWrapper<float>& model_input_imgs) {
  if (raw_imgs.getSize() != params_.num_cams * (params_.raw_img_c * params_.raw_img_h * params_.raw_img_w)) {
    LOG_ERROR(kLogContext, "Input raw imgs' size mismatches with params.");

    return common::Status::kImgPreprocesSizeErr;
  }

  if (model_input_imgs.getSize() !=
      params_.num_cams * (params_.model_input_img_c * params_.model_input_img_h * params_.model_input_img_w)) {
    LOG_ERROR(kLogContext, "Model input imgs' size mismatches with params.");

    return common::Status::kImgPreprocesSizeErr;
  }

  const common::Status Ret_Code = imgPreprocessLauncher(
      raw_imgs.getPtr(), static_cast<std::uint32_t>(params_.num_cams), params_.raw_img_c, params_.raw_img_h,
      params_.raw_img_w, params_.model_input_img_h, params_.model_input_img_w, params_.img_aug.crop_h,
      params_.img_aug.crop_w, params_.img_aug.resize_ratio, stream, model_input_imgs.getPtr());

  return Ret_Code;
}

common::Status ImagePreprocessor::forward(const CudaWrapper<std::uint8_t>& raw_imgs,
                                          const cudaStream_t& stream,
                                          const CudaWrapper<half>& model_input_imgs) {
  if (raw_imgs.getSize() != params_.num_cams * (params_.raw_img_c * params_.raw_img_h * params_.raw_img_w)) {
    LOG_ERROR(kLogContext, "Input raw imgs' size mismatches with params.");

    return common::Status::kImgPreprocesSizeErr;
  }

  if (model_input_imgs.getSize() !=
      params_.num_cams * (params_.model_input_img_c * params_.model_input_img_h * params_.model_input_img_w)) {
    LOG_ERROR(kLogContext, "Model input imgs' size mismatches with params.");

    return common::Status::kImgPreprocesSizeErr;
  }

  common::Status Ret_Code = imgPreprocessLauncher(
      raw_imgs.getPtr(), static_cast<std::uint32_t>(params_.num_cams), params_.raw_img_c, params_.raw_img_h,
      params_.raw_img_w, params_.model_input_img_h, params_.model_input_img_w, params_.img_aug.crop_h,
      params_.img_aug.crop_w, params_.img_aug.resize_ratio, stream, model_input_imgs.getPtr());

  return Ret_Code;
}

}  // namespace image
}  // namespace SparseEnd2End