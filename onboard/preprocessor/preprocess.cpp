// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "preprocess.h"

namespace SparseEnd2End {
namespace image {

Preprocess::Preprocess(const Params& params) : params_(params) {}

Status Preprocess::forward(const CudaWrapper<std::uint8_t>& raw_imgs,
                           const cudaStream_t& stream,
                           const CudaWrapper<float>& model_input_imgs) {
  if (raw_imgs.getSize() !=
      params_.cam_nums *
          (params_.raw_img_c * params_.raw_img_h * params_.raw_img_w)) {
    LOG_ERROR(kLogContext, "Input raw imgs' size mismatches with params.");

    return Status::kImgPreprocesSizeErr;
  }

  if (model_input_imgs.getSize() !=
      params_.cam_nums *
          (params_.model_input_img_c * params_.model_input_img_h *
           params_.model_input_img_w)) {
    LOG_ERROR(kLogContext, "Model input imgs' size mismatches with params.");

    return Status::kImgPreprocesSizeErr;
  }

  const Status Ret_Code = ImgPreprocessorLauncher(
      raw_imgs.getPtr(), static_cast<std::uint32_t>(params_.cam_nums),
      params_.raw_img_c, params_.raw_img_h, params_.raw_img_w,
      params_.model_input_img_h, params_.model_input_img_w,
      params_.img_aug.crop_h, params_.img_aug.crop_w,
      params_.img_aug.resize_ratio, stream, model_input_imgs.getPtr());

  return Ret_Code;
}

Status ImagePreprocessor::forward(const CudaWrapper<std::uint8_t>& raw_imgs,
                                  const cudaStream_t& stream,
                                  const CudaWrapper<half>& model_input_imgs) {
  if (raw_imgs.getSize() !=
      params_.cam_nums *
          (params_.raw_img_c * params_.raw_img_h * params_.raw_img_w)) {
    LOG_ERROR(kLogContext, "Input raw imgs' size mismatches with params.");

    return Status::kImgPreprocesSizeErr;
  }

  if (model_input_imgs.getSize() !=
      params_.cam_nums *
          (params_.model_input_img_c * params_.model_input_img_h *
           params_.model_input_img_w)) {
    LOG_ERROR(kLogContext, "Model input imgs' size mismatches with params.");

    return Status::kImgPreprocesSizeErr;
  }

  Status Ret_Code = ImgPreprocessorLauncher(
      raw_imgs.getPtr(), static_cast<std::uint32_t>(params_.cam_nums),
      params_.raw_img_c, params_.raw_img_h, params_.raw_img_w,
      params_.model_input_img_h, params_.model_input_img_w,
      params_.img_aug.crop_h, params_.img_aug.crop_w,
      params_.img_aug.resize_ratio, stream, model_input_imgs.getPtr());

  return Ret_Code;
}

}  // namespace image
}  // namespace SparseEnd2End