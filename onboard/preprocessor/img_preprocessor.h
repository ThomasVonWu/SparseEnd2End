// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_PREPROCESSOR_IMG_PREPROCESSOR_H
#define ONBOARD_PREPROCESSOR_IMG_PREPROCESSOR_H
#include <cuda_fp16.h>

#include <cstdint>

#include "../common/common.h"
#include "../common/cuda_wrapper.cu.h"
#include "../common/parameter.h"

namespace sparse_end2end {
namespace preprocessor {

class ImagePreprocessor {
 public:
  explicit ImagePreprocessor(const common::E2EParams& params);
  ImagePreprocessor() = delete;
  ~ImagePreprocessor() = default;

  common::Status forward(const common::CudaWrapper<std::uint8_t>& raw_imgs,
                         const cudaStream_t& stream,
                         const common::CudaWrapper<float>& model_input_imgs);

  common::Status forward(const common::CudaWrapper<std::uint8_t>& raw_imgs,
                         const cudaStream_t& stream,
                         const common::CudaWrapper<half>& model_input_imgs);

 private:
  const common::E2EParams params_;
};

}  // namespace preprocessor
}  // namespace sparse_end2end

#endif  // ONBOARD_PREPROCESSOR_IMG_PREPROCESSOR_H
