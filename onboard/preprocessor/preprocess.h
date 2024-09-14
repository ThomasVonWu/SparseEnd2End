// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_PREPROCESSOR_PREPROCESS_H
#define ONBOARD_PREPROCESSOR_PREPROCESS_H

#include <cuda_fp16.h>

namespace SparseEnd2End {
namespace image {

class Preprocess {
 public:
  explicit ImagePreprocess(const Params& params);
  ImagePreprocess() = delete;
  ~ImagePreprocess() = default;

  Status forward(const CudaWrapper<std::uint8_t>& raw_imgs,
                 const cudaStream_t& stream,
                 const CudaWrapper<float>& model_input_imgs);

  Status forward(const CudaWrapper<std::uint8_t>& raw_imgs,
                 const cudaStream_t& stream,
                 const CudaWrapper<half>& model_input_imgs);

 private:
  const Params params_;
};

}  // namespace image
}  // namespace SparseEnd2End

#endif  // ONBOARD_PREPROCESSOR_PREPROCESS_H
