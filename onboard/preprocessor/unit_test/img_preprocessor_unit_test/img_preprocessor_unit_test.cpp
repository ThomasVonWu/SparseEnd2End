// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

#include "../../../common/cuda_wrapper.cu.h"
#include "../../../common/parameter.h"
#include "../../../common/utils.h"
#include "../../img_preprocessor.h"
#include "../../parameters_parser.h"

namespace sparse_end2end {
namespace preprocessor {

float GetMaxError(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    std::cout << "Vector size  mismatch error !";
  }

  float max_error = 0.0F;
  for (size_t i = 0; i < a.size(); ++i) {
    const float error = std::abs(a[i] - b[i]);
    if (max_error < error) {
      max_error = error;
    }
  }
  std::cout << "[MaxError] = " << max_error << std::endl;
  return max_error;
}

TEST(ImgPreprocessorUnitTest, ImgPreprocessorFP32) {
  std::filesystem::path current_dir = std::filesystem::current_path();
  const common::E2EParams params = parseParams(current_dir / "../../../assets/model_cfg.yaml");

  float time_cost = 0.0F;

  cudaEvent_t start, stop;
  cudaStream_t stream = nullptr;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  const std::vector<std::tuple<std::string, std::string>> test_samples{
      {"/path/to/PublishRepos/SparseEnd2End/script/tutorial/asset/sample_0_ori_imgs_1*6*3*900*1600_uint8.bin",
       "/path/to/PublishRepos/SparseEnd2End/script/tutorial/asset/sample_0_imgs_1*6*3*256*704_float32.bin"},
      {"/path/to/PublishRepos/SparseEnd2End/script/tutorial/asset/sample_1_ori_imgs_1*6*3*900*1600_uint8.bin",
       "/path/to/PublishRepos/SparseEnd2End/script/tutorial/asset/sample_1_imgs_1*6*3*256*704_float32.bin"},
      {"/path/to/PublishRepos/SparseEnd2End/script/tutorial/asset/sample_2_ori_imgs_1*6*3*900*1600_uint8.bin",
       "/path/to/PublishRepos/SparseEnd2End/script/tutorial/asset/sample_2_imgs_1*6*3*256*704_float32.bin"}};

  for (const auto& sample : test_samples) {
    std::shared_ptr<ImagePreprocessor> image_preprocessor_ptr = std::make_shared<ImagePreprocessor>(params);

    std::vector<uint8_t> ori_img = common::readfile_wrapper<std::uint8_t>(std::get<0>(sample));
    std::vector<float> expected_model_input_img = common::readfile_wrapper<float>(std::get<1>(sample));

    common::CudaWrapper<std::uint8_t> ori_img_cuda(ori_img);

    std::vector<float> model_input_img;
    model_input_img.resize(params.preprocessor_params.num_cams * params.preprocessor_params.model_input_img_c *
                           params.preprocessor_params.model_input_img_h * params.preprocessor_params.model_input_img_w);
    common::CudaWrapper<float> model_input_img_cuda(model_input_img);

    cudaEventRecord(start, stream);
    image_preprocessor_ptr->forward(ori_img_cuda, stream, model_input_img_cuda);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cost, start, stop);
    std::cout << "[Time Cost] : "
              << "Image Preprocessor (CUDA float32) time = " << time_cost << "[ms]." << std::endl;

    model_input_img = model_input_img_cuda.cudaMemcpyD2HResWrap();
    float max_error = 0.0F;
    max_error = GetMaxError(model_input_img, expected_model_input_img);
    EXPECT_LE(max_error, 0.018F);
    image_preprocessor_ptr.reset();
  }

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));
}
}  // namespace preprocessor
}  // namespace sparse_end2end
