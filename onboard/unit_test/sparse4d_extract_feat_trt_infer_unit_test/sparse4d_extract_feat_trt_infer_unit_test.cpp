// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

/*
Running main() from /home/wxc/PublishRepos/SparseEnd2End/onboard/third_party/googletest/googletest/src/gtest_main.cc
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Sparse4dExtractFeatTrtInferUnitTest
[ RUN      ] Sparse4dExtractFeatTrtInferUnitTest.TrtInferConsistencyVerification
[Sparse4dTrtLog][I] Loaded engine size: 135 MiB
[Sparse4dTrtLog][W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[Sparse4dTrtLog][I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +132, now: CPU 0, GPU 132 (MiB)
[Sparse4dTrtLog][I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +318, now: CPU 0, GPU 450 (MiB)
[TensorRT Test] Sparse4d Feature Extraction Module TensorRT Inference (FP32) Time Costs = 13.271 [ms]
Error >0.1 percentage is: 0.0107776
MaxError = 1.3429
pred_feature: max=1684.9 min=-1741.75
expd_feature: max=1685.18 min=-1741.97

[TensorRT Test] Sparse4d Feature Extraction Module TensorRT Inference (FP32) Time Costs = 13.3939 [ms]
Error >0.1 percentage is: 0.0111983
MaxError = 1.38782
pred_feature: max=1685.58 min=-1760.41
expd_feature: max=1685.72 min=-1760.61

[TensorRT Test] Sparse4d Feature Extraction Module TensorRT Inference (FP32) Time Costs = 12.4273 [ms]
Error >0.1 percentage is: 0.0116563
MaxError = 1.37061
pred_feature: max=1670.9 min=-1762.13
expd_feature: max=1671.33 min=-1762.24

[       OK ] Sparse4dExtractFeatTrtInferUnitTest.TrtInferConsistencyVerification (9575 ms)
[----------] 1 test from Sparse4dExtractFeatTrtInferUnitTest (9575 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (9575 ms total)
[  PASSED  ] 1 test.
*/

#include <cuda_runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>

#include "../../common/cuda_wrapper.cu.h"
#include "../../common/utils.h"
#include "../../preprocessor/parameters_parser.h"
#include "../../tensorrt/tensorrt.h"

namespace sparse_end2end {
namespace engine {

float GetErrorPercentage(const std::vector<float>& a, const std::vector<float>& b, float threshold) {
  float max_error = 0.0F;
  if (a.size() != b.size()) {
    max_error = std::numeric_limits<float>::max();
  }

  std::vector<float> cache_errors;
  for (size_t i = 0; i < a.size(); ++i) {
    const float error = std::abs(a[i] - b[i]);
    cache_errors.push_back(error);
    if (max_error < error) {
      max_error = error;
    }
  }

  std::sort(cache_errors.begin(), cache_errors.end(), [](int a, int b) { return a > b; });

  std::vector<float> cache_roi_erros;
  for (auto x : cache_errors) {
    if (x > threshold) {
      cache_roi_erros.push_back(x);
    }
  }

  float p = float(cache_roi_erros.size()) / float(cache_errors.size());
  std::cout << "Error >" << threshold << " percentage is: " << p << std::endl;
  std::cout << "MaxError = " << max_error << std::endl;

  return p;
}

TEST(Sparse4dExtractFeatTrtInferUnitTest, TrtInferConsistencyVerification) {
  std::filesystem::path current_dir = std::filesystem::current_path();
  const common::E2EParams params = preprocessor::parseParams(current_dir / "../../assets/model_cfg.yaml");

  std::string sparse4d_extract_feat_engine_path = params.sparse4d_extract_feat_engine.engine_path;
  std::vector<std::string> sparse4d_extract_feat_engine_input_names = params.sparse4d_extract_feat_engine.input_names;
  std::vector<std::string> sparse4d_extract_feat_engine_output_names = params.sparse4d_extract_feat_engine.output_names;

  std::vector<std::uint32_t> sparse4d_extract_feat_shape_lc = params.model_cfg.sparse4d_extract_feat_shape_lc;

  cudaEvent_t start, stop;
  cudaStream_t stream = nullptr;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  std::shared_ptr<TensorRT> trt_engine =
      std::make_shared<TensorRT>(sparse4d_extract_feat_engine_path, "", sparse4d_extract_feat_engine_input_names,
                                 sparse4d_extract_feat_engine_output_names);

  const std::vector<std::tuple<std::string, std::string>> test_samples{
      {"/path/to/sample_0_imgs_1*6*3*256*704_float32.bin",
       "/path/to/sample_0_feature_1*89760*256_float32.bin"},
      {"/path/to/sample_1_imgs_1*6*3*256*704_float32.bin",
       "/path/to/sample_1_feature_1*89760*256_float32.bin"},
      {"/path/to/sample_2_imgs_1*6*3*256*704_float32.bin",
       "/path/to/sample_2_feature_1*89760*256_float32.bin"}};

  // Warmup
  for (const auto& sample : test_samples) {
    const std::vector<float> imgs = common::readfile_wrapper<float>(std::get<0>(sample));
    const std::vector<float> expected_pred_feature = common::readfile_wrapper<float>(std::get<1>(sample));

    EXPECT_EQ(imgs.size(), 1 * 6 * 3 * 256 * 704);
    EXPECT_EQ(expected_pred_feature.size(), 1 * 89760 * 256);

    const common::CudaWrapper<float> imgs_gpu(imgs);
    common::CudaWrapper<float> pred_feature_gpu(1 * 89760 * 256);

    std::vector<void*> buffers;
    buffers.push_back(imgs_gpu.getCudaPtr());
    buffers.push_back(pred_feature_gpu.getCudaPtr());

    if (!trt_engine->infer(buffers.data(), stream)) {
      std::cout << "[ERROR] TensorRT engine inference failed " << std::endl;
    }
    cudaStreamSynchronize(stream);
  }

  // Start
  for (const auto& sample : test_samples) {
    const std::vector<float> imgs = common::readfile_wrapper<float>(std::get<0>(sample));
    std::vector<float> expected_pred_feature = common::readfile_wrapper<float>(std::get<1>(sample));

    EXPECT_EQ(imgs.size(), 1 * 6 * 3 * 256 * 704);
    EXPECT_EQ(expected_pred_feature.size(), 1 * 89760 * 256);

    const common::CudaWrapper<float> imgs_gpu(imgs);
    common::CudaWrapper<float> pred_feature_gpu(1 * 89760 * 256);

    std::vector<void*> buffers;
    buffers.push_back(imgs_gpu.getCudaPtr());
    buffers.push_back(pred_feature_gpu.getCudaPtr());

    checkCudaErrors(cudaEventRecord(start, stream));
    if (!trt_engine->infer(buffers.data(), stream)) {
      std::cout << "[ERROR] TensorRT engine inference failed " << std::endl;
    }
    checkCudaErrors(cudaEventRecord(stop, stream));
    checkCudaErrors(cudaEventSynchronize(stop));
    float time_cost = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&time_cost, start, stop));
    std::cout << "[TensorRT Test] Sparse4d Feature Extraction Module TensorRT Inference (FP32) Time Costs = "
              << time_cost << " [ms]" << std::endl;
    cudaStreamSynchronize(stream);

    const auto& pred_feature = pred_feature_gpu.cudaMemcpyD2HResWrap();

    const std::uint32_t p0 = GetErrorPercentage(pred_feature, expected_pred_feature, 0.1);
    EXPECT_LE(p0, 0.01F);
    std::cout << "Pred_feature: max=" << *std::max_element(pred_feature.begin(), pred_feature.end())
              << " min=" << *std::min_element(pred_feature.begin(), pred_feature.end()) << std::endl;
    std::cout << "Expd_feature: max=" << *std::max_element(expected_pred_feature.begin(), expected_pred_feature.end())
              << " min=" << *std::min_element(expected_pred_feature.begin(), expected_pred_feature.end()) << std::endl
              << std::endl;
  }

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));
}

}  // namespace engine
}  // namespace sparse_end2end