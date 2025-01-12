// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

/*
unning main() from /home/thomasvowu/PublishRepos/SparseEnd2End/onboard/third_party/googletest/googletest/src/gtest_main.cc
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Sparse4dHeadFisrstFrameTrtInferUnitTest
[ RUN      ] Sparse4dHeadFisrstFrameTrtInferUnitTest.TrtInferConsistencyVerification
/home/thomasvowu/PublishRepos/SparseEnd2End/onboard/assets/trt_engine/sparse4dhead1st_polygraphy.engine
[Sparse4dTrtLog][I] Loaded engine size: 84 MiB
[Sparse4dTrtLog][W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[Sparse4dTrtLog][I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +1242, GPU +318, now: CPU 1433, GPU 4523 (MiB)
[Sparse4dTrtLog][I] [MemUsageChange] Init cuDNN: CPU +2, GPU +10, now: CPU 1435, GPU 4533 (MiB)
[Sparse4dTrtLog][W] TensorRT was linked against cuDNN 8.9.0 but loaded cuDNN 8.6.0
[Sparse4dTrtLog][I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +75, now: CPU 0, GPU 75 (MiB)
[Sparse4dTrtLog][I] [MS] Running engine with multi stream info
[Sparse4dTrtLog][I] [MS] Number of aux streams is 1
[Sparse4dTrtLog][I] [MS] Number of total worker streams is 2
[Sparse4dTrtLog][I] [MS] The main stream provided by execute/enqueue calls is the first worker stream
[Sparse4dTrtLog][I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 1435, GPU 4525 (MiB)
[Sparse4dTrtLog][I] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 1435, GPU 4533 (MiB)
[Sparse4dTrtLog][W] TensorRT was linked against cuDNN 8.9.0 but loaded cuDNN 8.6.0
[Sparse4dTrtLog][I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +101, now: CPU 0, GPU 176 (MiB)
[TensorRT Test] Sparse4d Head First Frame Inference (FP32) Time Costs = 28.7016 [ms]
Error >0.1 percentage is: 0
MaxError = 0.000582129
Pred_instance_feature: max=3.25359 min=-4.15375
Expd_instance_feature: max=3.25358 min=-4.15376

Error >0.1 percentage is: 0
MaxError = 0.000371933
Pred_anchor : max=55.3649 min=-54.0855
Expd_anchor: max=55.3649 min=-54.0855

Error >0.1 percentage is: 0
MaxError = 0.000299931
Pred_class_score: max=2.15206 min=-9.23959
Expd_class_score: max=2.15207 min=-9.23956

Error >0.1 percentage is: 0
MaxError = 0.000105441
Pred_quality_score: max=2.10492 min=-2.64644
Expd_quality_score: max=2.10492 min=-2.64644

[       OK ] Sparse4dHeadFisrstFrameTrtInferUnitTest.TrtInferConsistencyVerification (2539 ms)
[----------] 1 test from Sparse4dHeadFisrstFrameTrtInferUnitTest (2539 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (2539 ms total)
[  PASSED  ] 1 test.
*/

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

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
  std::cout << "Error >" << threshold << " percentage = " << p << std::endl;
  std::cout << "MaxError = " << max_error << std::endl;

  return p;
}

TEST(Sparse4dHeadFisrstFrameTrtInferUnitTest, TrtInferConsistencyVerification) {
  std::filesystem::path current_dir = std::filesystem::current_path();
  const common::E2EParams params = preprocessor::parseParams(current_dir / "../../assets/model_cfg.yaml");

  std::string sparse4d_head1st_engine_path = params.sparse4d_head1st_engine.engine_path;
  std::string multiview_multiscale_deformable_attention_aggregation_path =
      params.model_cfg.multiview_multiscale_deformable_attention_aggregation_path;
  std::vector<std::string> sparse4d_head1st_engine_input_names = params.sparse4d_head1st_engine.input_names;
  std::vector<std::string> sparse4d_head1st_engine_output_names = params.sparse4d_head1st_engine.output_names;

  std::vector<std::uint32_t> sparse4d_extract_feat_shape_lc = params.model_cfg.sparse4d_extract_feat_shape_lc;

  cudaEvent_t start, stop;
  cudaStream_t stream = nullptr;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  std::cout << sparse4d_head1st_engine_path << std::endl;
  std::shared_ptr<TensorRT> trt_engine = std::make_shared<TensorRT>(
      sparse4d_head1st_engine_path, multiview_multiscale_deformable_attention_aggregation_path,
      sparse4d_head1st_engine_input_names, sparse4d_head1st_engine_output_names);

  std::tuple<std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string,
             std::string, std::string, std::string, std::string>
      test_sample{
          "/path/to/sample_0_feature_1*89760*256_float32.bin",
          "/path/to/sample_0_spatial_shapes_6*4*2_int32.bin",
          "/path/to/sample_0_level_start_index_6*4_int32.bin",
          "/path/to/sample_0_instance_feature_1*900*256_float32.bin",
          "/path/to/sample_0_anchor_1*900*11_float32.bin",
          "/path/to/sample_0_time_interval_1_float32.bin",
          "/path/to/sample_0_image_wh_1*6*2_float32.bin",
          "/path/to/sample_0_lidar2img_1*6*4*4_float32.bin",
          "/path/to/sample_0_pred_instance_feature_1*900*256_float32.bin",
          "/path/to/sample_0_pred_anchor_1*900*11_float32.bin",
          "/path/to/sample_0_pred_class_score_1*900*10_float32.bin",
          "/path/to/sample_0_pred_quality_score_1*900*2_float32.bin"};

  const auto feature = common::readfile_wrapper<float>(std::get<0>(test_sample));
  const auto spatial_shapes = common::readfile_wrapper<int32_t>(std::get<1>(test_sample));
  const auto level_start_index = common::readfile_wrapper<int32_t>(std::get<2>(test_sample));
  const auto instance_feature = common::readfile_wrapper<float>(std::get<3>(test_sample));
  const auto anchor = common::readfile_wrapper<float>(std::get<4>(test_sample));
  const auto time_interval = common::readfile_wrapper<float>(std::get<5>(test_sample));
  const auto image_wh = common::readfile_wrapper<float>(std::get<6>(test_sample));
  const auto lidar2img = common::readfile_wrapper<float>(std::get<7>(test_sample));
  const auto expected_pred_instance_feature = common::readfile_wrapper<float>(std::get<8>(test_sample));
  const auto expected_pred_anchor = common::readfile_wrapper<float>(std::get<9>(test_sample));
  const auto expected_pred_class_score = common::readfile_wrapper<float>(std::get<10>(test_sample));
  const auto expected_pred_quality_score = common::readfile_wrapper<float>(std::get<11>(test_sample));

  EXPECT_EQ(feature.size(), 1 * 89760 * 256);
  EXPECT_EQ(spatial_shapes.size(), 6 * 4 * 2);
  EXPECT_EQ(level_start_index.size(), 6 * 4);
  EXPECT_EQ(instance_feature.size(), 1 * 900 * 256);
  EXPECT_EQ(anchor.size(), 1 * 900 * 11);
  EXPECT_EQ(time_interval.size(), 1);
  EXPECT_EQ(image_wh.size(), 1 * 6 * 2);
  EXPECT_EQ(lidar2img.size(), 1 * 6 * 4 * 4);
  EXPECT_EQ(expected_pred_instance_feature.size(), 1 * 900 * 256);
  EXPECT_EQ(expected_pred_anchor.size(), 1 * 900 * 11);
  EXPECT_EQ(expected_pred_class_score.size(), 1 * 900 * 10);
  EXPECT_EQ(expected_pred_quality_score.size(), 1 * 900 * 2);

  const common::CudaWrapper<float> warmup_feature_gpu(feature);
  const common::CudaWrapper<int32_t> warmup_spatial_shapes_gpu(spatial_shapes);
  const common::CudaWrapper<int32_t> warmup_level_start_index_gpu(level_start_index);
  const common::CudaWrapper<float> warmup_instance_feature_gpu(instance_feature);
  const common::CudaWrapper<float> warmup_anchor_gpu(anchor);
  const common::CudaWrapper<float> warmup_time_interval_gpu(time_interval);
  const common::CudaWrapper<float> warmup_image_wh_gpu(image_wh);
  const common::CudaWrapper<float> warmup_lidar2img_gpu(lidar2img);
  common::CudaWrapper<float> warmup_tmp_outs0(1 * 900 * 256);
  common::CudaWrapper<float> warmup_tmp_outs1(1 * 900 * 256);
  common::CudaWrapper<float> warmup_tmp_outs2(1 * 900 * 256);
  common::CudaWrapper<float> warmup_tmp_outs3(1 * 900 * 256);
  common::CudaWrapper<float> warmup_tmp_outs4(1 * 900 * 256);
  common::CudaWrapper<float> warmup_tmp_outs5(1 * 900 * 256);
  common::CudaWrapper<float> warmup_pred_instance_feature_gpu(1 * 900 * 256);
  common::CudaWrapper<float> warmup_pred_anchor_gpu(1 * 900 * 11);
  common::CudaWrapper<float> warmup_pred_class_score_gpu(1 * 900 * 10);
  common::CudaWrapper<float> warmup_pred_quality_score_gpu(1 * 900 * 2);

  std::vector<void*> warmup_buffers;
  warmup_buffers.push_back(warmup_feature_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_spatial_shapes_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_level_start_index_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_instance_feature_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_anchor_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_time_interval_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_image_wh_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_lidar2img_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs0.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs1.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs2.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs3.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs4.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs5.getCudaPtr());
  warmup_buffers.push_back(warmup_pred_instance_feature_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_pred_anchor_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_pred_class_score_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_pred_quality_score_gpu.getCudaPtr());

  // trt_engine->getEngineInfo();

  // // Warmup
  for (int i = 0; i < 5; ++i) {
    if (trt_engine->infer(warmup_buffers.data(), stream) != true) {
      std::cout << "[ERROR] TensorRT engine inference failed during warmup" << std::endl;
    }
    cudaStreamSynchronize(stream);
  }

  const common::CudaWrapper<float> feature_gpu(feature);
  const common::CudaWrapper<int32_t> spatial_shapes_gpu(spatial_shapes);
  const common::CudaWrapper<int32_t> level_start_index_gpu(level_start_index);
  const common::CudaWrapper<float> instance_feature_gpu(instance_feature);
  const common::CudaWrapper<float> anchor_gpu(anchor);
  const common::CudaWrapper<float> time_interval_gpu(time_interval);
  const common::CudaWrapper<float> image_wh_gpu(image_wh);
  const common::CudaWrapper<float> lidar2img_gpu(lidar2img);
  common::CudaWrapper<float> tmp_outs0(1 * 900 * 256);
  common::CudaWrapper<float> tmp_outs1(1 * 900 * 256);
  common::CudaWrapper<float> tmp_outs2(1 * 900 * 256);
  common::CudaWrapper<float> tmp_outs3(1 * 900 * 256);
  common::CudaWrapper<float> tmp_outs4(1 * 900 * 256);
  common::CudaWrapper<float> tmp_outs5(1 * 900 * 256);
  common::CudaWrapper<float> pred_instance_feature_gpu(1 * 900 * 256);
  common::CudaWrapper<float> pred_anchor_gpu(1 * 900 * 11);
  common::CudaWrapper<float> pred_class_score_gpu(1 * 900 * 10);
  common::CudaWrapper<float> pred_quality_score_gpu(1 * 900 * 2);

  std::vector<void*> buffers;
  buffers.push_back(feature_gpu.getCudaPtr());
  buffers.push_back(spatial_shapes_gpu.getCudaPtr());
  buffers.push_back(level_start_index_gpu.getCudaPtr());
  buffers.push_back(instance_feature_gpu.getCudaPtr());
  buffers.push_back(anchor_gpu.getCudaPtr());
  buffers.push_back(time_interval_gpu.getCudaPtr());
  buffers.push_back(image_wh_gpu.getCudaPtr());
  buffers.push_back(lidar2img_gpu.getCudaPtr());
  buffers.push_back(tmp_outs0.getCudaPtr());
  buffers.push_back(tmp_outs1.getCudaPtr());
  buffers.push_back(tmp_outs2.getCudaPtr());
  buffers.push_back(tmp_outs3.getCudaPtr());
  buffers.push_back(tmp_outs4.getCudaPtr());
  buffers.push_back(tmp_outs5.getCudaPtr());
  buffers.push_back(pred_instance_feature_gpu.getCudaPtr());
  buffers.push_back(pred_anchor_gpu.getCudaPtr());
  buffers.push_back(pred_class_score_gpu.getCudaPtr());
  buffers.push_back(pred_quality_score_gpu.getCudaPtr());

  float time_cost = 0.0f;
  checkCudaErrors(cudaEventRecord(start, stream));
  if (!trt_engine->infer(buffers.data(), stream)) {
    std::cout << "[ERROR] TensorRT engine inference failed " << std::endl;
  }
  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&time_cost, start, stop));
  std::cout << "[TensorRT Test] Sparse4d Head First Frame Inference (FP32) Time Costs = " << time_cost << " [ms]"
            << std::endl;
  cudaStreamSynchronize(stream);

  auto pred_instance_feature = pred_instance_feature_gpu.cudaMemcpyD2HResWrap();
  auto pred_anchor = pred_anchor_gpu.cudaMemcpyD2HResWrap();
  auto pred_class_score = pred_class_score_gpu.cudaMemcpyD2HResWrap();
  auto pred_quality_score = pred_quality_score_gpu.cudaMemcpyD2HResWrap();

  const float p0 = GetErrorPercentage(pred_instance_feature, expected_pred_instance_feature, 0.1);
  EXPECT_LE(p0, 0.02F);
  std::cout << "Pred_instance_feature: max="
            << *std::max_element(pred_instance_feature.begin(), pred_instance_feature.end())
            << " min=" << *std::min_element(pred_instance_feature.begin(), pred_instance_feature.end()) << std::endl;
  std::cout << "Expd_instance_feature: max="
            << *std::max_element(expected_pred_instance_feature.begin(), expected_pred_instance_feature.end())
            << " min="
            << *std::min_element(expected_pred_instance_feature.begin(), expected_pred_instance_feature.end())
            << std::endl
            << std::endl;

  const float p1 = GetErrorPercentage(pred_anchor, expected_pred_anchor, 0.1);
  EXPECT_LE(p1, 0.02F);
  std::cout << "Pred_anchor : max=" << *std::max_element(pred_anchor.begin(), pred_anchor.end())
            << " min=" << *std::min_element(pred_anchor.begin(), pred_anchor.end()) << std::endl;
  std::cout << "Expd_anchor: max=" << *std::max_element(expected_pred_anchor.begin(), expected_pred_anchor.end())
            << " min=" << *std::min_element(expected_pred_anchor.begin(), expected_pred_anchor.end()) << std::endl
            << std::endl;

  const float p2 = GetErrorPercentage(pred_class_score, expected_pred_class_score, 0.1);
  EXPECT_LE(p2, 0.01F);
  std::cout << "Pred_class_score: max=" << *std::max_element(pred_class_score.begin(), pred_class_score.end())
            << " min=" << *std::min_element(pred_class_score.begin(), pred_class_score.end()) << std::endl;
  std::cout << "Expd_class_score: max="
            << *std::max_element(expected_pred_class_score.begin(), expected_pred_class_score.end())
            << " min=" << *std::min_element(expected_pred_class_score.begin(), expected_pred_class_score.end())
            << std::endl
            << std::endl;

  const float p3 = GetErrorPercentage(pred_quality_score, expected_pred_quality_score, 0.1);
  EXPECT_LE(p3, 0.01F);
  std::cout << "Pred_quality_score: max=" << *std::max_element(pred_quality_score.begin(), pred_quality_score.end())
            << " min=" << *std::min_element(pred_quality_score.begin(), pred_quality_score.end()) << std::endl;
  std::cout << "Expd_quality_score: max="
            << *std::max_element(expected_pred_quality_score.begin(), expected_pred_quality_score.end())
            << " min=" << *std::min_element(expected_pred_quality_score.begin(), expected_pred_quality_score.end())
            << std::endl
            << std::endl;

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));
}

}  // namespace engine
}  // namespace sparse_end2end