// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <time.h>

#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "../../common/common.h"
#include "../../common/parameter.h"
#include "../../common/utils.h"
#include "../../instance_bank.h"
#include "../../preprocessor/parameters_parser.h"

namespace sparse_end2end {
namespace head {

template <typename T>
T GetMaxError(const std::vector<T>& a, const std::vector<T>& b, const std::string& name) {
  if (a.size() != b.size()) {
    return std::numeric_limits<T>::max();
  }

  T max_error = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const T error = std::abs(a[i] - b[i]);
    if (max_error < error) {
      max_error = error;
    }
  }
  std::cout << name << " [MaxError] = " << max_error << std::endl;
  return max_error;
}

TEST(InstanceBankUnitTest, InstanceBankCpuImplInputOutputConsistencyVerification) {
  std::filesystem::path current_dir = std::filesystem::current_path();
  const common::E2EParams params = preprocessor::parseParams(current_dir / "../../../assets/model_cfg.yaml");

  InstanceBank ibank(params);

  /// 1st Frame `expected_temp_instance_feature` and `expected_temp_anchor` is emprty, `expected_mask` = 0,
  /// `pred_track_id` from head output  is empty.
  const std::vector<
      std::tuple<std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string,
                 std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string,
                 std::string, std::string, std::string>>
      test_samples{
          {"/path/to/sample_0_ibank_timestamp_1_float64.bin", 
          "/path/to/sample_0_ibank_global2lidar_4*4_float64.bin",
           "/path/to/sample_0_instance_feature_1*900*256_float32.bin", 
           "/path/to/sample_0_anchor_1*900*11_float32.bin",
           "/path/to/sample_1_temp_instance_feature_1*600*256_float32.bin",
           "/path/to/sample_1_temp_anchor_1*600*11_float32.bin", 
           "/path/to/sample_0_time_interval_1_float32.bin",
           "/path/to/sample_1_mask_1_int32.bin", 
           "/path/to/sample_0_pred_instance_feature_1*900*256_float32.bin",
           "/path/to/sample_0_pred_anchor_1*900*11_float32.bin",
           "/path/to/sample_0_pred_class_score_1*900*10_float32.bin",
           "/path/to/sample_0_ibank_temp_confidence_1*900_float32.bin",
           "/path/to/sample_0_ibank_confidence_1*600_float32.bin",
           "/path/to/sample_0_ibank_cached_feature_1*600*256_float32.bin",
           "/path/to/sample_0_ibank_cached_anchor_1*600*11_float32.bin",
           "/path/to/sample_1_pred_track_id_1*900_int32.bin",
           "/path/to/sample_0_ibank_updated_cur_track_id_1*900_int32.bin",
           "/path/to/sample_0_ibank_updated_temp_track_id_1*900_int32.bin",
           "/path/to/sample_0_ibank_prev_id_1_int32.bin"},
          {"/path/to/sample_1_ibank_timestamp_1_float64.bin", 
          "/path/to/sample_1_ibank_global2lidar_4*4_float64.bin",
           "/path/to/sample_1_instance_feature_1*900*256_float32.bin", 
           "/path/to/sample_1_anchor_1*900*11_float32.bin",
           "/path/to/sample_1_temp_instance_feature_1*600*256_float32.bin",
           "/path/to/sample_1_temp_anchor_1*600*11_float32.bin", 
           "/path/to/sample_1_time_interval_1_float32.bin",
           "/path/to/sample_1_mask_1_int32.bin", 
           "/path/to/sample_1_pred_instance_feature_1*900*256_float32.bin",
           "/path/to/sample_1_pred_anchor_1*900*11_float32.bin",
           "/path/to/sample_1_pred_class_score_1*900*10_float32.bin",
           "/path/to/sample_1_ibank_temp_confidence_1*900_float32.bin",
           "/path/to/sample_1_ibank_confidence_1*600_float32.bin",
           "/path/to/sample_1_ibank_cached_feature_1*600*256_float32.bin",
           "/path/to/sample_1_ibank_cached_anchor_1*600*11_float32.bin",
           "/path/to/sample_1_pred_track_id_1*900_int32.bin",
           "/path/to/sample_1_ibank_updated_cur_track_id_1*900_int32.bin",
           "/path/to/sample_1_ibank_updated_temp_track_id_1*900_int32.bin",
           "/path/to/sample_1_ibank_prev_id_1_int32.bin"},
          {"/path/to/sample_2_ibank_timestamp_1_float64.bin", 
          "/path/to/sample_2_ibank_global2lidar_4*4_float64.bin",
           "/path/to/sample_2_instance_feature_1*900*256_float32.bin", 
           "/path/to/sample_2_anchor_1*900*11_float32.bin",
           "/path/to/sample_2_temp_instance_feature_1*600*256_float32.bin",
           "/path/to/sample_2_temp_anchor_1*600*11_float32.bin", 
           "/path/to/sample_2_time_interval_1_float32.bin",
           "/path/to/sample_2_mask_1_int32.bin", 
           "/path/to/sample_2_pred_instance_feature_1*900*256_float32.bin",
           "/path/to/sample_2_pred_anchor_1*900*11_float32.bin",
           "/path/to/sample_2_pred_class_score_1*900*10_float32.bin",
           "/path/to/sample_2_ibank_temp_confidence_1*900_float32.bin",
           "/path/to/sample_2_ibank_confidence_1*600_float32.bin",
           "/path/to/sample_2_ibank_cached_feature_1*600*256_float32.bin",
           "/path/to/sample_2_ibank_cached_anchor_1*600*11_float32.bin",
           "/path/to/sample_2_pred_track_id_1*900_int32.bin",
           "/path/to/sample_2_ibank_updated_cur_track_id_1*900_int32.bin",
           "/path/to/sample_2_ibank_updated_temp_track_id_1*900_int32.bin",
           "/path/to/sample_2_ibank_prev_id_1_int32.bin"}};
  for (size_t i = 0; i < test_samples.size(); ++i) {
    std::vector<double> timestamp = common::readfile_wrapper<double>(std::get<0>(test_samples[i]));
    std::vector<double> global2lidar = common::readfile_wrapper<double>(std::get<1>(test_samples[i]));
    std::vector<float> expected_instance_feature = common::readfile_wrapper<float>(std::get<2>(test_samples[i]));
    std::vector<float> expected_anchor = common::readfile_wrapper<float>(std::get<3>(test_samples[i]));
    std::vector<float> expected_temp_instance_feature =
        common::readfile_wrapper<float>(std::get<4>(test_samples[i]));  // 1st Frame placehold
    std::vector<float> expected_temp_anchor =
        common::readfile_wrapper<float>(std::get<5>(test_samples[i]));  // 1st Frame placehold
    std::vector<float> expected_time_interval = common::readfile_wrapper<float>(std::get<6>(test_samples[i]));
    std::vector<std::int32_t> expected_mask =
        common::readfile_wrapper<std::int32_t>(std::get<7>(test_samples[i]));  // 1st Frame placehold

    std::vector<float> pred_instance_feature = common::readfile_wrapper<float>(std::get<8>(test_samples[i]));
    std::vector<float> pred_anchor = common::readfile_wrapper<float>(std::get<9>(test_samples[i]));
    std::vector<float> pred_class_score = common::readfile_wrapper<float>(std::get<10>(test_samples[i]));
    std::vector<float> expected_temp_confidence = common::readfile_wrapper<float>(std::get<11>(test_samples[i]));
    std::vector<float> expected_confidence = common::readfile_wrapper<float>(std::get<12>(test_samples[i]));
    std::vector<float> expected_cached_feature = common::readfile_wrapper<float>(std::get<13>(test_samples[i]));
    std::vector<float> expected_cached_anchor = common::readfile_wrapper<float>(std::get<14>(test_samples[i]));

    std::vector<std::int32_t> pred_track_id =
        common::readfile_wrapper<std::int32_t>(std::get<15>(test_samples[i]));  // 1st Frame placehold
    std::vector<std::int32_t> expected_updated_cur_track_id =
        common::readfile_wrapper<std::int32_t>(std::get<16>(test_samples[i]));
    std::vector<std::int32_t> expected_updated_temp_track_id =
        common::readfile_wrapper<std::int32_t>(std::get<17>(test_samples[i]));
    std::vector<std::int32_t> expected_prev_id = common::readfile_wrapper<std::int32_t>(std::get<18>(test_samples[i]));

    EXPECT_EQ(timestamp.size(), 1);
    EXPECT_EQ(global2lidar.size(), 4 * 4);
    EXPECT_EQ(expected_instance_feature.size(), 1 * 900 * 256);
    EXPECT_EQ(expected_anchor.size(), 1 * 900 * 11);
    EXPECT_EQ(expected_temp_instance_feature.size(), 1 * 600 * 256);
    EXPECT_EQ(expected_temp_anchor.size(), 1 * 600 * 11);
    EXPECT_EQ(expected_time_interval.size(), 1);
    EXPECT_EQ(expected_mask.size(), 1);
    EXPECT_EQ(pred_instance_feature.size(), 1 * 900 * 256);
    EXPECT_EQ(pred_anchor.size(), 1 * 900 * 11);
    EXPECT_EQ(pred_class_score.size(), 1 * 900 * 10);
    EXPECT_EQ(expected_temp_confidence.size(), 1 * 900);
    EXPECT_EQ(expected_confidence.size(), 1 * 600);
    EXPECT_EQ(expected_cached_feature.size(), 1 * 600 * 256);
    EXPECT_EQ(expected_cached_anchor.size(), 1 * 600 * 11);
    EXPECT_EQ(pred_track_id.size(), 900);
    EXPECT_EQ(expected_updated_cur_track_id.size(), 900);
    EXPECT_EQ(expected_updated_temp_track_id.size(), 900);
    EXPECT_EQ(expected_prev_id.size(), 1);

    std::cout << "\n[INFO] Instance bank error and time costs statistic, frameid = " << i + 1 << std::endl;

    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> global2lidar_mat(global2lidar.data());
    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    auto [ibank_got_instance_feature, ibank_got_kmeans_anchor, ibank_got_cached_feature, ibank_got_cached_anchor,
          ibank_got_time_interval, ibank_got_mask, ibank_got_trackid] = ibank.get(timestamp[0], global2lidar_mat);
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    float time_cost_in_ms = duration.count() * 1000.0F;
    std::cout << "[CPU Test] Instance Bank Get() Time Cost = " << time_cost_in_ms << " [ms]" << std::endl;

    float error1 = 0.0F, error2 = 0.0F, error3 = 0.0F, error4 = 0.0F;
    error1 = GetMaxError(ibank_got_instance_feature, expected_instance_feature, "ibank_got_instance_feature");
    error2 = GetMaxError(ibank_got_kmeans_anchor, expected_anchor, "ibank_got_kmeans_anchor");
    EXPECT_LE(error1, 1e-5);
    EXPECT_LE(error2, 1e-5);
    EXPECT_EQ(ibank_got_time_interval, expected_time_interval[0]);

    if (i == 0) {
      EXPECT_EQ(ibank_got_cached_feature.size(), 0);
      EXPECT_EQ(ibank_got_cached_anchor.size(), 0);
      EXPECT_EQ(ibank_got_mask, 0);
    } else {
      error3 = GetMaxError(ibank_got_cached_feature, expected_temp_instance_feature, "ibank_got_cached_feature");
      error4 = GetMaxError(ibank_got_cached_anchor, expected_temp_anchor, "ibank_got_cached_anchor");
      EXPECT_LE(error3, 1e-5);
      EXPECT_LE(error4, 1e-3);
      EXPECT_EQ(ibank_got_mask, expected_mask[0]);
    }

    start = std::chrono::high_resolution_clock::now();
    ibank.cache(pred_instance_feature, pred_anchor, pred_class_score);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    time_cost_in_ms = duration.count() * 1000.0F;
    std::cout << "[CPU Test] Instance Bank Cache() Time Cost = " << time_cost_in_ms << " [ms]" << std::endl;

    std::vector<float> ibank_cached_temp_confidence = ibank.getTempConfidence();
    std::vector<float> ibank_cached_confidence = ibank.getTempTopKConfidence();
    std::vector<float> ibank_cached_feature = ibank.getCachedFeature();
    std::vector<float> ibank_cached_anchor = ibank.getCachedAnchor();

    float error5 = 0.0F, error6 = 0.0F, error7 = 0.0F, error8 = 0.0F;
    error5 = GetMaxError(ibank_cached_temp_confidence, expected_temp_confidence, "ibank_cached_temp_confidence");
    error6 = GetMaxError(ibank_cached_confidence, expected_confidence, "ibank_cached_confidence");
    error7 = GetMaxError(ibank_cached_feature, expected_cached_feature, "ibank_cached_feature");
    error8 = GetMaxError(ibank_cached_anchor, expected_cached_anchor, "ibank_cached_anchor");
    EXPECT_LE(error5, 1e-5);
    EXPECT_LE(error6, 1e-5);
    EXPECT_LE(error7, 1e-5);
    EXPECT_LE(error8, 1e-5);

    if (i == 0) {
      pred_track_id.clear();
    }
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::int32_t> ibank_updated_cur_trackid = ibank.getTrackId(pred_track_id);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    time_cost_in_ms = duration.count() * 1000.0F;
    std::cout << "[CPU Test] Instance Bank GetTrackId() Time Cost = " << time_cost_in_ms << " [ms]" << std::endl;

    std::vector<std::int32_t> ibank_updated_temp_trackid = ibank.getCachedTrackIds();
    std::int32_t ibank_updated_previd = ibank.getPrevId();

    std::int32_t error9 = 0.0F, error10 = 0.0F;
    error9 = GetMaxError(ibank_updated_cur_trackid, expected_updated_cur_track_id, "ibank_updated_cur_trackid");
    error10 = GetMaxError(ibank_updated_temp_trackid, expected_updated_temp_track_id, "ibank_updated_temp_trackid");

    EXPECT_EQ(error9, 0);
    EXPECT_EQ(error10, 0);
    EXPECT_EQ(ibank_updated_previd, expected_prev_id[0]);
  }
}

}  // namespace head
}  // namespace sparse_end2end