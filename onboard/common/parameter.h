// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_COMMON_PARAMETER_H
#define ONBOARD_COMMON_PARAMETER_H

#include <cstdint>
#include <vector>

#include "common.h"
namespace sparse_end2end {
namespace common {

/// @brief Parameters of Preprocessor.
struct PreprocessorParams {
  std::uint32_t num_cams = 6U;
  std::uint32_t raw_img_c = 3U;
  std::uint32_t raw_img_h = 1080U;
  std::uint32_t raw_img_w = 1920U;
  std::uint32_t model_input_img_c = 3U;
  std::uint32_t model_input_img_h = 256U;
  std::uint32_t model_input_img_w = 704U;
  float resize_ratio = 1.0F;
  std::uint32_t crop_height = 0U;
  std::uint32_t crop_width = 0U;
};

struct ModelCfgParams {
  std::uint32_t embedfeat_dims = 256U;
  std::vector<std::uint32_t> sparse4d_extract_feat_shape_lc = {};
  std::vector<std::uint32_t> sparse4d_extract_feat_spatial_shapes_ld = {};
  std::vector<std::uint32_t> sparse4d_extract_feat_spatial_shapes_nld = {};
  std::vector<std::uint32_t> sparse4d_extract_feat_level_start_index = {};
};

/// @brief Parameters of InstanceBank.
/// @param query_dims: 11 indicates X, Y, Z, W, L,H, SIN_YAW, COS_YAW, VX, VY,VZ.
/// @param kmeans_anchors: anchors' values by kmeans, size  = num_querys * query_dims.
/// @param  max_time_interval: Unit s
/// @param  default_time_interval: Unit s
struct InstanceBankParams {
  std::uint32_t num_querys = 900U;
  std::uint32_t query_dims = 11U;
  std::vector<float> kmeans_anchors;
  std::uint32_t topk_querys = 600U;
  float max_time_interval = 2.0F;
  float default_time_interval = 0.5F;
  float confidence_decay = 0.6F;
};

/// @brief Parameters of Postprocessor.
struct PostprocessorParams {
  std::uint32_t post_process_out_nums = 300U;
  float post_process_threshold = 0.2F;
};

class E2EParams {
 public:
  E2EParams(const PreprocessorParams& preprocessor_params = {6U, 3U, 1080U, 1920U, 3U, 256U, 704U},
            const ModelCfgParams& model_cfg = {256U, {}, {}, {}, {}},
            const E2ETrtEngine& sparse4d_extract_feat_engine = {"", {}, {}},
            const E2ETrtEngine& sparse4d_head1st_engine = {"", {}, {}},
            const E2ETrtEngine& sparse4d_head2nd_engine = {"", {}, {}},
            const InstanceBankParams& instance_bank_params = {900U, 11U, std::vector<float>(900 * 11), 600U, 2.0F, 0.5F,
                                                              0.6F},
            const PostprocessorParams& postprocessor_params = {300U, 0.2F})
      : preprocessor_params(preprocessor_params),
        model_cfg(model_cfg),
        sparse4d_extract_feat_engine(sparse4d_extract_feat_engine),
        sparse4d_head1st_engine(sparse4d_head1st_engine),
        sparse4d_head2nd_engine(sparse4d_head2nd_engine),
        instance_bank_params(instance_bank_params),
        postprocessor_params(postprocessor_params) {}

  ~E2EParams() = default;

 public:
  const PreprocessorParams preprocessor_params;
  const ModelCfgParams model_cfg;
  const E2ETrtEngine sparse4d_extract_feat_engine;
  const E2ETrtEngine sparse4d_head1st_engine;
  const E2ETrtEngine sparse4d_head2nd_engine;
  const InstanceBankParams instance_bank_params;
  const PostprocessorParams postprocessor_params;
};

}  // namespace common
}  // namespace sparse_end2end

#endif  // ONBOARD_COMMON_PARAMETER_H
