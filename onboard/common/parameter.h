// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_COMMON_PARAMETER_H
#define ONBOARD_COMMON_PARAMETER_H

#include <cstdint>
#include <vector>

#include "common.h"
namespace sparse_end2end {
namespace common {

/// @brief Params of Preprocessor.
struct PreprocessorParams {
  std::uint32_t num_cams = 6U;
  std::uint32_t raw_img_c = 3U;
  std::uint32_t raw_img_h = 1080U;
  std::uint32_t raw_img_w = 1920U;
  //   float resize_ratio = 0.4297162313524393F; infer get it.
  std::uint32_t model_input_img_c = 3U;
  std::uint32_t model_input_img_h = 256U;
  std::uint32_t model_input_img_w = 704U;
};

/// @brief Params of InstanceBank.
/// @param query_dims: 11 indicates X, Y, Z, W, L,H, SIN_YAW, COS_YAW, VX, VY,VZ.
/// @param kmeans_anchors: anchors' values by kmeans, size  = num_querys * query_dims.
/// @param  max_time_interval: Unit s
/// @param  default_time_interval: Unit s
struct InstanceBankParams {
  std::uint32_t num_querys = 900U;
  std::uint32_t query_dims = 11U;
  std::vector<float> kmeans_anchors;
  std::uint32_t topK_num_querys = 600U;
  float max_time_interval = 2.0F;
  float default_time_interval = 0.5F;
  float confidence_decay = 0.6F;
};

class E2EParams {
 public:
  E2EParams(const InstanceBankParams& instance_bank_params = {900U, 11U, std::vector<float>(900 * 11), 600U, 2.0F, 0.5F,
                                                              0.6F},
            const PreprocessorParams& preprocessor_params = {6U, 3U, 1080U, 1920U, 3U, 256U, 704U},
            const E2EEngine& sparse4dbackbone_engine = {"", {}, {}},
            const E2EEngine& sparse4dhead1st_engine = {"", {}, {}},
            const E2EEngine& sparse4dhead2nd_engine = {"", {}, {}})
      : instance_bank_params(instance_bank_params), preprocessor_params(preprocessor_params) {}

  ~E2EParams() = default;

 public:
  const InstanceBankParams instance_bank_params;
  const PreprocessorParams preprocessor_params;
  const E2EEngine sparse4dbackbone_engine;
  const E2EEngine sparse4dhead1st_engine;
  const E2EEngine sparse4dhead2nd_engine;
};

}  // namespace common
}  // namespace sparse_end2end

#endif  // ONBOARD_COMMON_PARAMETER_H
