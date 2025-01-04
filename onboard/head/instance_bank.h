// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_HEAD_INSTANCE_BANK_H
#define ONBOARD_HEAD_INSTANCE_BANK_H

#include <Eigen/Dense>
#include <vector>

#include "../common/common.h"
#include "../common/parameter.h"

namespace sparse_end2end {
namespace head {

class InstanceBank {
 public:
  InstanceBank(const common::E2EParams& params);
  InstanceBank() = delete;
  ~InstanceBank() = default;

  common::Status reset();

  std::tuple<const std::vector<float>&,
             const std::vector<float>&,
             const std::vector<float>&,
             const std::vector<float>&,
             const float&,
             const std::int32_t&,
             const std::vector<std::int32_t>&>
  get(const double& timestamp, const Eigen::Matrix<double, 4, 4>& global_to_lidar_mat);

  /// @brief Cached current frame's topk instance_feature, topk anchor, fusioned confidence, topk fusioned confidence
  /// and topk index
  /// @param instance_feature shape is (num_querys, embedfeat_dims)
  /// @param anchor shape is (num_querys, query_dims);
  /// @param confidence_logits shape is (num_querys, class_nums);
  common::Status cache(const std::vector<float>& instance_feature,
                       const std::vector<float>& anchor,
                       const std::vector<float>& confidence_logits);

  /// @param refined_track_ids shape is (num_querys, );
  std::vector<std::int32_t> getTrackId(const std::vector<std::int32_t>& refined_track_ids);

  std::vector<float> getTempConfidence() const;
  std::vector<float> getTempTopKConfidence() const;
  std::vector<float> getCachedFeature() const;
  std::vector<float> getCachedAnchor() const;
  std::int32_t getPrevId() const;
  std::vector<std::int32_t> getCachedTrackIds() const;

 private:
  /// @brief Spatial temporal alignment: t-1's anchor to t's anchor, it indicates temp_topk_anchors_.
  /// @param temp_anchor shape is (temp_num_querys, query_dims_);
  void anchorProjection(std::vector<float>& temp_anchor,
                        const Eigen::Matrix<float, 4, 4>& temp_to_cur_mat,
                        const float& time_interval);

  /// @brief Calculate tensor.max(-1).values.sigmoid, tensor shape is (x, y).
  ///@param confidence_logits shape is (num_querys, );
  /// @return Max confidence scores of each query, type: std::vector<T>.
  template <typename T>
  static std::vector<T> getMaxConfidenceScores(const std::vector<T>& confidence_logits,
                                               const std::uint32_t& num_querys);

  template <typename T>
  static T sigmoid(const T& logits);

  /// @param track_ids shape is (num_querys, );
  void updateTrackId(const std::vector<std::int32_t>& track_ids);

  common::E2EParams params_;

  /// @brief Instance bank init params.
  std::uint32_t num_querys_;           // default 900
  std::uint32_t num_topk_querys_;      // default 600
  std::vector<float> kmeans_anchors_;  // Shape is (num_querys, query_dims,）
  float max_time_interval_;            // Uint s
  float default_time_interval_;        // Uint s
  uint32_t query_dims_;                // dims indicates : X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ
  uint32_t embedfeat_dims_;            // default 256
  float confidence_decay_;             // default 0.5

  std::int32_t mask_;
  float time_interval_;       // Uint s
  double history_time_;       // Uint s
  std::uint32_t track_size_;  // default 900*10
  Eigen::Matrix<double, 4, 4> temp_lidar_to_global_mat_;
  std::vector<float> instance_feature_{};            // Shape is (num_querys_, embedfeat_dims)
  std::vector<float> temp_topk_instance_feature_{};  // Shape is (topk, embedfeat_dims)
  std::vector<float> temp_topk_anchors_{};           // Shape is (topk, query_dims)
  std::vector<uint32_t> temp_topk_index_{};          // Shape is (topk, )
  std::vector<std::int32_t> temp_track_ids_{};       // Shape is (num_querys_, )
  std::vector<float> temp_confidence_{};             // Shape is (num_querys_, )
  std::vector<float> temp_topk_confidence_{};        // Shape is (topk, )

  std::int32_t prev_id_;
};

}  // namespace head
}  // namespace sparse_end2end

#endif  // ONBOARD_HEAD_INSTANCE_BANK_H
