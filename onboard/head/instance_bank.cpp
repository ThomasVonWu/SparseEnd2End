// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "instance_bank.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "utils.h"

namespace sparse_end2end {
namespace head {

InstanceBank::InstanceBank(const common::E2EParams& params)
    : params_(params),
      num_querys_(params_.instance_bank_params.num_querys),
      num_topk_querys_(params_.instance_bank_params.topk_querys),
      kmeans_anchors_(std::move(params_.instance_bank_params.kmeans_anchors)),
      max_time_interval_(params_.instance_bank_params.max_time_interval),
      default_time_interval_(params_.instance_bank_params.default_time_interval),
      query_dims_(params_.instance_bank_params.query_dims),
      embedfeat_dims_(params_.model_cfg.embedfeat_dims),
      confidence_decay_(params_.instance_bank_params.confidence_decay) {
  mask_ = 0;
  history_time_ = 0.0F;
  time_interval_ = 0.0F;
  temp_lidar_to_global_mat_ = Eigen::Matrix<double, 4, 4>::Zero();
  track_size_ = static_cast<std::uint32_t>(common::ObstacleType::OBSTACLETYPE_Max) * num_querys_;
  instance_feature_.resize(num_querys_ * embedfeat_dims_);
  temp_topk_instance_feature_.reserve(num_topk_querys_ * embedfeat_dims_);
  temp_topk_anchors_.reserve(num_topk_querys_ * query_dims_);
  temp_topk_index_.reserve(num_topk_querys_);
  temp_track_ids_.reserve(num_querys_);
  temp_confidence_.reserve(num_querys_);
  temp_topk_confidence_.reserve(num_topk_querys_);
  reset();
}

common::Status InstanceBank::reset() {
  temp_topk_instance_feature_.clear();
  temp_topk_anchors_.clear();
  temp_topk_index_.clear();
  temp_track_ids_.clear();
  temp_confidence_.clear();
  temp_topk_confidence_.clear();
  prev_id_ = 0;
  return common::Status::kSuccess;
}

void InstanceBank::anchorProjection(std::vector<float>& temp_topk_anchors,
                                    const Eigen::Matrix<float, 4, 4>& temp2cur_mat,
                                    const float& time_interval) {
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> temp_topk_anchors_mat(
      temp_topk_anchors.data(), num_topk_querys_, query_dims_);

  // t-1 center to t center
  auto center = temp_topk_anchors_mat.leftCols<3>();  // num_topk_querys_ * 3
  auto vel = temp_topk_anchors_mat.rightCols<3>();    // num_topk_querys_ * 3
  auto translation = vel * (-time_interval);          // num_topk_querys_ * 3
  center = center - translation;                      // num_topk_querys_ × 3

  Eigen::MatrixXf center_homo(num_topk_querys_, 4);
  center_homo.block(0, 0, num_topk_querys_, 3) = center;
  center_homo.col(3).setOnes();
  center_homo = center_homo * temp2cur_mat.transpose();

  // t-1 vel to t vel
  vel = vel * temp2cur_mat.block(0, 0, 3, 3).transpose();

  // t-1 yaw to t yaw
  auto yaw = temp_topk_anchors_mat.block(0, 6, num_topk_querys_, 2);  // num_topk_querys_ × 2
  yaw.col(0).swap(yaw.col(1));
  yaw = yaw * temp2cur_mat.block(0, 0, 2, 2).transpose();  // num_topk_querys_ × 2

  auto size = temp_topk_anchors_mat.block(0, 3, num_topk_querys_, 3);
  Eigen::MatrixXf temp2cur_anchor_m(num_topk_querys_, query_dims_);
  temp2cur_anchor_m << center_homo.leftCols<3>(), size, yaw, vel;

  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> temp2cur_anchor_map(
      temp_topk_anchors.data(), num_topk_querys_, query_dims_);
  temp2cur_anchor_map = temp2cur_anchor_m;
}

std::tuple<const std::vector<float>&,
           const std::vector<float>&,
           const std::vector<float>&,
           const std::vector<float>&,
           const float&,
           const std::int32_t&,
           const std::vector<std::int32_t>&>
InstanceBank::get(const double& timestamp, const Eigen::Matrix<double, 4, 4>& global_to_lidar_mat) {
  if (!temp_topk_anchors_.empty()) {
    time_interval_ = static_cast<float>(std::fabs(timestamp - history_time_));
    float epsilon = std::numeric_limits<float>::epsilon();
    mask_ = (time_interval_ < max_time_interval_ || std::fabs(time_interval_ - max_time_interval_) < epsilon) ? 1 : 0;
    time_interval_ = (static_cast<bool>(mask_) && time_interval_ > epsilon) ? time_interval_ : default_time_interval_;

    Eigen::Matrix<double, 4, 4> temp2cur_mat_double = global_to_lidar_mat * temp_lidar_to_global_mat_;
    Eigen::Matrix<float, 4, 4> temp2cur_mat = temp2cur_mat_double.cast<float>();
    anchorProjection(temp_topk_anchors_, temp2cur_mat, time_interval_);
  } else {
    reset();
    time_interval_ = default_time_interval_;
  }

  history_time_ = timestamp;
  temp_lidar_to_global_mat_ = global_to_lidar_mat.inverse();

  return std::make_tuple(std::cref(instance_feature_), std::cref(kmeans_anchors_),
                         std::cref(temp_topk_instance_feature_), std::cref(temp_topk_anchors_),
                         std::cref(time_interval_), std::cref(mask_), std::cref(temp_track_ids_));
}

common::Status InstanceBank::cache(const std::vector<float>& instance_feature,
                                   const std::vector<float>& anchor,
                                   const std::vector<float>& confidence_logits) {
  std::vector<float> confidence = InstanceBank::getMaxConfidenceScores(confidence_logits, num_querys_);

  if (!temp_topk_confidence_.empty()) {
    std::vector<float> temp_topk_confidence_with_decay(temp_topk_confidence_.size());
    std::transform(temp_topk_confidence_.begin(), temp_topk_confidence_.end(), temp_topk_confidence_with_decay.begin(),
                   [this](float element) { return element * confidence_decay_; });
    std::transform(confidence.begin(), confidence.begin() + num_topk_querys_, temp_topk_confidence_with_decay.begin(),
                   confidence.begin(), [](float a, float b) { return std::max(a, b); });
  }
  temp_confidence_ = confidence;  // fusioned anchor with unordered confidence

  temp_topk_confidence_.clear();
  temp_topk_instance_feature_.clear();
  temp_topk_anchors_.clear();
  temp_topk_index_.clear();
  getTopkInstance(confidence, instance_feature, anchor, num_querys_, query_dims_, embedfeat_dims_, num_topk_querys_,
                  temp_topk_confidence_, temp_topk_instance_feature_, temp_topk_anchors_, temp_topk_index_);

  return common::Status::kSuccess;
}

std::vector<std::int32_t> InstanceBank::getTrackId(const std::vector<std::int32_t>& refined_track_ids) {
  std::vector<std::int32_t> track_ids(num_querys_, -1);
  if (!refined_track_ids.empty()) {
    if (refined_track_ids.size() != num_querys_) {
      throw "[ERROR] refined_track_ids size is mismatch !";
    }
    std::copy(refined_track_ids.begin(), refined_track_ids.end(), track_ids.begin());
  }

  auto nums_new_anchor = static_cast<size_t>(
      std::count_if(track_ids.begin(), track_ids.end(), [](std::int32_t track_id) { return track_id < 0; }));
  std::vector<std::int32_t> new_track_ids(nums_new_anchor);
  for (size_t k = 0; k < nums_new_anchor; ++k) {
    new_track_ids[k] = k + prev_id_;
  }

  for (size_t i = num_querys_ - nums_new_anchor, j = 0; i < num_querys_ && j < nums_new_anchor; ++i, ++j) {
    track_ids[i] = new_track_ids[j];
  }

  prev_id_ += nums_new_anchor;
  updateTrackId(track_ids);
  return track_ids;
}

void InstanceBank::updateTrackId(const std::vector<std::int32_t>& track_ids) {
  std::vector<std::int32_t> topk_track_ids;
  topk_track_ids.reserve(num_topk_querys_);
  for (const auto& i : temp_topk_index_) {
    topk_track_ids.emplace_back(track_ids[i]);
  }

  temp_track_ids_.clear();
  temp_track_ids_.insert(temp_track_ids_.end(), topk_track_ids.begin(), topk_track_ids.end());
  temp_track_ids_.resize(num_querys_, -1);
}

std::vector<float> InstanceBank::getTempConfidence() const { return temp_confidence_; }

std::vector<float> InstanceBank::getTempTopKConfidence() const { return temp_topk_confidence_; }

std::vector<float> InstanceBank::getCachedFeature() const { return temp_topk_instance_feature_; }

std::vector<float> InstanceBank::getCachedAnchor() const { return temp_topk_anchors_; }

std::int32_t InstanceBank::getPrevId() const { return prev_id_; }

std::vector<std::int32_t> InstanceBank::getCachedTrackIds() const { return temp_track_ids_; }

template <typename T>
std::vector<T> InstanceBank::getMaxConfidenceScores(const std::vector<T>& confidence_logits,
                                                    const std::uint32_t& num_querys) {
  std::vector<T> max_confidence_scores;
  for (std::uint32_t i = 0; i < num_querys; ++i) {
    T max_confidence_logit = confidence_logits[i * static_cast<std::uint32_t>(common::ObstacleType::OBSTACLETYPE_Max)];
    for (std::uint32_t j = 0; j < static_cast<std::uint32_t>(common::ObstacleType::OBSTACLETYPE_Max); ++j) {
      std::uint32_t index = i * static_cast<std::uint32_t>(common::ObstacleType::OBSTACLETYPE_Max) + j;
      if (confidence_logits[index] > max_confidence_logit) {
        max_confidence_logit = confidence_logits[index];
      }
    }
    T max_confidence_score = sigmoid(max_confidence_logit);
    max_confidence_scores.emplace_back(max_confidence_score);
  }
  return max_confidence_scores;
}

template <typename T>
T InstanceBank::sigmoid(const T& score_logits) {
  T scores = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-score_logits));

  return scores;
}

}  // namespace head
}  // namespace sparse_end2end