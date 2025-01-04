// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_HEAD_UTILS_H
#define ONBOARD_HEAD_UTILS_H

#include <vector>

#include "../common/common.h"

namespace sparse_end2end {
namespace head {

/// @brief Update Topk TrackID.
common::Status getTopKTrackID(const std::vector<float>& confidence,  // Shape is (nums_anchors, )
                              const std::uint32_t& anchor_nums,      // confidence size = nums_anchors
                              const std::uint32_t& k,                // Size =  temp_anchor_nums
                              const std::vector<int>& track_ids,
                              std::vector<int>& topk_track_ids);

/// @brief Update Topk Instance: instance_feature, anchor.
common::Status getTopkInstance(const std::vector<float>& confidence,        // Shape is (num_querys, )
                               const std::vector<float>& instance_feature,  // Shape is (num_querys, embedfeat_dims)
                               const std::vector<float>& anchor,            // Shape is (num_querys, query_dims)
                               const std::uint32_t& num_querys,             // Ssize = num_querys
                               const std::uint32_t& query_dims,             // Size = num_querys
                               const std::uint32_t& embedfeat_dims,         // Size = embedfeat_dims
                               const std::uint32_t& num_topk_querys,
                               std::vector<float>& temp_topk_confidence,
                               std::vector<float>& temp_topk_instance_feature,
                               std::vector<float>& temp_topk_anchors,
                               std::vector<std::uint32_t>& temp_topk_index);

/// @brief Get Topk Scores and Index.
common::Status getTopKScores(const std::vector<float>& confidence,  // Shape is (nums_anchors, )
                             const std::uint32_t& anchor_nums,      // Size = anchor_nums
                             const std::uint32_t& k,                // default = post_process_out_nums
                             std::vector<float>& topk_confidence,
                             std::vector<std::uint32_t>& topk_indices);

/// @brief Update PostProcess Topk scores, quality.
common::Status topK(const std::vector<float>& topk_cls_scores_origin,  // Shape is (topk, )
                    const std::vector<std::uint32_t>& topk_index,      // Shape is (topk, )
                    const std::vector<float>& fusioned_scores,         // Shape is (topk, )
                    const std::vector<std::uint8_t>& cls_ids,          // Shape is (nums_anchors, )
                    const std::vector<float>& box_preds,               // Shape is (nums_anchors, )
                    const std::vector<int>& track_ids,                 // Shape is (nums_anchors, )
                    const float& threshold,
                    const std::uint32_t& kmeans_anchor_dims,
                    const std::uint32_t& k,
                    std::vector<float>& topk_cls_scores,
                    std::vector<float>& topk_fusioned_scores,
                    std::vector<std::uint8_t>& topk_cls_ids,
                    std::vector<float>& topk_box_preds,
                    std::vector<int>& topk_track_ids,
                    std::uint32_t& actual_topk_out);

}  // namespace head
}  // namespace sparse_end2end
#endif  // ONBOARD_HEAD_UTILS_H
