// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "utils.h"

#include <algorithm>

namespace sparse_end2end {
namespace head {

struct IndexedValue {
  float value;
  std::uint32_t index;
};

bool compareIndexedValue(const IndexedValue& a, const IndexedValue& b) { return a.value > b.value; }

common::Status getTopKTrackID(const std::vector<float>& confidence,  // Shape is (nums_anchors, )
                              const std::uint32_t& anchor_nums,
                              const std::uint32_t& k,
                              const std::vector<int>& track_ids,  // Shape is (nums_anchors, )
                              std::vector<int>& topk_track_ids    // Shape is (k, )
) {
  std::vector<IndexedValue> indexedValues(anchor_nums);

  for (std::uint32_t i = 0; i < anchor_nums; ++i) {
    indexedValues[i].value = confidence[i];
    indexedValues[i].index = i;
  }

  std::sort(indexedValues.begin(), indexedValues.end(), compareIndexedValue);

  for (std::uint32_t i = 0; i < k; ++i) {
    topk_track_ids.emplace_back(track_ids[indexedValues[i].index]);
  }

  return common::Status::kSuccess;
}

common::Status getTopkInstance(const std::vector<float>& confidence,        // Shape is (num_querys, )
                               const std::vector<float>& instance_feature,  // Shape is (num_querys, embedfeat_dims)
                               const std::vector<float>& anchor,            // Shape is (num_querys, query_dims)
                               const std::uint32_t& num_querys,             // Size = num_querys
                               const std::uint32_t& query_dims,             // Size = num_querys
                               const std::uint32_t& embedfeat_dims,         // Size = embedfeat_dims
                               const std::uint32_t& num_topk_querys,
                               std::vector<float>& temp_topk_confidence,
                               std::vector<float>& temp_topk_instance_feature,
                               std::vector<float>& temp_topk_anchors,
                               std::vector<std::uint32_t>& temp_topk_index) {
  std::vector<IndexedValue> indexedValues(num_querys);

  for (std::uint32_t i = 0; i < num_querys; ++i) {
    indexedValues[i].value = confidence[i];
    indexedValues[i].index = i;
  }

  std::sort(indexedValues.begin(), indexedValues.end(), compareIndexedValue);

  for (std::uint32_t i = 0; i < num_topk_querys; ++i) {
    std::uint32_t ind = indexedValues[i].index;
    temp_topk_confidence.emplace_back(confidence[ind]);
    std::copy(instance_feature.begin() + ind * embedfeat_dims, instance_feature.begin() + (ind + 1) * embedfeat_dims,
              std::back_inserter(temp_topk_instance_feature));
    std::copy(anchor.begin() + ind * query_dims, anchor.begin() + (ind + 1) * query_dims,
              std::back_inserter(temp_topk_anchors));
    temp_topk_index.emplace_back(ind);
  }

  return common::Status::kSuccess;
}

common::Status getTopKScores(const std::vector<float>& confidence,  // Shape is (nums_anchors, )
                             const std::uint32_t& anchor_nums,      // default = anchor_nums
                             const std::uint32_t& k,                // default = post_process_out_nums
                             std::vector<float>& topk_confidence,
                             std::vector<std::uint32_t>& topk_indices) {
  std::vector<IndexedValue> indexedValues(anchor_nums);
  for (std::uint32_t i = 0; i < anchor_nums; ++i) {
    indexedValues[i].value = confidence[i];
    indexedValues[i].index = i;
  }

  std::sort(indexedValues.begin(), indexedValues.end(), compareIndexedValue);

  for (std::uint32_t j = 0; j < k; ++j) {
    topk_confidence.emplace_back(indexedValues[j].value);
    topk_indices.emplace_back(indexedValues[j].index);
  }
  return common::Status::kSuccess;
}

common::Status topK(const std::vector<float>& topk_cls_scores_origin,
                    const std::vector<std::uint32_t>& topk_index,
                    const std::vector<float>& topk_fusioned_cls_scores,
                    const std::vector<std::uint8_t>& cls_ids,
                    const std::vector<float>& box_preds,
                    const std::vector<int>& track_ids,
                    const float& threshold,
                    const std::uint32_t& kmeans_anchor_dims,
                    const std::uint32_t& k,
                    std::vector<float>& actual_topk_cls_scores_origin,
                    std::vector<float>& topk_fusioned_scores,
                    std::vector<std::uint8_t>& topk_cls_ids,
                    std::vector<float>& topk_box_preds,
                    std::vector<int>& topk_track_ids,
                    std::uint32_t& actual_topk_out) {
  std::vector<IndexedValue> indexedValues;

  if (std ::abs(threshold) < std::numeric_limits<float>::epsilon()) {
    for (std::uint32_t i = 0; i < k; ++i) {
      IndexedValue iv;
      iv.value = topk_fusioned_cls_scores[i];
      iv.index = i;
      indexedValues.emplace_back(iv);
    }
    actual_topk_out = k;
  } else {
    for (std::uint32_t i = 0; i < k; ++i) {
      if (topk_cls_scores_origin[i] < threshold) {
        continue;
      }
      IndexedValue iv;
      iv.value = topk_cls_scores_origin[i];
      iv.index = i;
      indexedValues.emplace_back(iv);
    }

    if (indexedValues.empty()) {
      return common::Status::kSuccess;
    }

    actual_topk_out = std::min(k, static_cast<std::uint32_t>(indexedValues.size()));
    std::vector<IndexedValue> indexedValues2;
    for (std::uint32_t i = 0; i < actual_topk_out; ++i) {
      std::uint32_t mask_index = indexedValues[i].index;
      IndexedValue iv;
      iv.value = topk_fusioned_cls_scores[mask_index];
      iv.index = mask_index;
      indexedValues2.emplace_back(iv);
    }

    indexedValues.assign(indexedValues2.begin(), indexedValues2.end());
  }

  std::sort(indexedValues.begin(), indexedValues.end(), compareIndexedValue);
  for (std::uint32_t i = 0; i < actual_topk_out; ++i) {
    topk_fusioned_scores.emplace_back(indexedValues[i].value);

    std::uint32_t actual_topk_index = topk_index[indexedValues[i].index];
    actual_topk_cls_scores_origin.emplace_back(topk_cls_scores_origin[indexedValues[i].index]);
    topk_cls_ids.emplace_back(cls_ids[actual_topk_index]);
    topk_box_preds.insert(topk_box_preds.end(), box_preds.begin() + actual_topk_index * kmeans_anchor_dims,
                          box_preds.begin() + (actual_topk_index + 1U) * kmeans_anchor_dims);
    topk_track_ids.emplace_back(track_ids[actual_topk_index]);
  }

  return common::Status::kSuccess;
}

}  // namespace head
}  // namespace sparse_end2end