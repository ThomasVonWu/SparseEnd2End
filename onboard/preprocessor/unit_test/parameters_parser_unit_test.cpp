// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "../../common/parameter.h"
#include "../parameters_parser.h"

namespace sparse_end2end {
namespace preprocessor {

TEST(ParseParamsUnitTest, ParseParamsFunctionCall) {
  std::filesystem::path current_dir = std::filesystem::current_path();
  const common::E2EParams params = parseParams(current_dir / "../../assets/model_cfg.yaml");

  printf("[Preprocessor Parameters Infos]:\n");
  printf("num_cams = %u\n", params.preprocessor_params.num_cams);
  printf("raw_img_c = %u\n", params.preprocessor_params.raw_img_c);
  printf("raw_img_h = %u\n", params.preprocessor_params.raw_img_h);
  printf("raw_img_w = %u\n", params.preprocessor_params.raw_img_w);
  printf("model_input_img_c = %u\n", params.preprocessor_params.model_input_img_c);
  printf("model_input_img_h = %u\n", params.preprocessor_params.model_input_img_h);
  printf("model_input_img_w = %u\n", params.preprocessor_params.model_input_img_w);

  printf("[ModelCfg Parameters Infos]:\n");
  printf("embedfeat_dims = %u\n", params.model_cfg.embedfeat_dims);
  // printf("sparse4d_extract_feat_shape_lc = %u\n", params.model_cfg.sparse4d_extract_feat_shape_lc);
  // printf("sparse4d_extract_feat_spatial_shapes_ld = %u\n", params.model_cfg.sparse4d_extract_feat_spatial_shapes_ld);
  // printf("sparse4d_extract_feat_spatial_shapes_nld = %u\n",
  // params.model_cfg.sparse4d_extract_feat_spatial_shapes_nld); printf("sparse4d_extract_feat_level_start_index =
  // %u\n", params.model_cfg.sparse4d_extract_feat_level_start_index);

  printf("[Sparse4dExtractFeatEngine Parameters Infos]:\n");
  // printf("sparse4d_extract_feat_engine.engine_path = %u\n", params.sparse4d_extract_feat_engine.engine_path.c_str());
  // printf("sparse4d_extract_feat_engine. input_names= %u\n",
  // params.sparse4d_extract_feat_engine.input_names[0].c_str()); printf("sparse4d_extract_feat_engine. output_names=
  // %u\n",
  //        params.sparse4d_extract_feat_engine.output_names[0].c_str());

  printf("[Sparse4dHead1stEngine Parameters Infos]:\n");
  // printf("sparse4d_head1st_engine.engine_path = %u\n", params.sparse4d_head1st_engine.engine_path.c_str());
  // printf("sparse4d_head1st_engine. input_names= %u\n", params.sparse4d_head1st_engine.input_names[0].c_str());
  // printf("sparse4d_head1st_engine. output_names= %u\n", params.sparse4d_head1st_engine.output_names[0].c_str());

  printf("[Sparse4dHead2ndEngine Parameters Infos]:\n");
  // printf("sparse4d_head2nd_engine.engine_path = %u\n", params.sparse4d_head2nd_engine.engine_path.c_str());
  // printf("sparse4d_head2nd_engine. input_names= %u\n", params.sparse4d_head2nd_engine.input_names[0].c_str());
  // printf("sparse4d_head2nd_engine. output_names= %u\n", params.sparse4d_head2nd_engine.output_names[0].c_str());

  printf("\n[InstanceBank Parameters Infos]:\n");
  printf("num_querys = %u\n", params.instance_bank_params.num_querys);
  printf("query_dims = %u\n", params.instance_bank_params.query_dims);
  // printf("kmeans_anchors = %u\n", params.instance_bank_params.kmeans_anchors);
  printf("topk_querys = %u\n", params.instance_bank_params.topk_querys);
  printf("max_time_interval = %f\n", params.instance_bank_params.max_time_interval);
  printf("default_time_interval = %f\n", params.instance_bank_params.default_time_interval);
  printf("confidence_decay = %f\n", params.instance_bank_params.confidence_decay);

  printf("\n[Postprocessor Parameters Infos]:\n");
  printf("post_process_out_nums = %u\n", params.postprocessor_params.post_process_out_nums);
  printf("post_process_threshold = %f\n", params.postprocessor_params.post_process_threshold);

  // printf("\nsparse4dhead2nd_trtio.engine_path = %s\n", params.sparse4dhead2nd_trtio.engine_path.c_str());
  // printf("sparse4dhead2nd_trtio.input_names:\n");
  // for (size_t i = 0; i < params.sparse4dhead2nd_trtio.input_names.size(); ++i) {
  //   printf("%s ", params.sparse4dhead2nd_trtio.input_names[i].c_str());
  // }
  // printf("\nsparse4dhead2nd_trtio.output_names:\n");
  // for (size_t i = 0; i < params.sparse4dhead2nd_trtio.output_names.size(); ++i) {
  //   printf("%s ", params.sparse4dhead2nd_trtio.output_names[i].c_str());
  // }
}
}  // namespace preprocessor
}  // namespace sparse_end2end