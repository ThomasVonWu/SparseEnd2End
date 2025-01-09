// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "parameters_parser.h"

#include <yaml-cpp/yaml.h>

#include <map>
#include <numeric>
#include <string>

#include "../common/common.h"
#include "../common/utils.h"
namespace sparse_end2end {
namespace preprocessor {

common::E2EParams parseParams(const std::string& model_cfg_path) {
  const std::map<std::string, common::ObstacleType> obstacle_type_lut{
      {"CAR", common::ObstacleType::CAR},
      {"TRCUK", common::ObstacleType::TRCUK},
      {"CONSTRUCTION_VEHICLE", common::ObstacleType::CONSTRUCTION_VEHICLE},
      {"BUS", common::ObstacleType::BUS},
      {"TRAILER", common::ObstacleType::TRAILER},
      {"BARRIER", common::ObstacleType::BARRIER},
      {"MOTORCYCLE", common::ObstacleType::MOTORCYCLE},
      {"BICYCLE", common::ObstacleType::BICYCLE},
      {"PEDESTRIAN", common::ObstacleType::PEDESTRIAN},
      {"TRAFFIC_CONE", common::ObstacleType::TRAFFIC_CONE}};

  const std::map<std::string, common::CameraFrameID> task_camera_frameid_lut{
      {"CAM_FRONT", common::CameraFrameID::CAM_FRONT},
      {"CAM_FRONT_RIGHT", common::CameraFrameID::CAM_FRONT_RIGHT},
      {"CAM_FRONT_LEFT", common::CameraFrameID::CAM_FRONT_LEFT},
      {"CAM_BACK", common::CameraFrameID::CAM_BACK},
      {"CAM_BACK_LEFT", common::CameraFrameID::CAM_BACK_LEFT},
      {"CAM_BACK_RIGHT", common::CameraFrameID::CAM_BACK_RIGHT},
  };

  /// @brief STEP-1: Load model config file from YAML file.
  YAML::Node config_file_node = common::loadYamlFile(model_cfg_path);

  YAML::Node offline_onboard_bin_file_path_node = common::getYamlSubNode(config_file_node, "OfflineOnboardBinFilePath");

  /// @brief STEP-2: Parse kmeans anchor from offline bin file.
  YAML::Node kmeans_anchors_path_node =
      common::getYamlSubNode(offline_onboard_bin_file_path_node[0], "KmeansAnchorsPath");
  std::string kmeans_anchors_path = kmeans_anchors_path_node.as<std::string>();
  std::vector<float> kmeans_anchors = common::readfile_wrapper<float>(kmeans_anchors_path);

  /// @brief STEP-3: Parse calibration:lidar2image from offline bin file:
  YAML::Node lidar2image_path_node = common::getYamlSubNode(offline_onboard_bin_file_path_node[1], "Lidar2ImagePath");
  std::string lidar2image_path = lidar2image_path_node.as<std::string>();
  std::vector<float> lidar2img = common::readfile_wrapper<float>(lidar2image_path);

  YAML::Node sparse_e2e_node = common::getYamlSubNode(config_file_node, "SparseE2E");

  /// @brief STEP-4: Parse Node TaskCameraFrameID.
  std::vector<common::CameraFrameID> task_camera_frame_id;
  YAML::Node task_camera_frame_id_node = common::getYamlSubNode(sparse_e2e_node, "TaskCameraFrameID");
  std::vector<std::string> task_camera_frame_id_strs = task_camera_frame_id_node.as<std::vector<std::string>>();
  size_t num_cams = task_camera_frame_id_strs.size();

  /// @brief STEP-5: Parse Node: `ImgPreprocessor` in YAML.
  YAML::Node img_preprocessor_node = common::getYamlSubNode(sparse_e2e_node, "ImgPreprocessor");

  /// STEP-5.1: Parse Node: `ImgPreprocessor/RawImgShape_CHW` in YAML.
  YAML::Node raw_img_shape_chw_node = common::getYamlSubNode(img_preprocessor_node[0], "RawImgShape_CHW");
  std::vector<std::uint32_t> raw_img_shape_chw = raw_img_shape_chw_node.as<std::vector<std::uint32_t>>();

  /// STEP-5.2: Parse Node: `ImgPreprocessor/ModelInputImgShape_CHW` in YAML.
  YAML::Node model_input_img_shape_chw_node =
      common::getYamlSubNode(img_preprocessor_node[1], "ModelInputImgShape_CHW");
  std::vector<std::uint32_t> model_input_img_shape_chw =
      model_input_img_shape_chw_node.as<std::vector<std::uint32_t>>();

  /// @attention Calculate image preprocess's parameters: `resize_ratio/crop_height/crop_width` based on the given
  /// configuration parameters in YAML.
  float resize_ratio =
      std::max(static_cast<float>(model_input_img_shape_chw[1]) / static_cast<float>(raw_img_shape_chw[1]),
               static_cast<float>(model_input_img_shape_chw[2]) / static_cast<float>(raw_img_shape_chw[2]));
  std::uint32_t resize_dimH =
      static_cast<std::uint32_t>(std::floor(resize_ratio * static_cast<float>(raw_img_shape_chw[1])));
  std::uint32_t resize_dimW =
      static_cast<std::uint32_t>(std::floor(resize_ratio * static_cast<float>(raw_img_shape_chw[2])));
  std::uint32_t crop_height = resize_dimH - static_cast<std::uint32_t>(model_input_img_shape_chw[1]);
  std::uint32_t crop_width = static_cast<std::uint32_t>(
      std::max(0.0F, static_cast<float>(resize_dimW) - static_cast<float>(model_input_img_shape_chw[2])) / 2.0F);

  /// @brief STEP-6: Parse Node: `EmbedFeatDims` in YAML.
  YAML::Node embedfeat_dims_node = common::getYamlSubNode(sparse_e2e_node, "EmbedFeatDims");
  std::uint32_t embedfeat_dims = embedfeat_dims_node.as<std::uint32_t>();

  /// @brief STEP-7: Parse Node: `Sparse4dExtractFeatTrtEngine` in YAML.
  YAML::Node sparse4d_extract_feat_trt_engine_node =
      common::getYamlSubNode(sparse_e2e_node, "Sparse4dExtractFeatTrtEngine");

  /// STEP-7.1: Parse Node: `Sparse4dExtractFeatTrtEngine/EnginePath` in YAML.
  YAML::Node sparse4d_extract_feat_trt_engine_path_node =
      common::getYamlSubNode(sparse4d_extract_feat_trt_engine_node[0], "EnginePath");
  std::string sparse4d_extract_feat_trt_engine_path = sparse4d_extract_feat_trt_engine_path_node.as<std::string>();

  /// STEP-7.2 : Parse Node: `Sparse4dExtractFeatTrtEngine/EngineInputNames` in YAML.
  YAML::Node sparse4d_extract_feat_trt_engine_input_names_node =
      common::getYamlSubNode(sparse4d_extract_feat_trt_engine_node[1], "EngineInputNames");
  std::vector<std::string> sparse4d_extract_feat_trt_engine_input_names =
      sparse4d_extract_feat_trt_engine_input_names_node.as<std::vector<std::string>>();

  /// STEP-7.3 : Parse Node: `Sparse4dExtractFeatTrtEngine/EngineOutputNames` in YAML.
  YAML::Node sparse4d_extract_feat_trt_engine_output_names_node =
      common::getYamlSubNode(sparse4d_extract_feat_trt_engine_node[2], "EngineOutputNames");
  std::vector<std::string> sparse4d_extract_feat_trt_engine_output_names =
      sparse4d_extract_feat_trt_engine_output_names_node.as<std::vector<std::string>>();

  /// STEP-7.4: Parse Node: `Sparse4dExtractFeatTrtEngine/Sparse4dExtractFeatShape_LC` in YAML.
  YAML::Node sparse4d_extract_feat_shape_lc_node =
      common::getYamlSubNode(sparse4d_extract_feat_trt_engine_node[3], "Sparse4dExtractFeatShape_LC");
  std::vector<std::uint32_t> sparse4d_extract_feat_shape_lc =
      sparse4d_extract_feat_shape_lc_node.as<std::vector<std::uint32_t>>();

  /// STEP-7.5 Parse Node: `Sparse4dExtractFeatTrtEngine/Sparse4dExtractFeatSpatialShapes_LD` in YAML.
  YAML::Node sparse4d_extract_feat_spatial_shapes_ld_node =
      common::getYamlSubNode(sparse4d_extract_feat_trt_engine_node[4], "Sparse4dExtractFeatSpatialShapes_LD");
  std::vector<std::uint32_t> sparse4d_extract_feat_spatial_shapes_ld =
      sparse4d_extract_feat_spatial_shapes_ld_node.as<std::vector<std::uint32_t>>();

  /// Get sparse4d_extract_feat_level_start_index based on sparse4d_extract_feat_spatial_shapes_ld.
  std::vector<std::uint32_t> sparse4d_extract_feat_spatial_shapes_nld{};
  for (size_t i = 0; i < num_cams; ++i) {
    sparse4d_extract_feat_spatial_shapes_nld.insert(
        sparse4d_extract_feat_spatial_shapes_nld.end(),
        {sparse4d_extract_feat_spatial_shapes_ld[0], sparse4d_extract_feat_spatial_shapes_ld[1],
         sparse4d_extract_feat_spatial_shapes_ld[2], sparse4d_extract_feat_spatial_shapes_ld[3],
         sparse4d_extract_feat_spatial_shapes_ld[4], sparse4d_extract_feat_spatial_shapes_ld[5],
         sparse4d_extract_feat_spatial_shapes_ld[6], sparse4d_extract_feat_spatial_shapes_ld[7]});
  }

  std::vector<std::uint32_t> sparse4d_extract_feat_level_start_index_tmp{};
  for (size_t i = 0; i < num_cams; ++i) {
    sparse4d_extract_feat_level_start_index_tmp.insert(
        sparse4d_extract_feat_level_start_index_tmp.end(),
        {sparse4d_extract_feat_spatial_shapes_nld[0] * sparse4d_extract_feat_spatial_shapes_nld[1],
         sparse4d_extract_feat_spatial_shapes_nld[2] * sparse4d_extract_feat_spatial_shapes_nld[3],
         sparse4d_extract_feat_spatial_shapes_nld[4] * sparse4d_extract_feat_spatial_shapes_nld[5],
         sparse4d_extract_feat_spatial_shapes_nld[6] * sparse4d_extract_feat_spatial_shapes_nld[7]});
  }
  std::vector<std::uint32_t> sparse4d_extract_feat_level_start_index(
      sparse4d_extract_feat_level_start_index_tmp.size());
  std::partial_sum(sparse4d_extract_feat_level_start_index_tmp.begin(),
                   sparse4d_extract_feat_level_start_index_tmp.end(), sparse4d_extract_feat_level_start_index.begin());
  sparse4d_extract_feat_level_start_index.pop_back();
  sparse4d_extract_feat_level_start_index.insert(sparse4d_extract_feat_level_start_index.begin(), 0);

  /// @brief STEP-8 Parse Node: `MultiViewMultiScaleDeformableAttentionAggregationPath` in YAML.
  YAML::Node multiview_multiscale_deformable_attention_aggregation_path_node =
      common::getYamlSubNode(sparse_e2e_node, "MultiViewMultiScaleDeformableAttentionAggregationPath");
  std::string multiview_multiscale_deformable_attention_aggregation_path =
      multiview_multiscale_deformable_attention_aggregation_path_node.as<std::string>();

  /// @brief STEP-9 Parse Node: `Sparse4dHeadFirstFrameEngine` in YAML.
  YAML::Node sparse4d_head_first_frame_engine_node =
      common::getYamlSubNode(sparse_e2e_node, "Sparse4dHeadFirstFrameEngine");

  /// STEP-9.1 Parse Node:`Sparse4dHeadFirstFrameEngine/EnginePath` in YAML.
  YAML::Node sparse4d_head1st_engine_path_node =
      common::getYamlSubNode(sparse4d_head_first_frame_engine_node[0], "EnginePath");
  std::string sparse4d_head1st_engine_path = sparse4d_head1st_engine_path_node.as<std::string>();

  /// STEP-9.2 : Parse Node: `Sparse4dHeadFirstFrameEngine/EngineInputNames` in YAML.
  YAML::Node sparse4d_head1st_engine_input_names_node =
      common::getYamlSubNode(sparse4d_head_first_frame_engine_node[1], "EngineInputNames");
  std::vector<std::string> sparse4d_head1st_engine_input_names =
      sparse4d_head1st_engine_input_names_node.as<std::vector<std::string>>();

  /// STEP-9.3 : Parse Node: `Sparse4dHeadFirstFrameEngine/EngineOutputNames` in YAML.
  YAML::Node sparse4d_head1st_engine_output_names_node =
      common::getYamlSubNode(sparse4d_head_first_frame_engine_node[2], "EngineOutputNames");
  std::vector<std::string> sparse4d_head1st_engine_output_names =
      sparse4d_head1st_engine_output_names_node.as<std::vector<std::string>>();

  /// @brief STEP-10 Parse Node: `Sparse4dHeadSecondFrameEngine` in YAML.
  YAML::Node sparse4d_head_second_frame_engine_node =
      common::getYamlSubNode(sparse_e2e_node, "Sparse4dHeadSecondFrameEngine");

  /// STEP-10.1 Parse Node:`Sparse4dHeadSecondFrameEngine/EnginePath` in YAML.
  YAML::Node sparse4d_head2nd_engine_path_node =
      common::getYamlSubNode(sparse4d_head_second_frame_engine_node[0], "EnginePath");
  std::string sparse4d_head2nd_engine_path = sparse4d_head2nd_engine_path_node.as<std::string>();

  /// STEP-10.2 : Parse Node: `Sparse4dHeadSecondFrameEngine/EngineInputNames` in YAML.
  YAML::Node sparse4d_head2nd_engine_input_names_node =
      common::getYamlSubNode(sparse4d_head_second_frame_engine_node[1], "EngineInputNames");
  std::vector<std::string> sparse4d_head2nd_engine_input_names =
      sparse4d_head2nd_engine_input_names_node.as<std::vector<std::string>>();

  /// STEP-10.3 : Parse Node: `Sparse4dHeadSecondFrameEngine/EngineOutputNames` in YAML.
  YAML::Node sparse4d_head2nd_engine_output_names_node =
      common::getYamlSubNode(sparse4d_head_second_frame_engine_node[2], "EngineOutputNames");
  std::vector<std::string> sparse4d_head2nd_engine_output_names =
      sparse4d_head2nd_engine_output_names_node.as<std::vector<std::string>>();

  ///@brief STEP-11 Parse Node: `InstanceBankParams` in YAML.
  YAML::Node instance_bank_params_node = common::getYamlSubNode(sparse_e2e_node, "InstanceBankParams");

  YAML::Node num_querys_node = common::getYamlSubNode(instance_bank_params_node[0], "NumQuerys");
  std::uint32_t num_querys = num_querys_node.as<std::uint32_t>();

  YAML::Node query_dims_node = common::getYamlSubNode(instance_bank_params_node[1], "QueryDims");
  std::uint32_t query_dims = query_dims_node.as<std::uint32_t>();

  YAML::Node topk_querys_node = common::getYamlSubNode(instance_bank_params_node[2], "TopKQuerys");
  std::uint32_t topk_querys = topk_querys_node.as<std::uint32_t>();

  YAML::Node max_time_interval_node = common::getYamlSubNode(instance_bank_params_node[3], "MaxTimeInterval");
  float max_time_interval = max_time_interval_node.as<float>();

  YAML::Node default_time_interval_node = common::getYamlSubNode(instance_bank_params_node[4], "DefaultTimeInterval");
  float default_time_interval = default_time_interval_node.as<float>();

  YAML::Node confidence_decay_node = common::getYamlSubNode(instance_bank_params_node[5], "ConfidenceDecay");
  float confidence_decay = confidence_decay_node.as<float>();

  ///@brief STEP-12 Parse Node: `PostProcess` in YAML.
  YAML::Node post_process_node = common::getYamlSubNode(sparse_e2e_node, "PostProcess");

  YAML::Node post_process_out_nums_node = common::getYamlSubNode(post_process_node[0], "PostProcessOutNums");
  std::uint32_t post_process_out_nums = post_process_out_nums_node.as<std::uint32_t>();

  YAML::Node post_process_threshold_node = common::getYamlSubNode(post_process_node[1], "PostProcessThreshold");
  float post_process_threshold = post_process_threshold_node.as<float>();

  ///@brief STEP-13 Construct E2EParams::params based on the the parsed parameters.
  common::PreprocessorParams preprocessor_params;
  preprocessor_params.num_cams = num_cams;
  preprocessor_params.raw_img_c = raw_img_shape_chw[0];
  preprocessor_params.raw_img_h = raw_img_shape_chw[1];
  preprocessor_params.raw_img_w = raw_img_shape_chw[2];
  preprocessor_params.model_input_img_c = model_input_img_shape_chw[0];
  preprocessor_params.model_input_img_h = model_input_img_shape_chw[1];
  preprocessor_params.model_input_img_w = model_input_img_shape_chw[2];
  preprocessor_params.resize_ratio = resize_ratio;
  preprocessor_params.crop_height = crop_height;
  preprocessor_params.crop_width = crop_width;

  common::ModelCfgParams model_cfg_params;
  model_cfg_params.embedfeat_dims = embedfeat_dims;
  model_cfg_params.sparse4d_extract_feat_shape_lc = sparse4d_extract_feat_shape_lc;
  model_cfg_params.sparse4d_extract_feat_spatial_shapes_ld = sparse4d_extract_feat_spatial_shapes_ld;
  model_cfg_params.sparse4d_extract_feat_level_start_index = sparse4d_extract_feat_level_start_index;
  model_cfg_params.multiview_multiscale_deformable_attention_aggregation_path =
      multiview_multiscale_deformable_attention_aggregation_path;

  common::E2ETrtEngine sparse4d_extract_feat_trt_engine;
  sparse4d_extract_feat_trt_engine.engine_path = sparse4d_extract_feat_trt_engine_path;
  sparse4d_extract_feat_trt_engine.input_names = sparse4d_extract_feat_trt_engine_input_names;
  sparse4d_extract_feat_trt_engine.output_names = sparse4d_extract_feat_trt_engine_output_names;

  common::E2ETrtEngine sparse4d_head1st_engine;
  sparse4d_head1st_engine.engine_path = sparse4d_head1st_engine_path;
  sparse4d_head1st_engine.input_names = sparse4d_head1st_engine_input_names;
  sparse4d_head1st_engine.output_names = sparse4d_head1st_engine_output_names;

  common::E2ETrtEngine sparse4d_head2nd_engine;
  sparse4d_head2nd_engine.engine_path = sparse4d_head2nd_engine_path;
  sparse4d_head2nd_engine.input_names = sparse4d_head2nd_engine_input_names;
  sparse4d_head2nd_engine.output_names = sparse4d_head2nd_engine_output_names;

  common::InstanceBankParams instance_bank_params;
  instance_bank_params.num_querys = num_querys;
  instance_bank_params.query_dims = query_dims;
  instance_bank_params.kmeans_anchors = kmeans_anchors;
  instance_bank_params.topk_querys = topk_querys;
  instance_bank_params.max_time_interval = max_time_interval;
  instance_bank_params.default_time_interval = default_time_interval;
  instance_bank_params.confidence_decay = confidence_decay;

  common::PostprocessorParams postprocessor_params;
  postprocessor_params.post_process_out_nums = post_process_out_nums;
  postprocessor_params.post_process_threshold = post_process_threshold;

  const common::E2EParams params(preprocessor_params, model_cfg_params, sparse4d_extract_feat_trt_engine,
                                 sparse4d_head1st_engine, sparse4d_head2nd_engine, instance_bank_params,
                                 postprocessor_params);

  return params;
}  // namespace preprocessor

}  // namespace preprocessor
}  // namespace sparse_end2end