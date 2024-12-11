// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "parameters_parser.h"

#include <yaml-cpp/yaml.h>

#include <map>
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

  /// @brief STEP-1: Parse model config file from YAML file.
  YAML::Node config_file_node = common::loadYamlFile(model_cfg_path);

  YAML::Node offline_onboard_bin_file_path_node = common::getYamlSubNode(config_file_node, "OfflineOnboardBinFilePath");

  /// @brief STEP-2: Parse kmeans anchor from offline bin file.
  YAML::Node kmeans_anchors_path_node = common::getYamlSubNode(offline_onboard_bin_file_path_node, "KmeansAnchorsPath");
  std::string kmeans_anchors_path = kmeans_anchors_path_node.as<std::string>();
  std::vector<float> kmeans_anchors = common::read_wrapper<float>(kmeans_anchors_path);

  /// @brief STEP-3: Parse calibration parameters:
  YAML::Node lidar2image_path_node = common::getYamlSubNode(offline_onboard_bin_file_path_node, "Lidar2ImagePath");
  std::string lidar2image_path = lidar2image_path_node.as<std::string>();
  std::vector<float> lidar2img = common::read_wrapper(lidar2image_path);

  YAML::Node sparse_e2e_node = common::getYamlSubNode(config_file_node, "SparseE2E");

  /// @brief STEP-4: Parse Node TaskCameraFrameID.
  std::vector<common::CameraFrameID> task_camera_frame_id;
  YAML::Node task_camera_frame_id_node = getYamlSubNode(sparse_e2e_node, "TaskCameraFrameID");
  std::vector<std::string> task_camera_frame_id_strs = task_camera_frame_id_node.as<std::vector<std::string>>();
  size_t num_cams = task_camera_frame_id_strs.size();

  /// @brief STEP-5: Parse Node: `ImgPreprocessor` in YAML.
  YAML::Node img_preprocessor_node = common::getYamlSubNode(sparse_e2e_node, "ImgPreprocessor");

  /// STEP-5.1: Parse Node: `ImgPreprocessor/RawImgShape_CHW` in YAML.
  YAML::Node raw_img_shape_chw_node = getYamlSubNode(img_preprocessor_node, "RawImgShape_CHW");
  std::vector<std::uint32_t> raw_img_shape_chw = raw_img_shape_chw_node.as<std::vector<std::uint32_t>>();

  /// STEP-5.2: Parse Node: `ImgPreprocessor/ModelInputImgShape_CHW` in YAML.
  std::vector<std::uint32_t> model_input_img_shape_chw_tmp;
  YAML::Node model_input_img_shape_chw_node = common::getYamlSubNode(img_preprocessor_node, "ModelInputImgShape_CHW");
  std::vector<std::uint32_t> model_input_img_shape_chw =
      model_input_img_shape_chw_node.as<std::vector<std::uint32_t>>();
  model_input_img_shape_chw_tmp.assign(model_input_img_shape_chw.begin(), model_input_img_shape_chw.end());

  /// @attention Calculate image preprocess's parameters: `crop_height/crop_width/resize_ratio` based on the given
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
  float resize_ratio =
      std::max(static_cast<float>(model_input_img_shape_chw[1]) / static_cast<float>(raw_img_shape_chw[1]),
               static_cast<float>(model_input_img_shape_chw[2]) / static_cast<float>(raw_img_shape_chw[2]));

  /// @brief STEP-6: Parse Node: `EmbedFeatDims` in YAML.
  YAML::Node embedfeat_dims_node = common::getYamlSubNode(sparse_e2e_node, "EmbedFeatDims");
  std::uint32_t embedfeat_dims = embedfeat_dims_node.as<std::uint32_t>();

  /// @brief STEP-7: Parse Node: `ModelExtractFeatTrtEngine` in YAML.
  YAML::Node model_extract_feat_trt_engine_node = common::getYamlSubNode(sparse_e2e_node, "ModelExtractFeatTrtEngine");

  /// STEP-7.1: Parse Node: `ModelExtractFeatTrtEngine/EnginePath` in YAML.
  YAML::Node sparse4d_backbone_engine_path_node =
      common::getYamlSubNode(model_extract_feat_trt_engine_node, "EnginePath");
  std::string sparse4d_backbone_engine_path = sparse4d_backbone_engine_path_node.as<std::string>();

  /// STEP-7.2 : Parse Node: `ModelExtractFeatTrtEngine/EngineInputNames` in YAML.
  YAML::Node sparse4d_backbone_engine_input_names_node =
      common::getYamlSubNode(model_extract_feat_trt_engine_node, "EngineInputNames");
  std::vector<std::string> sparse4d_backbone_engine_input_names =
      sparse4d_backbone_engine_input_names_node.as<std::vector<std::string>>();

  /// STEP-7.3 : Parse Node: `ModelExtractFeatTrtEngine/EngineOutputNames` in YAML.
  YAML::Node sparse4d_backbone_engine_output_names_node =
      common::getYamlSubNode(model_extract_feat_trt_engine_node, "EngineOutputNames");
  std::vector<std::string> sparse4d_backbone_engine_output_names =
      sparse4d_backbone_engine_output_names_node.as<std::vector<std::string>>();

  /// STEP-7.4: Parse Node: `ModelExtractFeatTrtEngine/ModelExtractFeatShape_LC` in YAML.
  YAML::Node model_extract_feat_shape_lc_node =
      common::getYamlSubNode(img_preprocessor_node, "ModelExtractFeatShape_LC");
  std::vector<std::uint32_t> model_extract_feat_shape_lc =
      model_extract_feat_shape_lc_node.as<std::vector<std::uint32_t>>();

  /// STEP-7.5 Parse Node: `ModelExtractFeatTrtEngine/ModelExtractFeatSpatialShapes_LD` in YAML.
  YAML::Node model_extract_feat_spatial_shapes_ld_node =
      common::getYamlSubNode(img_preprocessor_node, "ModelExtractFeatSpatialShapes_LD");
  std::vector<std::int32_t> model_extract_feat_spatial_shapes_ld =
      model_extract_feat_spatial_shapes_ld_node.as<std::vector<std::inst32_t>>();

  /// Get model_extract_feat_level_start_index based on model_extract_feat_spatial_shapes_ld.
  std::vector<std::int32_t> model_extract_feat_spatial_shapes_nld{};
  for (size_t i = 0; i < num_cams; ++i) {
    model_extract_feat_spatial_shapes_nld.insert(
        model_extract_feat_spatial_shapes_nld.end(),
        {model_extract_feat_spatial_shapes_ld[0], model_extract_feat_spatial_shapes_ld[1],
         model_extract_feat_spatial_shapes_ld[2], model_extract_feat_spatial_shapes_ld[3],
         model_extract_feat_spatial_shapes_ld[4], model_extract_feat_spatial_shapes_ld[5],
         model_extract_feat_spatial_shapes_ld[6], model_extract_feat_spatial_shapes_ld[7]});
  }

  std::vector<std::int32_t> model_extract_feat_level_start_index_tmp{};
  for (size_t i = 0; i < num_cams; ++i) {
    model_extract_feat_level_start_index_tmp.insert(
        model_extract_feat_level_start_index_tmp.end(),
        {model_extract_feat_spatial_shapes_nld[0] * model_extract_feat_spatial_shapes_nld[1],
         model_extract_feat_spatial_shapes_nld[2] * model_extract_feat_spatial_shapes_nld[3],
         model_extract_feat_spatial_shapes_nld[4] * model_extract_feat_spatial_shapes_nld[5],
         model_extract_feat_spatial_shapes_nld[6] * model_extract_feat_spatial_shapes_nld[7]});
  }
  std::vector<std::int32_t> model_extract_feat_level_start_index(model_extract_feat_level_start_index_tmp.size());
  std::partial_sum(model_extract_feat_level_start_index_tmp.begin(), model_extract_feat_level_start_index_tmp.end(),
                   model_extract_feat_level_start_index.begin());
  model_extract_feat_level_start_index.pop_back();
  model_extract_feat_level_start_index.insert(model_extract_feat_level_start_index.begin(), 0);

  /// @brief STEP-8 Parse Node: `ModelHeadFirstFrameEngine` in YAML.
  YAML::Node model_head_first_frame_engine_node = common::getYamlSubNode(sparse_e2e_node, "ModelHeadFirstFrameEngine");

  /// STEP-8.1 Parse Node:`ModelHeadFirstFrameEngine/EnginePath` in YAML.
  YAML::Node sparse4d_head1st_engine_path_node =
      common::getYamlSubNode(model_head_first_frame_engine_node, "EnginePath");
  std::string sparse4d_head1st_engine_path = sparse4d_head1st_engine_path_node.as<std::string>();

  /// STEP-8.2 : Parse Node: `ModelHeadFirstFrameEngine/EngineInputNames` in YAML.
  YAML::Node sparse4d_head1st_engine_input_names_node =
      common::getYamlSubNode(model_head_first_frame_engine_node, "EngineInputNames");
  std::vector<std::string> sparse4d_head1st_engine_input_names =
      sparse4d_head1st_engine_input_names_node.as<std::vector<std::string>>();

  /// STEP-8.3 : Parse Node: `ModelHeadFirstFrameEngine/EngineOutputNames` in YAML.
  YAML::Node sparse4d_head1st_engine_output_names_node =
      common::getYamlSubNode(model_head_first_frame_engine_node, "EngineOutputNames");
  std::vector<std::string> sparse4d_head1st_engine_output_names =
      sparse4d_head1st_engine_output_names_node.as<std::vector<std::string>>();

  /// @brief STEP-9 Parse Node: `ModelHeadSecondFrameEngine` in YAML.
  YAML::Node model_head_second_frame_engine_node =
      common::getYamlSubNode(sparse_e2e_node, "ModelHeadSecondFrameEngine");

  /// STEP-9.1 Parse Node:`ModelHeadSecondFrameEngine/EnginePath` in YAML.
  YAML::Node sparse4d_head2nd_engine_path_node =
      common::getYamlSubNode(model_head_second_frame_engine_node, "EnginePath");
  std::string sparse4d_head2nd_engine_path = sparse4d_head2nd_engine_path_node.as<std::string>();

  /// STEP-9.2 : Parse Node: `ModelHeadSecondFrameEngine/EngineInputNames` in YAML.
  YAML::Node sparse4d_head2nd_engine_input_names_node =
      common::getYamlSubNode(model_head_second_frame_engine_node, "EngineInputNames");
  std::vector<std::string> sparse4d_head2nd_engine_input_names =
      sparse4d_head2nd_engine_input_names_node.as<std::vector<std::string>>();

  /// STEP-9.3 : Parse Node: `ModelHeadSecondFrameEngine/EngineOutputNames` in YAML.
  YAML::Node sparse4d_head2nd_engine_output_names_node =
      common::getYamlSubNode(model_head_second_frame_engine_node, "EngineOutputNames");
  std::vector<std::string> sparse4d_head2nd_engine_output_names =
      sparse4d_head2nd_engine_output_names_node.as<std::vector<std::string>>();

  ///@brief STEP-10 Parse Node: `InstanceBankParams` in YAML.
  YAML::Node instance_bank_params_node = common::getYamlSubNode(sparse_e2e_node, "InstanceBankParams");

  YAML::Node num_querys_node = common::getYamlSubNode(instance_bank_params_node, "NumQuerys");
  std::uint32_t num_querys = num_querys_node.as<std::uint32_t>();

  YAML::Node query_dims_node = common::getYamlSubNode(instance_bank_params_node, "QueryDims");
  std::uint32_t query_dims = query_dims_node.as<std::uint32_t>();

  YAML::Node topk_querys_node = common::getYamlSubNode(instance_bank_params_node, "TopKQuerys");
  std::uint32_t topk_querys = topk_querys_node.as<std::uint32_t>();

  YAML::Node max_time_interval_node = common::getYamlSubNode(instance_bank_params_node, "MaxTimeInterval");
  float max_time_interval = max_time_interval_node.as<float>();

  YAML::Node default_time_interval_node = common::getYamlSubNode(instance_bank_params_node, "DefaultTimeInterval");
  float default_time_interval = default_time_interval_node.as<float>();

  YAML::Node confidence_decay_node = common::getYamlSubNode(instance_bank_params_node, "ConfidenceDecay");
  float confidence_decay = confidence_decay_node.as<float>();

  ///@brief STEP-11 Parse Node: `PostProcess` in YAML.
  YAML::Node post_process_node = common::getYamlSubNode(sparse_e2e_node, "PostProcess");

  YAML::Node post_process_out_nums_node = common::getYamlSubNode(post_process_node, "PostProcessOutNums");
  std::uint32_t post_process_out_nums = post_process_out_nums_node.as<std::uint32_t>();

  YAML::Node post_process_threshold_node = common::getYamlSubNode(instance_bank_params_node, "PostProcessThreshold");
  float post_process_threshold = post_process_threshold_node.as<float>();

  ///@brief STEP-12 Construct E2EParams::params based on the the parsed parameters.
  common::PreprocessorParams preprocessor_params;
  preprocessor_params.num_cams = num_cams;
  preprocessor_params.raw_img_c = raw_img_shape_chw[0];
  preprocessor_params.raw_img_h = raw_img_shape_chw[1];
  preprocessor_params.raw_img_w = raw_img_shape_chw[2];
  preprocessor_params.model_input_img_c = model_input_img_shape_chw_tmp[0];
  preprocessor_params.model_input_img_h = model_input_img_shape_chw_tmp[1];
  preprocessor_params.model_input_img_w = model_input_img_shape_chw_tmp[2];

  common::E2ETrtEngine& sparse4d_backbone_engine;
  sparse4d_backbone_engine.engine_path = sparse4d_backbone_engine_path;
  sparse4d_backbone_engine.input_names = sparse4d_backbone_engine_input_names;
  sparse4d_backbone_engine.output_names = sparse4d_backbone_engine_output_names;

  common::E2ETrtEngine& sparse4d_head1st_engine;
  sparse4d_head1st_engine.engine_path = sparse4d_head1st_engine_path;
  sparse4d_head1st_engine.input_names = sparse4d_head1st_engine_input_names;
  sparse4d_head1st_engine.output_names = sparse4d_head1st_engine_output_names;

  common::E2ETrtEngine& sparse4d_head2nd_engine;
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
  post_process_out_nums.post_process_threshold = post_process_threshold;

  const common::E2EParams params(preprocessor_params, sparse4d_backbone_engine, sparse4d_head1st_engine,
                                 sparse4d_head2nd_engine, instance_bank_params, postprocessor_params);

  return params;
}  // namespace preprocessor

}  // namespace preprocessor
}  // namespace sparse_end2end