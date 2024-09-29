// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "parameters_parser.h"

#include <yaml-cpp/yaml.h>

#include <map>

#include "../common/common.h"
#include "../common/utils.h"
namespace sparse_end2end {
namespace processor {

common::E2EParams parseParams(const std::string& params_dir, const std::string& vehicle_name) {
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

  YAML::Node config_file_node = common::loadYamlFile(params_dir + "/model_cfg.yaml");
  YAML::Node model_node = common::getYamlSubNode(config_file_node, "SparseE2E");
  YAML::Node calib_file_node = common::loadYamlFile(params_dir + "/calib_params.yaml");

  /// may be in model struct ?
  std::vector<float>(success_status, kmeans_anchor) =
      common::read_wrapper<float>(params_dir + "/kmeans_anchor_900x11_float32.bin");

  /// Parse task camera frameid from YAML.
  std::vector<common::CameraFrameID> task_camera_frame_id;
  YAML::Node task_camera_frame_id_node = getYamlSubNode(e2e_model_node, "TaskCameraFrameID");
  std::string task_camera_frame_id_str = task_camera_frame_id_node.as<std::vector<std::string>>();
  for (auto& camera_name_str : task_camera_frame_id_str) {
    task_camera_frame_id.emplace_back(camera_frameid_lut.at(camera_name_str));
  }

  /// Parse task camera frameid from YAML.
}

}  // namespace processor
}  // namespace sparse_end2end