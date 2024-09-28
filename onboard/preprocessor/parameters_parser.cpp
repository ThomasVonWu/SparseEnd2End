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

  const std::map<std::string, common::CameraFrameID> camera_frameid_lut{
      {"CAM_FRONT", common::CameraFrameID::CAM_FRONT},
      {"CAM_FRONT_RIGHT", common::CameraFrameID::CAM_FRONT_RIGHT},
      {"CAM_FRONT_LEFT", common::CameraFrameID::CAM_FRONT_LEFT},
      {"CAM_BACK", common::CameraFrameID::CAM_BACK},
      {"CAM_BACK_LEFT", common::CameraFrameID::CAM_BACK_LEFT},
      {"CAM_BACK_RIGHT", common::CameraFrameID::CAM_BACK_RIGHT},
  };

  YAML::Node config_file_node = common::loadYamlFile(params_dir + "/config.yaml");
  YAML::Node model_node = common::getYamlSubNode(config_file_node, "e2e_model");
  YAML::Node calib_file_node = common::loadYamlFile(params_dir + "/calibration/calib.yaml");

  /// may be in model struct ?
  std::tuple < bool, std::vector<float>(ok, kmeans_anchor) =
                         common::read_wrapper<float>(params_dir + "/kmeans_anchor_900x11_float32.bin");
}

}  // namespace processor
}  // namespace sparse_end2end