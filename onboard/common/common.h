// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_COMMON_COMMON_H
#define ONBOARD_COMMON_COMMON_H

#include <array>
#include <vector>

namespace sparse_end2end {
namespace common {

static constexpr const char* const kLogContext = "SparseE2ELog";

struct E2EEngine {
  std::string engine_path;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

enum class ObstacleType : std::uint8_t {
  CAR = 0U,
  TRCUK = 1U,
  CONSTRUCTION_VEHICLE = 2U,
  BUS = 3U,
  TRAILER = 4U,
  BARRIER = 5U,
  MOTORCYCLE = 6U,
  BICYCLE = 7U,
  PEDESTRIAN = 8U,
  TRAFFIC_CONE = 9U,
  OBSTACLETYPE_Max = 10U,
  OBSTACLETYPE_INVALID = 255U,
};

enum class CameraFrameID : std::uint8_t {
  CAM_FRONT = 0U,
  CAM_FRONT_RIGHT = 1U,
  CAM_FRONT_LEFT = 2U,
  CAM_BACK = 3U,
  CAM_BACK_LEFT = 4U,
  CAM_BACK_RIGHT = 5U,
  CameraFrameID_Max = 6U,
  CameraFrameID_INVALID = 255U,
};

struct Obstacle {
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
  float w = 0.0F;
  float l = 0.0F;
  float h = 0.0F;
  float yaw = 0.0F;
  float vx = 0.0F;
  float vy = 0.0F;
  float vz = 0.0F;
  float confidence = 0.0F;
  float fusioned_confidence = 0.0F;
  std::uint8_t label = 255U;
  std::int32_t trackid = -1;
};

enum class Status : std::uint8_t {
  kSuccess = 0U,

  kImgPreprocesSizeErr = 1U,

  kBackboneInferErr = 20U,

  kHead1stInferErr = 30U,
  kHead2ndInferErr = 31U,

  kDecoderErr = 50U
};

}  // namespace common
}  // namespace sparse_end2end
#endif  // ONBOARD_COMMON_COMMON_H
