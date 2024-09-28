// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_PREPROCESSOR_PARAMETERS_PARSER_H
#define ONBOARD_PREPROCESSOR_PARAMETERS_PARSER_H
#include <string>

#include "../common/parameter.h"

namespace sparse_end2end {
namespace processor {

/// @brief parse parameters from local YAML config.
/// @param model_root_dir: root directory of TensorRT files.
common::E2EParams parseParams(const std::string& model_root_dir, const std::string& vehicle_name);

}  // namespace processor
}  // namespace sparse_end2end

#endif  // ONBOARD_PREPROCESSOR_PARAMETERS_PARSER_H
