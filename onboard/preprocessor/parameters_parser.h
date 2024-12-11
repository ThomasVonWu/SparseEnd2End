// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_PREPROCESSOR_PARAMETERS_PARSER_H
#define ONBOARD_PREPROCESSOR_PARAMETERS_PARSER_H
#include <string>

#include "../common/parameter.h"

namespace sparse_end2end {
namespace preprocessor {

/// @brief Parse parameters from local YAML config.
/// @param model_cfg_path: Offline onboard assets: model config path.
common::E2EParams parseParams(const std::string& model_cfg_path);

}  // namespace preprocessor
}  // namespace sparse_end2end

#endif  // ONBOARD_PREPROCESSOR_PARAMETERS_PARSER_H
