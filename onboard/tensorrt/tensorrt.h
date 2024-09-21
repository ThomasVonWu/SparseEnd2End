// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_TENSORRT_TENSORRT_H
#define ONBOARD_TENSORRT_TENSORRT_H

#include <map>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"

namespace SparseEnd2End {

class TensorRT {
 public:
  TensorRT(const std::string& engine_path,
           const std::vector<std::string>& input_names,
           const std::vector<std::string>& output_names);
  TensorRT() = delete;
  ~TensorRT();

  bool init();
  bool setInputDimensions(
      const std::vector<std::vector<std::int32_t>>& input_dims);
  bool infer(void* const* buffers, const cudaStream_t& stream);
  std::map<std::string, std::tuple<std::int32_t, std::int32_t>> getInputIndex();
  std::map<std::string, std::tuple<std::int32_t, std::int32_t>>
  getOutputIndex();
  void getEngineInfo();

 private:
  const std::string engine_path_;
  const std::string plugin_path_;
  const std::vector<std::string> input_names_;
  const std::vector<std::string> output_names_;
  nvinfer1::IRuntime* runtime_;
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IExecutionContext* context_;
};

}  // namespace SparseEnd2End

#endif  // ONBOARD_TENSORRT_TENSORRT_H
