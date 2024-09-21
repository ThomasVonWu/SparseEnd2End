// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "tensorrt.h"

#include <dlfcn.h>

namespace SparseEnd2End {

TensorRT::TensorRT(const std::string& engine_path,
                   const std::vector<std::string>& input_names,
                   const std::vector<std::string>& output_names)
    : engine_path_(engine_path),
      input_names_(input_names),
      output_names_(output_names) {}
TensorRT::~TensorRT() {
  context_->destroy();
  engine_->destroy();
  runtime_->destroy();
}

bool TensorRT::Init() {
  auto [status, engine_data] = deserializate<char>(engine_path_);
  if (!ok) {
    LOG_ERROR(kLogContext,
              "Failed to find TensorRT engine file." << engine_path_);
    return false;
  }

  void* pluginLibraryHandle = dlopen(plugin_path_, RTLD_LAZY);
  if (!pluginLibraryHandle) {
    LOG_ERROR(kLogContext, "Failed to load plugin: " << plugin_path_);
    return false;
  }

  initLibNvInferPlugins(&gLogger, "");
  runtime_ = nvinfer1::createInferRuntime(gLogger);
  engine_ = runtime_->deserializeCudaEngine(engine_data.data(),
                                            engine_data.size(), nullptr);
  context_ = engine_->createExecutionContext();

  if (!runtime_ || !engine_ || !context_) {
    LOG_ERROR(kLogContext, "TensorRT engine initialized failed!");
    return false;
  }
  return true;
}

bool TensorRT::Infer(void* const* buffers, const cudaStream_t& stream) {
  return context_->enqueueV2(buffers, stream, nullptr);
}

bool TensorRT::SetInputDimensions(
    const std::vector<std::vector<std::int32_t>>& input_dims) {
  if (input_dims.size() != input_names_.size()) {
    LOG_ERROR(kLogContext, "Mismatched number of input dimensions!");
    return false;
  }

  for (size_t i = 0; i < input_names_.size(); ++i) {
    const std::string& input_name = input_names_[i];
    const std::int32_t binding_index =
        engine_->getBindingIndex(input_name.c_str());
    nvinfer1::Dims dims = engine_->getBindingDimensions(binding_index);
    if (static_cast<size_t>(dims.nbDims) != input_dims[i].size()) {
      LOG_ERROR(
          kLogContext,
          "Mismatched number of dimensions for input tensor: " << input_name);
      return false;
    }

    for (size_t j = 0; j < static_cast<size_t>(dims.nbDims); ++j) {
      dims.d[j] = input_dims[i][j];
    }
    if (!context_->setBindingDimensions(binding_index, dims)) {
      LOG_ERROR(kLogContext, "Error binding input_name of " << input_name);
      return false;
    }
  }
  return true;
}

std::map<std::string, std::tuple<std::int32_t, std::int32_t>>
TensorRT::GetInputIndex() {
  std::map<std::string, std::tuple<std::int32_t, std::int32_t>>
      inputs_index_map;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    const std::string input_name = input_names_[i];
    const std::int32_t binding_index =
        engine_->getBindingIndex(input_name.c_str());
    const nvinfer1::Dims dims = engine_->getBindingDimensions(binding_index);
    std::int32_t tensor_length = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
      tensor_length *= dims.d[j];
    }
    inputs_index_map[input_name] =
        std::make_tuple(tensor_length, binding_index);
  }
  return inputs_index_map;
}

std::map<std::string, std::tuple<std::int32_t, std::int32_t>>
TensorRT::GetOutputIndex() {
  std::map<std::string, std::tuple<std::int32_t, std::int32_t>>
      outputs_index_map;
  for (size_t i = 0; i < output_names_.size(); ++i) {
    const std::string output_name = output_names_[i];
    const std::int32_t binding_index =
        engine_->getBindingIndex(output_name.c_str());
    const nvinfer1::Dims dims = engine_->getBindingDimensions(binding_index);
    std::int32_t tensor_length = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
      tensor_length *= dims.d[j];
    }
    outputs_index_map[output_name] =
        std::make_tuple(tensor_length, binding_index);
  }
  return outputs_index_map;
}

void getEngineInfo() {
  int numBindings = engine_->getNbBindings();
  for (int i = 0; i < numBindings; ++i) {
    bool isInput = engine_->bindingIsInput(i);
    std::string type = isInput ? "Input" : "Output";
    LOG_INFO << type << " binding " << i << ": " << std::endl;
    LOG_INFO << "  Name: " << engine_->getBindingName(i) << std::endl;
    nvinfer1::Dims dims = engine_->getBindingDimensions(i);
    LOG_INFO << "  Dimensions: ";
    for (int j = 0; j < dims.nbDims; ++j) {
      LOG_INFO << dims.d[j] << (j < dims.nbDims - 1 ? "x" : "");
    }
    LOG_INFO << std::endl;
    nvinfer1::DataType dtype = engine_->getBindingDataType(i);
    LOG_INFO << "  Data type: ";
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        LOG_INFO << "FLOAT";
        break;
      case nvinfer1::DataType::kHALF:
        LOG_INFO << "HALF";
        break;
      case nvinfer1::DataType::kINT8:
        LOG_INFO << "INT8";
        break;
      case nvinfer1::DataType::kINT32:
        LOG_INFO << "INT32";
        break;
      default:
        LOG_INFO << "UNKNOWN";
        break;
    }
    LOG_INFO << std::endl;
  }
}

}  // namespace SparseEnd2End