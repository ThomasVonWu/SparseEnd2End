// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "tensorrt.h"

#include <dlfcn.h>

#include <iostream>

#include "../common/utils.h"
#include "logging.h"

namespace sparse_end2end {
namespace engine {

TensorRT::TensorRT(const std::string& engine_path,
                   const std::string& plugin_path,
                   const std::vector<std::string>& input_names,
                   const std::vector<std::string>& output_names)
    : engine_path_(engine_path), plugin_path_(plugin_path), input_names_(input_names), output_names_(output_names) {
  init();
}

TensorRT::~TensorRT() {
  context_->destroy();
  engine_->destroy();
  runtime_->destroy();
}

void TensorRT::init() {
  std::vector<char> engine_data = common::readfile_wrapper<char>(engine_path_);

  if (!plugin_path_.empty()) {
    void* pluginLibraryHandle = dlopen(plugin_path_.c_str(), RTLD_LAZY);
    if (!pluginLibraryHandle) {
      // LOG_ERROR(kLogContext, "Failed to load plugin: " << plugin_path_);
      std::cout << "[ERROR] Failed to load TensorRT plugin : " << plugin_path_ << std::endl;
    }
  }

  initLibNvInferPlugins(&gLogger, "");
  runtime_ = nvinfer1::createInferRuntime(gLogger);
  engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr);
  context_ = engine_->createExecutionContext();

  if (!runtime_ || !engine_ || !context_) {
    // LOG_ERROR(kLogContext, "TensorRT engine initialized failed!");
    std::cout << "[ERROR] TensorRT engine initialized failed !" << std::endl;
  }
}

bool TensorRT::infer(void* const* buffers, const cudaStream_t& stream) {
  return context_->enqueueV2(buffers, stream, nullptr);
}

bool TensorRT::setInputDimensions(const std::vector<std::vector<std::int32_t>>& input_dims) {
  if (input_dims.size() != input_names_.size()) {
    // LOG_ERROR(kLogContext, "Mismatched number of input dimensions!");
    std::cout << "[ERROR] Mismatched number of input dimensions ! " << std::endl;
    return false;
  }

  for (size_t i = 0; i < input_names_.size(); ++i) {
    const std::string& input_name = input_names_[i];
    const std::int32_t binding_index = engine_->getBindingIndex(input_name.c_str());
    nvinfer1::Dims dims = engine_->getBindingDimensions(binding_index);
    if (static_cast<size_t>(dims.nbDims) != input_dims[i].size()) {
      // LOG_ERROR(kLogContext, "Mismatched number of dimensions for input tensor: " << input_name);
      std::cout << "Mismatched number of dimensions for input tensor :  " << input_name << " "
                << static_cast<size_t>(dims.nbDims) << " v.s. " << input_dims[i].size() << std::endl;
      return false;
    }

    for (size_t j = 0; j < static_cast<size_t>(dims.nbDims); ++j) {
      dims.d[j] = input_dims[i][j];
    }

    if (!context_->setBindingDimensions(binding_index, dims)) {
      // LOG_ERROR(kLogContext, "Error binding input_name of " << input_name);
      std::cout << "[ERROR]  Failed to set binding dimensions for index  " << binding_index
                << "for input tensor : " << input_name << std::endl;
      return false;
    }
  }
  return true;
}

std::map<std::string, std::tuple<std::int32_t, std::int32_t>> TensorRT::getInputIndex() {
  std::map<std::string, std::tuple<std::int32_t, std::int32_t>> inputs_index_map;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    const std::string input_name = input_names_[i];
    const std::int32_t binding_index = engine_->getBindingIndex(input_name.c_str());
    const nvinfer1::Dims dims = engine_->getBindingDimensions(binding_index);
    std::int32_t tensor_length = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
      tensor_length *= dims.d[j];
    }
    inputs_index_map[input_name] = std::make_tuple(tensor_length, binding_index);
  }
  return inputs_index_map;
}

std::map<std::string, std::tuple<std::int32_t, std::int32_t>> TensorRT::getOutputIndex() {
  std::map<std::string, std::tuple<std::int32_t, std::int32_t>> outputs_index_map;
  for (size_t i = 0; i < output_names_.size(); ++i) {
    const std::string output_name = output_names_[i];
    const std::int32_t binding_index = engine_->getBindingIndex(output_name.c_str());
    const nvinfer1::Dims dims = engine_->getBindingDimensions(binding_index);
    std::int32_t tensor_length = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
      tensor_length *= dims.d[j];
    }
    outputs_index_map[output_name] = std::make_tuple(tensor_length, binding_index);
  }
  return outputs_index_map;
}

void TensorRT::getEngineInfo() {
  int numBindings = engine_->getNbBindings();
  for (int i = 0; i < numBindings; ++i) {
    bool isInput = engine_->bindingIsInput(i);
    std::string type = isInput ? "Input" : "Output";
    std::cout << type << " binding " << i << ": " << std::endl;
    std::cout << "  Name: " << engine_->getBindingName(i) << std::endl;
    nvinfer1::Dims dims = engine_->getBindingDimensions(i);
    std::cout << "  Dimensions: ";
    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j] << (j < dims.nbDims - 1 ? "x" : "");
    }
    std::cout << std::endl;
    nvinfer1::DataType dtype = engine_->getBindingDataType(i);
    std::cout << "  Data type: ";
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        std::cout << "FLOAT";
        break;
      case nvinfer1::DataType::kHALF:
        std::cout << "HALF";
        break;
      case nvinfer1::DataType::kINT8:
        std::cout << "INT8";
        break;
      case nvinfer1::DataType::kINT32:
        std::cout << "INT32";
        break;
      default:
        std::cout << "UNKNOWN";
        break;
    }
    std::cout << std::endl;
  }
}

}  // namespace engine
}  // namespace sparse_end2end