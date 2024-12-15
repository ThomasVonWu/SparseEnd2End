// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_COMMON_UTILS_H
#define ONBOARD_COMMON_UTILS_H

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace sparse_end2end {
namespace common {

template <typename T>
std::vector<T> readfile_wrapper(const std::string& filename) {
  std::ifstream file(filename, std::ios::in);
  if (!file) {
    std::cout << "Read file failed : " << filename << std::endl;
    throw "[ERROR] Read file failed";
    // LOG_ERROR(kLogContext, "Read file failed: " + file_path);
    return std::vector<T>{};
  }

  file.seekg(0, std::ifstream::end);
  auto fsize = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ifstream::beg);

  std::vector<T> buffer(static_cast<size_t>(fsize) / sizeof(T));
  file.read((char*)buffer.data(), static_cast<std::streamsize>(fsize));
  file.close();

  return buffer;
}

template <typename T>
void writefile_wrapper(const std::vector<T>& data, const std::string& filename) {
  std::ofstream file;
  file.open(filename, std::ios::out);
  if (!file.is_open()) {
    throw "[ERROR] Open file failed";
    return;
  }
  file.write((char*)data.data(), static_cast<std::streamsize>(data.size()) * sizeof(T));
  file.close();
}

YAML::Node loadYamlFile(const std::string& file_path) {
  try {
    YAML::Node node = YAML::LoadFile(file_path);
    if (!node) {
      std::cout << "Loaded node is null : " << file_path << std::endl;
      // LOG_ERROR(kLogContext, "Loaded node is null: " + file_path);
      exit(EXIT_FAILURE);
      return YAML::Node();
    }
    return node;
  } catch (const YAML::Exception& e) {
    std::string errorMessage = e.what();
    std::cout << errorMessage << "| Failed to load YAML file: " << file_path << std::endl;
    // LOG_ERROR(kLogContext, "Failed to load YAML file: " + file_path + "->" + errorMessage);
    throw "[ERROR] Load YAML file failed! ";  // 抛出异常，推出函数，执行后续程序
    // exit(EXIT_FAILURE);                       // 直接推出程序
    return YAML::Node();
  }
}

/// @brief Read Yaml key(Node) and get one value assigned to variable `out_val`.
template <typename T>
inline void readYamlNode(const YAML::Node& yaml, const std::string& key, T& out_val) {
  if (!yaml[key] || yaml[key].Type() == YAML::NodeType::Null) {
    std::cout << "Yaml file not set : " << key << std::endl;
    // LOG_ERROR(kLogContext, "Yaml file not set " << key << " value, Aborting!!!");
    exit(EXIT_FAILURE);
  } else {
    try {
      out_val = yaml[key].as<T>();
    } catch (const YAML::BadConversion& e) {
      std::string errorMessage = e.what();
      std::cout << "Failed to convert YAML key : " << key << std::endl;
      // LOG_ERROR(kLogContext, "Failed to convert YAML key " << key << " value, Aborting!!!");
      exit(EXIT_FAILURE);
    }
  }
}

/// @brief Read Yaml Node(key) and get sequence values push into STL `vector`.
template <typename T>
inline void readYamlNode(const YAML::Node& yaml, const std::string& key, std::vector<T>& out_vals) {
  if (!yaml[key]) {
    std::cout << "Yaml file not set : " << key << std::endl;
    // LOG_ERROR(kLogContext, "Yaml file not set " << key << " Aborting!!!");
    exit(EXIT_FAILURE);
  }
  if (yaml[key].IsSequence() && yaml[key].size() == out_vals.size()) {
    std::cout << "Yaml key node:" << key << " must be sequence and it's length must be the same with output vector!"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0, len = yaml[key].size(); i < len; ++i) {
    out_vals[i] = yaml[key].as<T>();
  }
}

YAML::Node getYamlSubNode(const YAML::Node& yaml, const std::string& node) {
  YAML::Node ret = yaml[node.c_str()];
  if (ret && ret.Type() != YAML::NodeType::Null) {
    return ret;
  } else {
    std::cout << "Failed to get YAML subnode : " << node << std::endl;
    // LOG_ERROR(kLogContext, " : Failed to  get  YAML  subnode " << node << ". Aborting!!!");
    exit(EXIT_FAILURE);
  }
}

void fileExists(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.good()) {
    std::cout << "Failed to Open file : " << filename << std::endl;
    // LOG_ERROR(kLogContext, "Failed to Open file: " + filename);
    exit(EXIT_FAILURE);
  }
}

}  // namespace common
}  // namespace sparse_end2end

#endif  // ONBOARD_COMMON_UTILS_H
