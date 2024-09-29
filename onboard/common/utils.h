#ifndef ONBOARD_COMMON_UTILS_H
#define ONBOARD_COMMON_UTILS_H

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace sparse_end2end {
namespace common {

template <typename T>
std::vector<T> read_wrapper(const std::string& filename) {
  std::ifstream file(filename, std::ios::in);
  if (!file) {
    throw "[ERROR] read file failed";
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
void write_wrapper(const std::vector<T>& data, const std::string& filename) {
  std::ofstream file;
  file.open(filename, std::ios::out);
  if (!file.is_open()) {
    throw "[ERROR] open file failed";
    return;
  }
  file.write((char*)data.data(), static_cast<std::streamsize>(data.size()) * sizeof(T));
  file.close();
}

YAML::Node loadYamlFile(const std::string& file_path) {
  try {
    YAML::Node node = YAML::LoadFile(file_path);
    if (!node) {
      // LOG_ERROR(kLogContext, "Loaded node is null: " + file_path);
      exit(EXIT_FAILURE);
      return YAML::Node();
    }
    return node;
  } catch (const YAML::Exception& e) {
    std::string errorMessage = e.what();
    // LOG_ERROR(kLogContext, "Failed to load YAML file: " + file_path + "->" + errorMessage);
    exit(EXIT_FAILURE);
    return YAML::Node();
  }
}

template <typename T>
inline void readYamlNode(const YAML::Node& yaml, const std::string& key, T& out_val) {
  if (!yaml[key] || yaml[key].Type() == YAML::NodeType::Null) {
    // LOG_ERROR(kLogContext, "Yaml file not set " << key << " value, Aborting!!!");
    exit(EXIT_FAILURE);
  } else {
    try {
      out_val = yaml[key].as<T>();
    } catch (const YAML::BadConversion& e) {
      std::string errorMessage = e.what();
      // LOG_ERROR(kLogContext, "Failed to convert YAML key " << key << " value, Aborting!!!");
      exit(EXIT_FAILURE);
    }
  }
}

YAML::Node getYamlSubNode(const YAML::Node& yaml, const std::string& node) {
  YAML::Node ret = yaml[node.c_str()];
  if (ret && ret.Type() != YAML::NodeType::Null) {
    return ret;
  } else {
    // LOG_ERROR(kLogContext, " : Failed to  get  YAML  subnode " << node << ". Aborting!!!");
    exit(EXIT_FAILURE);
  }
}

void fileExists(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.good()) {
    // LOG_ERROR(kLogContext, "Failed to Open file: " + filename);
    exit(EXIT_FAILURE);
  }
}

}  // namespace common
}  // namespace sparse_end2end

#endif  // ONBOARD_COMMON_UTILS_H
