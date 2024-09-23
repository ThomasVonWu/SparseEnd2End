#ifndef ONBOARD_COMMON_UTILS_H
#define ONBOARD_COMMON_UTILS_H

#include <fstream>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace sparse_end2end {
namespace common {

template <typename T>
std::tuple<bool, std::vector<T>> read_wrapper(const std::string& filename) {
  std::ifstream file(filename, std::ios::in);
  if (!file) {
    throw "[ERROR] read file failed";
    return std::make_tuple(false, std::vector<T>{});
  }

  file.seekg(0, std::ifstream::end);
  auto fsize = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ifstream::beg);

  std::vector<T> buffer(static_cast<size_t>(fsize) / sizeof(T));
  file.read((char*)buffer.data(), static_cast<std::streamsize>(fsize));
  file.close();

  return std::make_tuple(true, buffer);
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

}  // namespace common
}  // namespace sparse_end2end

#endif  // ONBOARD_COMMON_UTILS_H
