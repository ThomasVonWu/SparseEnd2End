// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <chrono>

#include "utils.h"

namespace SparseEnd2End {
namespace common {

double ToSeconds(const Clock::time_point time) {
  using Second = std::chrono::duration<double, std::ratio<1, 1>>;
  const auto seconds =
      std::chrono::duration_cast<Second>(time.time_since_epoch());
  return seconds.count();
}

double ToSeconds(const Clock::duration& duration) {
  using Seconds = std::chrono::duration<double, std::ratio<1, 1>>;
  Seconds seconds{std::chrono::duration_cast<Seconds>(duration)};
  return seconds.count();
}

}  // namespace common
}  // namespace SparseEnd2End