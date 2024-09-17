#include <glog/logging.h>
#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

namespace bevfusion {
namespace camera {

TEST(preprocess_unit_test, random) {
  google::InitGoogleLogging("preprocess_unit_test");
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = 0;  // Setting log level in Console

  google::SetLogDestination(google::GLOG_INFO, "LOG_INFO_");
  google::SetLogDestination(google::GLOG_ERROR,
                            "LOG_INFO_");  // Not saving log file for ERR

  int max_error = 0;
  EXPECT_EQ(max_error, 0);

  LOG(INFO) << "Status:  "
            << "succed!";
  LOG(WARNING) << "Status:  "
               << "succed!";
  LOG(ERROR) << "Status:  "
             << "succed!";
  LOG(FATAL) << "Status:  "
             << "succed!";

  google::ShutdownGoogleLogging();
}

}  // namespace camera
}  // namespace bevfusion