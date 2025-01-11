#ifndef ONBOARD_TENSORRT_LOGGING_H
#define ONBOARD_TENSORRT_LOGGING_H

#include <NvInfer.h>

#include <cassert>
#include <iostream>
#include <map>

using namespace nvinfer1;
using Severity = nvinfer1::ILogger::Severity;

class Logger : public ILogger {
 public:
  Logger(Severity severity) : mOstream(&std::cout), mReportableSeverity(severity) {}

  template <typename T>
  Logger& operator<<(T const& obj) {
    if (mOstream != nullptr) {
      *mOstream << obj;
    }
    return *this;
  }

  Logger& report(Severity severity, const char* msg) {
    if (severity <= mReportableSeverity) {
      const std::map<Severity, std::string> prefixMapping = {{Severity::kINTERNAL_ERROR, "[Sparse4dTrtLog][F] "},
                                                             {Severity::kERROR, "[Sparse4dTrtLogTrtLog][E] "},
                                                             {Severity::kWARNING, "[Sparse4dTrtLog][W] "},
                                                             {Severity::kINFO, "[Sparse4dTrtLog][I] "},
                                                             {Severity::kVERBOSE, "[Sparse4dTrtLog][V] "}};

      assert(prefixMapping.find(severity) != prefixMapping.end());

      mOstream = &std::cout;

      *this << prefixMapping.at(severity) << msg;

      return *this;
    }
    mOstream = nullptr;
    return *this;
  }

 private:
  void log(Severity severity, const char* msg) noexcept override { report(severity, msg) << "\n"; }

  std::ostream* mOstream;
  Severity mReportableSeverity;
};

extern Logger gLogger;
#define gLogFatal gLogger.report(Severity::kINTERNAL_ERROR, "")
#define gLogError gLogger.report(Severity::kERROR, "")
#define gLogWarning gLogger.report(Severity::kWARNING, "")
#define gLogInfo gLogger.report(Severity::kINFO, "")
#define gLogVerbose gLogger.report(Severity::kVERBOSE, "")

#endif  // ONBOARD_TENSORRT_LOGGING_H
