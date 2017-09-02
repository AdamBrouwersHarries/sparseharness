#pragma once

#include "common.h"
#include <chrono>
#include <ctime>
#include <iostream>

// #define TREE_PERF

#define start_timer(name, context)                                             \
  auto _csds_timer_name_context = CSDSTimer(#name, #context, std::cerr);

#define report_timing(name, context, time)                                     \
  CSDSTimer::reportTiming(#name, #context, std::chrono::nanoseconds(time));

using clock_type = std::chrono::system_clock;

class CSDSTimer {

public:
  // Construct and start the timer
  // CSDSTimer(std::string name, std::string context = "global",
  //           std::ostream &stream = std::cout);
  CSDSTimer(const std::string &name);
  CSDSTimer(const std::string &name, const std::string &context);
  CSDSTimer(const std::string &name, const std::string &context,
            std::ostream &stream);

  // Destruct and stop the timer, reporting the elapseed time
  ~CSDSTimer();
  // static void SetStream(std::ostream &str) { _default_str = &str; }

  // report timing measured by other means (e.g. from OpenCL profiling
  // information)
  static void reportTiming(const std::string &name, const std::string &context,
                           std::chrono::nanoseconds nanoseconds);

private:
  void reportStart();
  void reportEnd();
  std::string _name;
  std::string _context;

  std::ostream *_default_str;
  std::chrono::time_point<clock_type> _start;

  // static std::ostream *_default_str;
};