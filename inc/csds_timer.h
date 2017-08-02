#pragma once

#include <chrono>
#include <ctime>
#include <iostream>

#define TREE_PERF

#define start_timer(name, context)                                             \
  auto _csds_timer = CSDSTimer(#name, #context);

class CSDSTimer {

public:
  // Construct and start the timer
  // CSDSTimer(std::string name, std::string context = "global",
  //           std::ostream &stream = std::cout);
  CSDSTimer(std::string name);
  CSDSTimer(std::string name, std::string context);
  CSDSTimer(std::string name, std::string context, std::ostream &stream);

  // Destruct and stop the timer, reporting the elapseed time
  ~CSDSTimer();
  // static void SetStream(std::ostream &str) { _default_str = &str; }

private:
  void reportStart();
  void reportEnd();
  std::string _name;
  std::string _context;
  std::chrono::time_point<std::chrono::system_clock> _start, _end;
  std::ostream *_default_str;

  // static std::ostream *_default_str;
};