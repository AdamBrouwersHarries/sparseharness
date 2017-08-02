#pragma once

#include <chrono>
#include <ctime>
#include <iostream>

class CSDSTimer {

public:
  // TODO: add stream changing
  CSDSTimer(std::string name, std::string context = "global",
            std::ostream &stream = std::cout)
      : _name(name), _context(context), _default_str(&stream) {
    _start = std::chrono::system_clock::now();
  }
  ~CSDSTimer() {
    _end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = _end - _start;
    *_default_str << "PROFILING_DATUM(\"" << _name << "\", \"" << _context
                  << "\", " << elapsed_seconds.count() * 1000 << ", \"C++\")"
                  << std::endl;
  }
  // static void SetStream(std::ostream &str) { _default_str = &str; }

private:
  std::string _name;
  std::string _context;
  std::chrono::time_point<std::chrono::system_clock> _start, _end;
  std::ostream *_default_str;
};

void foo(/*some arguments*/) {
  auto timer = CSDSTimer("foo", "example");
  /*do some more stuff*/
}
