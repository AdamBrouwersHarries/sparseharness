#include "csds_timer.h"

CSDSTimer::CSDSTimer(std::string name)
    : _name(name), _context("default"), _default_str(&std::cout) {
  _start = std::chrono::system_clock::now();
#ifdef TREE_PERF
  reportStart();
#endif
}

CSDSTimer::CSDSTimer(std::string name, std::string context)
    : _name(name), _context(context), _default_str(&std::cout) {
  _start = std::chrono::system_clock::now();
#ifdef TREE_PERF
  reportStart();
#endif
}

CSDSTimer::CSDSTimer(std::string name, std::string context,
                     std::ostream &stream)
    : _name(name), _context(context), _default_str(&stream) {
  _start = std::chrono::system_clock::now();
#ifdef TREE_PERF
  reportStart();
#endif
}

CSDSTimer::~CSDSTimer() {
  _end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = _end - _start;
  *_default_str << "PROFILING_DATUM(\"" << _name << "\", \"" << _context
                << "\", " << elapsed_seconds.count() * 1000 << ", \"C++\")"
                << std::endl;
#ifdef TREE_PERF
  reportEnd();
#endif
}

void CSDSTimer::reportStart() {
  *_default_str << "PFTimerStart(\"" << _name << "\", \"" << _context << "\")"
                << std::endl;
}

void CSDSTimer::reportEnd() {
  *_default_str << "PFTimerEnd(\"" << _name << "\", \"" << _context << "\")"
                << std::endl;
}