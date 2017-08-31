#include "csds_timer.h"

CSDSTimer::CSDSTimer(const std::string &name)
    : _name(name), _context("default"),
      _default_str(&std::cout), _start{clock_type::now()} {
#ifdef TREE_PERF
  reportStart();
#endif
}

CSDSTimer::CSDSTimer(const std::string &name, const std::string &context)
    : _name(name), _context(context),
      _default_str(&std::cout), _start{clock_type::now()} {
#ifdef TREE_PERF
  reportStart();
#endif
}

CSDSTimer::CSDSTimer(const std::string &name, const std::string &context,
                     std::ostream &stream)
    : _name(name), _context(context),
      _default_str(&stream), _start{clock_type::now()} {
#ifdef TREE_PERF
  reportStart();
#endif
}

CSDSTimer::~CSDSTimer() {
  auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      clock_type::now() - _start);
  *_default_str << "PROFILING_DATUM(\"" << _name << "\", \"" << _context
                << "\", " << ((double)elapsed_ns.count()) / 1000000.0
                << ", \"C++\")" << ENDL;
#ifdef TREE_PERF
  reportEnd();
#endif
}

void CSDSTimer::reportTiming(const std::string &name,
                             const std::string &context,
                             std::chrono::nanoseconds elapsed_ns) {
  // auto elapsed_ns =
  //     std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_ns);
  std::cout << "PROFILING_DATUM(\"" << name << "\", \"" << context << "\", "
            << ((double)elapsed_ns.count()) / 1000000.0 << ", \"C++\")" << ENDL;
}

void CSDSTimer::reportStart() {
  *_default_str << "PFTimerStart(\"" << _name << "\", \"" << _context << "\")"
                << ENDL;
}

void CSDSTimer::reportEnd() {
  *_default_str << "PFTimerEnd(\"" << _name << "\", \"" << _context << "\")"
                << ENDL;
}