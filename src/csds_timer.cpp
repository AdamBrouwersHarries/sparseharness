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
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      clock_type::now() - _start);
  *_default_str << "PROFILING_DATUM(\"" << _name << "\", \"" << _context
                << "\", " << elapsed_ms.count() << ", \"C++\")" << ENDL;
#ifdef TREE_PERF
  reportEnd();
#endif
}

void CSDSTimer::reportStart() {
  *_default_str << "PFTimerStart(\"" << _name << "\", \"" << _context << "\")"
                << ENDL;
}

void CSDSTimer::reportEnd() {
  *_default_str << "PFTimerEnd(\"" << _name << "\", \"" << _context << "\")"
                << ENDL;
}