#include "run.h"
#include <assert.h>

Run::Run(CSV::csv_line line) {
  assert(line.size() == 6 && "Bad CSV format");

  // read the global ranges
  global1 = CSV::read_size_t(line[0]);
  global2 = CSV::read_size_t(line[1]);
  global3 = CSV::read_size_t(line[2]);

  // read the local ranges
  local1 = CSV::read_size_t(line[3]);
  local2 = CSV::read_size_t(line[4]);
  local3 = CSV::read_size_t(line[5]);
}

std::size_t Run::num_work_items() { return local1 * local2 * local3; }

std::ostream &operator<<(std::ostream &stream, const Run &r) {
  stream << "{" << r.global1 << " " << r.global2 << " " << r.global3 << " / "
         << r.local1 << " " << r.local2 << " " << r.local3 << "}";
  return stream;
};