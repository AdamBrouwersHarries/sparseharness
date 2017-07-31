#pragma once

#include "csv_utils.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

class Run {
public:
  Run(std::size_t g1, std::size_t g2, std::size_t g3, std::size_t l1,
      std::size_t l2, std::size_t l3)
      : global1(g1), global2(g2), global3(g3), local1(l1), local2(l2),
        local3(l3) {}

  Run(CSV::csv_line line);

  std::size_t num_work_items();

  friend std::ostream &operator<<(std::ostream &stream, const Run &r);

  // private:
  // global ranges
  std::size_t global1 = 0;
  std::size_t global2 = 0;
  std::size_t global3 = 0;

  // local ranges
  std::size_t local1 = 0;
  std::size_t local2 = 0;
  std::size_t local3 = 0;
};