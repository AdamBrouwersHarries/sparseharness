#pragma once

class Run {
  // global ranges
  std::size_t global1 = 0;
  std::size_t global2 = 0;
  std::size_t global3 = 0;

  // local ranges
  std::size_t local1 = 0;
  std::size_t local2 = 0;
  std::size_t local3 = 0;

  Run(std::size_t g1, std::size_t g2, std::size_t g3, std::size_t l1,
      std::size_t l2, std::size_t l3, )
      : global1(g1), global2(g2), global3(g3), local1(l1), local2(l2),
        local3(l3) {}
}