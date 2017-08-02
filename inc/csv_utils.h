#pragma once

#include "csds_timer.h"
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
namespace CSV {

using csv_token = std::string;
using csv_line = std::vector<csv_token>;

// split CSV values into an array of strings
inline csv_line tokenise_line(std::string line) {
  std::vector<std::string> tokens;
  std::stringstream lineStream(line);
  std::string cell;

  while (std::getline(lineStream, cell, ','))
    tokens.push_back(cell);

  return tokens;
}

inline std::size_t read_size_t(const csv_token &str) {
  std::stringstream buffer(str);
  std::size_t value = 0;
  buffer >> value;
  return value;
}

inline std::vector<csv_line> load_csv(const std::string &filename) {
  start_timer(load_csv, csv_utils);
  csv_line line_tokens;
  std::vector<csv_line> lines;

  std::ifstream fstr(filename);

  do {
    std::string line;
    std::getline(fstr, line);
    line_tokens = tokenise_line(line);
    if (line_tokens.size() > 0)
      lines.push_back(line_tokens);
  } while (line_tokens.size() != 0);
  return lines;
}
}