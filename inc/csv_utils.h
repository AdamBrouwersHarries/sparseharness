#pragma once

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
  std::stringstream lineS(line);
  std::string cell;

  while (std::getline(lineS, cell, ','))
    tokens.push_back(cell);
  return tokens;
}

inline std::vector<csv_line> load_csv(const std::string &filename) {
  csv_line line_tokens;
  std::vector<csv_line> lines;

  std::ifstream fstr(filename);

  do {
    std::string line;
    std::getline(fstr, line);
    std::cout << "Line: " << line << std::endl;
    line_tokens = tokenise_line(line);
    if (line_tokens.size() > 0)
      lines.push_back(line_tokens);
  } while (line_tokens.size() != 0);
  return lines;
}
}