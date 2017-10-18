#pragma once
#include "Logger.h"
#include "csds_timer.h"
#include <string>
#include <vector>

template <typename T> void printc_vec(std::vector<char> &v, int stride) {
  start_timer(printc_vec, buffer_utils);
  // get a pointer to the underlying data
  char *cptr = v.data();
  // cast to a char type
  T *tptr = reinterpret_cast<T *>(cptr);
  // get the lenght
  unsigned int tptrlen = (v.size() * sizeof(char)) / (sizeof(T));

  // traverse the pointer, printing each element
  std::cout << " Using stride: " << stride << " \n";
  std::cout << "[\n\t[";
  for (unsigned int i = 0; i < tptrlen; i++) {
    if (i % stride == 0 && i != 0) {
      std::cout << "],\n\t[";
    }
    std::cout << *(tptr + i) << ",";
  }
  std::cout << "]\n]\n";
}

// from:
// https://stackoverflow.com/questions/17294629/merging-sub-vectors-int-a-single-vector-c
// flatten a nested vector-of-vectors
template <template <typename...> class R = std::vector, typename Top,
          typename Sub = typename Top::value_type>
R<typename Sub::value_type> flatten(Top const &all) {
  start_timer(flatten, buffer_utils);
  using std::begin;
  using std::end;

  R<typename Sub::value_type> accum;

  for (auto &sub : all)
    accum.insert(end(accum), begin(sub), end(sub));

  return accum;
}

template <typename T> std::vector<char> enchar(std::vector<T> in) {
  start_timer(enchar, buffer_utils);
  // get a pointer to the underlying data
  T *uptr = in.data();
  // cast it into a char type
  char *cptr = reinterpret_cast<char *>(uptr);
  // get the length
  unsigned int cptrlen = in.size() * (sizeof(T) / sizeof(char));
  // build a vector from that
  std::vector<char> result(cptr, cptr + cptrlen);
  return result;
}
