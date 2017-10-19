#pragma once
#include "Logger.h"
#include "csds_timer.h"
#include <string>
#include <vector>

template <typename T> void printc_vec(std::vector<char> &v, int stride) {
  start_timer(printc_vec, buffer_utils);
  // get a pointer to the underlying data
  char *cptr = v.data();
  // cast to a T type
  T *tptr = reinterpret_cast<T *>(cptr);
  // get the length
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

template <typename T>
void print_rsa_matrix(std::vector<char> &v,
                      std::vector<unsigned long> offsets) {
  start_timer(print_rsa_matrix, buffer_utils);
  // get a pointer to the underlying data
  char *cptr = v.data();

  // initialise the begin/end ptrs

  std::cout << "[\n";
  for (unsigned int i = 0; i < offsets.size() - 1; i++) {
    char *begin = cptr + offsets[i];
    char *end = cptr + offsets[i + 1];

    // print between begin and end
    std::cout << "\t[";
    // get begin as an int ptr
    int *begin_i = reinterpret_cast<int *>(begin);
    std::cout << begin_i[0] << "," << begin_i[1];

    // now get as a T ptr
    T *t_ptr = reinterpret_cast<T *>(begin + (sizeof(int) * 2));
    unsigned int length =
        ((offsets[i + 1] - offsets[i]) - (sizeof(int) * 2)) / sizeof(T);
    for (unsigned int j = 0; j < length; j++) {
      std::cout << "," << t_ptr[j];
    }
    std::cout << "]\n";
  }
  std::cout << "]\n";
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
