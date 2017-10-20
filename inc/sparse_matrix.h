#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include "Logger.h"
#include "mmio.h"
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <tuple>
#include <vector>

#include "buffer_utils.h"
#include "common.h"
#include "csds_timer.h"

class CL_matrix {
public:
  CL_matrix(unsigned int ixs_arr_size, unsigned int vals_arr_size, int _width,
            int _height)
      : indices(ixs_arr_size), values(vals_arr_size), cl_width(_width),
        cl_height(_height) {}
  std::vector<char> indices;
  std::vector<char> values;
  int cl_width;
  int cl_height;
};

template <typename EType> class SparseMatrix {
public:
  // Constructors
  SparseMatrix(std::string filename);
  // SparseMatrix(float lo, float hi, int length, int elements);

  // readers
  template <typename T> using ellpack_row = std::vector<std::pair<int, T>>;
  template <typename T> using ellpack_matrix = std::vector<ellpack_row<T>>;

  template <typename T>
  using soa_ellpack_matrix =
      std::pair<std::vector<std::vector<int>>, std::vector<std::vector<T>>>;

  using cl_arg = std::vector<char>;

  CL_matrix cl_encode(unsigned int device_max_alloc_bytes, EType zero,
                      bool pad_height, bool pad_width, bool rsa,
                      int height_pad_modulo, int width_pad_modulo);

  SparseMatrix::ellpack_matrix<EType> &ellpack_encode(void);

  // template ellpack_matrix<float> asFloatELLPACK();
  // ellpack_matrix<double> asDoubleELLPACK();
  // ellpack_matrix<int> asIntELLPACK();

  // getters
  inline int height();
  int width();
  int nonZeros();
  void printMatrix();

private:
  // private initialisers
  void load_from_file(std::string filename);
  void calculate_ellpack();

  // tuples are: x, y, value
  // various members used to load from a file
  std::vector<std::tuple<int, int, EType>> nz_entries;
  int rows;
  int cols;
  int nonz;

  // ellpack data
  bool ellpack_calculated = false;
  std::vector<unsigned int> row_lengths;
  unsigned int max_width = 0;
  SparseMatrix::ellpack_matrix<EType> ellpackMatrix;

  // file data
  std::string filename;
};

#endif