#pragma once

#include "sparse_matrix.h"
#include "vector_generator.h"
#include "csds_timer.h"

template <typename T> class Gold {
public:
  static std::vector<T> spmv(SparseMatrix<T> A, XVectorGenerator<T> &x,
                             YVectorGenerator<T> &y, T alpha, T beta, T zero) {
    start_timer(spmv, gold);
    // get the matrix in ellpack format
    auto ellpack_a = A.ellpack_encode();
    // create a vector of the right height
    std::vector<T> result(ellpack_a.size(), 0);
    // iterate over the rows
    for (unsigned int i = 0; i < ellpack_a.size(); i++) {
      // for each row, perform the dot product.
      T acc = zero;
      for (unsigned int j = 0; j < ellpack_a[i].size(); j++) {
        auto elem = ellpack_a[i][j];
        acc += (alpha * (x.get(elem.first) * elem.second)) +
               (beta * y.get(elem.second));
      }
      result[i] = acc;
    }
    return result;
  }

private:
  Gold() {}
};
