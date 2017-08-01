#pragma once

#include <functional>

#include "executor/Executor.h"
#include "executor/GlobalArg.h"
#include "kernel_config.h"
#include "sparse_matrix.h"
#include "vector_generator.h"

using namespace executor;

// from:
// https://stackoverflow.com/questions/17294629/merging-sub-vectors-int-a-single-vector-c
// flatten a nested vector-of-vectors
template <template <typename...> class R = std::vector, typename Top,
          typename Sub = typename Top::value_type>
R<typename Sub::value_type> flatten(Top const &all) {
  using std::begin;
  using std::end;

  R<typename Sub::value_type> accum;

  for (auto &sub : all)
    accum.insert(end(accum), begin(sub), end(sub));

  return accum;
}

// given a loaded sparse matrix, encode it in a form that we can use in the
// executor - i.e. as a set of kernel arguments
template <typename T>
std::vector<KernelArg *>
executorEncodeMatrix(KernelConfig<T> kernel, SparseMatrix<T> matrix, T zero,
                     XVectorGenerator<T> &xgen, YVectorGenerator<T> &ygen) {
  // get the configuration patterns of the kernel
  auto kprops = kernel.getProperties();
  // get the matrix as standard ELLPACK
  auto rawmat = matrix.asPaddedSOAELLPACK(zero, kprops.splitSize);

  // add on as many rows are needed
  // first check that we _need_ to
  if (rawmat.first.size() % kprops.chunkSize != 0) {
    // calculate the new height required to get to a multiple of the
    int new_height =
        kprops.chunkSize * ((rawmat.first.size() / kprops.chunkSize) + 1);
    // get the row length
    int row_length = rawmat.first[0].size();
    // construct a vector of "-1" values, and one of "0" values
    std::vector<int> indices(row_length, -1);
    std::vector<T> values(row_length, zero);
    // resize the raw vector with the new values
    rawmat.first.resize(new_height, indices);
    rawmat.second.resize(new_height, values);
  }

  // reminder - rawmat has the following type:
  //   template <typename T>
  // using soa_ellpack_matrix =
  //     std::pair<std::vector<std::vector<int>>, std::vector<std::vector<T>>>;

  // extract some kernel arguments from it, and from the intermediate arrays in
  // the kernel
  // auto flat_indices = flatten(rawmat.first()[0]);
  auto flat_indices = flatten(rawmat.first);
  // auto flat_values = flatten(rawmat.second()[0]);
  auto flat_values = flatten(rawmat.second);

  // auto index_arg = GlobalArg::create((void *)flat_indices.data(),
  //                                    (size_t)flat_indices.size() *
  //                                    sizeof(int));

  // create args for the matrix inputs
  std::vector<KernelArg *> matrix_inputs = {
      GlobalArg::create((void *)flat_indices.data(),
                        (size_t)flat_indices.size() * sizeof(int)),
      GlobalArg::create((void *)flat_values.data(),
                        (size_t)flat_values.size() * sizeof(T))};

  // create an arg for the vector input
  std::vector<T> xvector = xgen.generate(matrix, kernel);
  std::vector<T> yvector = ygen.generate(matrix, kernel);
  std::vector<KernelArg *> vector_inputs = {
      GlobalArg::create((void *)xvector.data(),
                        (size_t)xvector.size() * sizeof(T)),
      GlobalArg::create((void *)yvector.data(),
                        (size_t)yvector.size() * sizeof(T))};

  // std::vector<KernelArg> matrix_inputs;

  return matrix_inputs;
}