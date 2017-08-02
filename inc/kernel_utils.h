#pragma once

#include <functional>

#include "arithexpr_evaluator.h"
#include "csds_timer.h"
#include "executor/Executor.h"
#include "executor/GlobalArg.h"
#include "executor/LocalArg.h"
#include "executor/ValueArg.h"
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
                     // std::vector<T> xvector, std::vector<T> yvector) {
                     XVectorGenerator<T> &xgen, YVectorGenerator<T> &ygen,
                     int v_MWidth_1, int v_MHeight_2, int v_VLength_3,
                     T alpha = static_cast<T>(1), T beta = static_cast<T>(0)) {
  start_timer(executorEncodeMatrix, kernel_utils);
  // get the configuration patterns of the kernel
  auto kprops = kernel.getProperties();
  // get the matrix as standard ELLPACK
  auto rawmat = matrix.asPaddedSOAELLPACK(zero, kprops.splitSize);

  // add on as many rows are needed
  // first check that we _need_ to
  int mem_height = matrix.height();
  if (rawmat.first.size() % kprops.chunkSize != 0) {
    // calculate the new height required to get to a multiple of the
    mem_height =
        kprops.chunkSize * ((rawmat.first.size() / kprops.chunkSize) + 1);
    // get the row length
    int row_length = rawmat.first[0].size();
    // construct a vector of "-1" values, and one of "0" values
    std::vector<int> indices(row_length, -1);
    std::vector<T> values(row_length, zero);
    // resize the raw vector with the new values
    rawmat.first.resize(mem_height, indices);
    rawmat.second.resize(mem_height, values);
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

  // generate the vector inputs
  std::cerr << "Filling with these sizes: \n\tx = " << matrix.height()
            << " \n\ty = " << mem_height << std::endl;
  std::vector<T> xvector = xgen.generate(matrix.height(), matrix, kernel);
  std::vector<T> yvector = ygen.generate(mem_height, matrix, kernel);

  // ---- CREATE THE ACTUAL ARGS ----
  // Args must be in this order:
  //  1) inputs (globals + values)
  //  2) temporary/intermediate global values
  //  3) output buffer
  //  4) temporary locals
  //  5) size args

  std::vector<KernelArg *> kernel_args;

  // create args for the matrix inputs
  kernel_args.push_back(GlobalArg::create(
      (void *)flat_indices.data(), (size_t)flat_indices.size() * sizeof(int)));
  kernel_args.push_back(GlobalArg::create(
      (void *)flat_values.data(), (size_t)flat_values.size() * sizeof(T)));

  // create args for the vector inputs
  kernel_args.push_back(GlobalArg::create((void *)xvector.data(),
                                          (size_t)xvector.size() * sizeof(T)));
  kernel_args.push_back(GlobalArg::create((void *)yvector.data(),
                                          (size_t)yvector.size() * sizeof(T)));

  // create the alpha and beta args
  kernel_args.push_back(ValueArg::create(&alpha, sizeof(T)));
  kernel_args.push_back(ValueArg::create(&beta, sizeof(T)));

  // start creating buffers with sizes that we need to evaluate
  // initialise the evaluator to figure them out
  // Evaluator::initialise_variables(v_MWidth_1, v_MHeight_2, v_VLength_3);

  // create temporary global buffers
  for (auto arg : kernel.getTempGlobals()) {
    std::cout << "Global temp arg: " << arg.variable << ", " << arg.addressSpace
              << "," << arg.size << std::endl;
    int memsize =
        Evaluator::evaluate(arg.size, v_MWidth_1, v_MHeight_2, v_VLength_3);
    std::cout << "realsize: " << memsize << std::endl;
    kernel_args.push_back(GlobalArg::create(memsize));
  }

  // create output buffer
  {
    std::cout << "expr: " << kernel.getOutputArg()->size << std::endl;
    int outputMemsize = Evaluator::evaluate(
        kernel.getOutputArg()->size, v_MWidth_1, v_MHeight_2, v_VLength_3);
    std::cout << "outputMemSize: " << outputMemsize << std::endl;
    kernel_args.push_back(GlobalArg::create(outputMemsize, true));
  }

  // create temporary local buffers

  for (auto arg : kernel.getTempLocals()) {
    std::cout << "Local temp arg: " << arg.variable << ", " << arg.addressSpace
              << "," << arg.size << std::endl;
    int memsize =
        Evaluator::evaluate(arg.size, v_MWidth_1, v_MHeight_2, v_VLength_3);
    std::cout << "realsize: " << memsize << std::endl;
    kernel_args.push_back(LocalArg::create(memsize));
  }

  // create size buffers
  kernel_args.push_back(ValueArg::create(&v_MHeight_2, sizeof(int)));
  kernel_args.push_back(ValueArg::create(&v_MWidth_1, sizeof(int)));
  kernel_args.push_back(ValueArg::create(&v_VLength_3, sizeof(int)));

  return kernel_args;
}