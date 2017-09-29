#pragma once

#include <functional>
#include <map>

#include "Logger.h"
#include "arithexpr_evaluator.h"
#include "common.h"
#include "csds_timer.h"
#include "kernel_config.h"
#include "sparse_matrix.h"
#include "vector_generator.h"
#include <cassert>

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

template <typename T> std::vector<char> enchar(std::vector<T> in) {
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

typedef std::vector<char> raw_arg;

template <typename T> class ArgContainer {
public:
  raw_arg m_idxs;
  raw_arg m_vals;
  raw_arg x_vect;
  raw_arg y_vect;
  T alpha;
  T beta;
  // the rest are just sizes ready for allocation!
  std::vector<unsigned int> temp_globals;
  unsigned int output;
  std::vector<unsigned int> temp_locals;
  std::vector<unsigned int> size_args;
};

// given a loaded sparse matrix, encode it in a form that we can use in the
// executor - i.e. as a set of kernel arguments
template <typename T>
ArgContainer<T>
executorEncodeMatrix(KernelConfig<T> kernel, SparseMatrix<T> matrix, T zero,
                     // std::vector<T> xvector, std::vector<T> yvector) {
                     XVectorGenerator<T> &xgen, YVectorGenerator<T> &ygen,
                     // int v_MWidth_1, int v_MHeight_2, int v_VLength_3,
                     T alpha = static_cast<T>(1), T beta = static_cast<T>(1)) {
  start_timer(executorEncodeMatrix, kernel_utils);
  // get the configuration patterns of the kernel
  auto kprops = kernel.getProperties();

  // get the matrix as standard ELLPACK
  auto rawmat = kprops.arrayType == "ragged"
                    ? matrix.asSOAELLPACK()
                    : matrix.asPaddedSOAELLPACK(zero, kprops.splitSize);

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

  auto v_MWidth_1 = kprops.arrayType == "ragged"
                        ? matrix.width()
                        : (int)rawmat.first[0].size() / kprops.splitSize;
  // change it if we're ragged
  auto v_MHeight_2 = (int)(rawmat.first.size()) / kprops.chunkSize;
  auto v_VLength_3 = matrix.width();

  std::cerr << "Encoding matrix with sizes:"
            << "\n\tv_MWidth_1 = " << v_MWidth_1
            << "\n\tv_MHeight_2 = " << v_MHeight_2
            << "\n\tv_VLength_3 = " << v_VLength_3 << "\n";

  if (kprops.arrayType == "ragged") {
    // if we're ragged, we need to prepend row sizes + capacities to the arrays:
    for (auto &arr : rawmat.first) {
      // get the length
      int len = arr.size();
      arr.insert(arr.begin(), len);
      arr.insert(arr.begin(), len);
    }
    for (auto &arr : rawmat.second) {
      // get the length
      int len = arr.size();
      arr.insert(arr.begin(), len);
      arr.insert(arr.begin(), len);
    }

    // we now need to build "offset" data, and prepend it
    // as we do some casts here, we need to make sure that int and T have the
    // same size
    assert(sizeof(T) == sizeof(int));
    std::vector<int> offsets;
    std::vector<T> offsets_T;
    int sum = sizeof(T) * rawmat.first.size();
    T *tsptr = reinterpret_cast<T *>(&sum);
    // manually do a prefix sum of the sizes
    for (unsigned int i = 0; i < rawmat.first.size(); i++) {
      // assign the current sum to offsets[i] (by appending)
      offsets.push_back(sum);
      offsets_T.push_back(*tsptr);
      sum += rawmat.first[i].size() * sizeof(T);
    }

    // append the offsets array to the indices and values arrays
    rawmat.first.insert(rawmat.first.begin(), offsets);
    rawmat.second.insert(rawmat.second.begin(), offsets_T);
  }

  // reminder - rawmat has the following type:
  //   template <typename T>
  // using soa_ellpack_matrix =
  //     std::pair<std::vector<std::vector<int>>, std::vector<std::vector<T>>>;

  // extract some kernel arguments from it, and from the intermediate arrays in
  // the kernel
  // auto flat_indices = flatten(rawmat.first()[0]);
  // std::cout << "Raw matrix indices: "
  //           << "\n";
  // for (auto row : rawmat.first) {
  //   std::ostringstream ss;
  //   ss << "[";
  //   std::copy(row.begin(), row.end() - 1, std::ostream_iterator<int>(ss, ",
  //   ")); ss << row.back(); ss << "]";

  //   std::cout << ss.str() << "\n";
  // }

  // std::cout << "Raw matrix values: "
  //           << "\n";
  // for (auto row : rawmat.second) {
  //   std::ostringstream ss;
  //   ss << "[";
  //   std::copy(row.begin(), row.end() - 1, std::ostream_iterator<T>(ss, ",
  //   ")); ss << row.back(); ss << "]";

  //   std::cout << ss.str() << "\n";
  // }

  auto flat_indices = flatten(rawmat.first);
  // auto flat_values = flatten(rawmat.second()[0]);
  auto flat_values = flatten(rawmat.second);

  // generate the vector inputs
  std::cerr << "Filling with these sizes: \n\tx = " << matrix.height()
            << " \n\ty = " << mem_height << ENDL;
  std::vector<T> xvector = xgen.generate(mem_height, matrix, kernel);
  std::vector<T> yvector = ygen.generate(mem_height, matrix, kernel);

  // ---- CREATE THE ACTUAL ARGS ----
  // Args must be in this order:
  //  1) inputs (globals + values)
  //  2) temporary/intermediate global values
  //  3) output buffer
  //  4) temporary locals
  //  5) size args

  // create an arg container!
  ArgContainer<T> arg_cnt;

  // create args for the matrix inputs
  LOG_DEBUG("Input matrix mem arg: ",
            (size_t)flat_indices.size() * sizeof(int));
  LOG_DEBUG("Input matrix mem arg: ", (size_t)flat_indices.size() * sizeof(T));

  // std::cout << "First 10 elements of matrix: [[";
  // for (int i = 0; i < 10; i++) {
  //   std::cout << "(" << flat_indices[i] << "," << flat_values[i] << "),";
  // }
  // std::cout << "...\n";

  arg_cnt.m_idxs = enchar<int>(flat_indices);
  arg_cnt.m_vals = enchar<T>(flat_values);

  // create args for the vector inputs
  // TODO: do we actually need to make the x vector bigger when we pad
  // vertically?
  LOG_DEBUG("Input vector arg: ", (size_t)xvector.size() * sizeof(T));
  LOG_DEBUG("Input vector arg: ", (size_t)yvector.size() * sizeof(T));

  arg_cnt.x_vect = enchar<T>(xvector);
  arg_cnt.y_vect = enchar<T>(yvector);

  // create the alpha and beta args
  arg_cnt.alpha = alpha;
  arg_cnt.beta = beta;

  // create output buffer
  {
    int memsize = Evaluator::evaluate(kernel.getOutputArg()->size, v_MWidth_1,
                                      v_MHeight_2, v_VLength_3);
    arg_cnt.output = memsize;
    LOG_DEBUG("Global output arg - arg: ", kernel.getOutputArg()->variable,
              ", address space: ", kernel.getOutputArg()->addressSpace,
              ", size:", kernel.getOutputArg()->size, ", realsize: ", memsize);
  }

  for (auto arg : kernel.getTempGlobals()) {
    int memsize =
        Evaluator::evaluate(arg.size, v_MWidth_1, v_MHeight_2, v_VLength_3);
    arg_cnt.temp_globals.push_back(memsize);
    LOG_DEBUG("Global temp arg - arg: ", arg.variable,
              ", address space: ", arg.addressSpace, ", size:", arg.size,
              ", realsize: ", memsize);
  }

  // create temporary local buffers
  for (auto arg : kernel.getTempLocals()) {
    int memsize =
        Evaluator::evaluate(arg.size, v_MWidth_1, v_MHeight_2, v_VLength_3);
    arg_cnt.temp_locals.push_back(memsize);
    LOG_DEBUG("Local temp arg - arg: ", arg.variable,
              ", address space: ", arg.addressSpace, ", size:", arg.size,
              ", realsize: ", memsize);
  }

  // create size buffers
  // match the paramvars to the buffers
  auto sizeMap = std::map<std::string, int>{
      {"MWidthC", v_MWidth_1},
      {"MHeight", v_MHeight_2},
      {"VLength", v_VLength_3},
  };

  // iterate over the size args, and do a lookup for each of them.
  // this should keep the order correct, and also correctly provide the total
  // amount that we need, rather than overspecifying when we don't need some
  for (auto sizeArg : kernel.getParamVars()) {
    int size = sizeMap[sizeArg];
    LOG_DEBUG("Size argument - name: ", sizeArg, " value: ", size);
    arg_cnt.size_args.push_back(size);
  }

  // arg_cnt.size_args.push_back(v_MHeight_2);
  // arg_cnt.size_args.push_back(v_MWidth_1);
  // arg_cnt.size_args.push_back(v_VLength_3);

  return arg_cnt;
}
