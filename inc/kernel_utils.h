#pragma once

#include <functional>
#include <map>

#include "Logger.h"
#include "arithexpr_evaluator.h"
#include "buffer_utils.h"
#include "common.h"
#include "csds_timer.h"
#include "kernel_config.h"
#include "sparse_matrix.h"
#include "vector_generator.h"
#include <cassert>

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
executorEncodeMatrix(unsigned int device_max_alloc_bytes,
                     KernelConfig<T> kernel, SparseMatrix<T> matrix, T zero,
                     // std::vector<T> xvector, std::vector<T> yvector) {
                     XVectorGenerator<T> &xgen, YVectorGenerator<T> &ygen,
                     // int v_MWidth_1, int v_MHeight_2, int v_VLength_3,
                     T alpha = static_cast<T>(1), T beta = static_cast<T>(1)) {
  start_timer(executorEncodeMatrix, kernel_utils);
  // get the configuration patterns of the kernel
  auto kprops = kernel.getProperties();

  auto cl_matrix =
      matrix.cl_encode(device_max_alloc_bytes, zero, kprops.chunkSize != 1,
                       kprops.splitSize != 1, kprops.arrayType == "ragged",
                       kprops.chunkSize, kprops.splitSize);

  auto v_MWidth_1 = kprops.arrayType == "ragged"
                        ? matrix.width()
                        : cl_matrix.cl_width / kprops.splitSize;
  // change it if we're ragged
  // auto v_MHeight_2 = (int)(cl_matrix.cl_height / kprops.chunkSize);
  auto v_MHeight_2 = cl_matrix.cl_height;
  auto v_VLength_3 = matrix.width();

  std::cerr << "Encoding matrix with sizes:"
            << "\n\tv_MWidth_1 = " << v_MWidth_1
            << "\n\tv_MHeight_2 = " << v_MHeight_2
            << "\n\tv_VLength_3 = " << v_VLength_3 << "\n";

  // generate the vector inputs
  std::cerr << "Filling with these sizes: \n\tx = " << matrix.height()
            << " \n\ty = " << cl_matrix.cl_height << ENDL;
  std::vector<T> xvector = xgen.generate(cl_matrix.cl_height, matrix, kernel);
  std::vector<T> yvector = ygen.generate(cl_matrix.cl_height, matrix, kernel);

  // ---- CREATE THE ACTUAL ARGS ----
  // Args must be in this order:
  //  1) inputs (globals + values)
  //  2) temporary/intermediate global values
  //  3) output buffer
  //  4) temporary locals
  //  5) size args

  // create an arg container!
  ArgContainer<T> arg_cnt;

  arg_cnt.m_idxs = std::move(cl_matrix.indices);
  arg_cnt.m_vals = std::move(cl_matrix.values);

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
    start_timer(outputBuffer, executorEncodeMatrix);
    {
      int memsize = Evaluator::evaluate(kernel.getOutputArg()->size, v_MWidth_1,
                                        v_MHeight_2, v_VLength_3);
      arg_cnt.output = memsize;
      LOG_DEBUG("Global output arg - arg: ", kernel.getOutputArg()->variable,
                ", address space: ", kernel.getOutputArg()->addressSpace,
                ", size:", kernel.getOutputArg()->size,
                ", realsize: ", memsize);
    }
  }
  {
    start_timer(tempGlobal, executorEncodeMatrix);
    for (auto arg : kernel.getTempGlobals()) {
      int memsize =
          Evaluator::evaluate(arg.size, v_MWidth_1, v_MHeight_2, v_VLength_3);
      arg_cnt.temp_globals.push_back(memsize);
      LOG_DEBUG("Global temp arg - arg: ", arg.variable,
                ", address space: ", arg.addressSpace, ", size:", arg.size,
                ", realsize: ", memsize);
    }
  }

  // create temporary local buffers
  {
    start_timer(tempLocal, executorEncodeMatrix);
    for (auto arg : kernel.getTempLocals()) {
      int memsize =
          Evaluator::evaluate(arg.size, v_MWidth_1, v_MHeight_2, v_VLength_3);
      arg_cnt.temp_locals.push_back(memsize);
      LOG_DEBUG("Local temp arg - arg: ", arg.variable,
                ", address space: ", arg.addressSpace, ", size:", arg.size,
                ", realsize: ", memsize);
    }
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
  {
    start_timer(sizeArgs, executorEncodeMatrix);
    for (auto sizeArg : kernel.getParamVars()) {
      int size = sizeMap[sizeArg];
      LOG_DEBUG("Size argument - name: ", sizeArg, " value: ", size);
      arg_cnt.size_args.push_back(size);
    }
  }

  // arg_cnt.size_args.push_back(v_MHeight_2);
  // arg_cnt.size_args.push_back(v_MWidth_1);
  // arg_cnt.size_args.push_back(v_VLength_3);

  return arg_cnt;
}
