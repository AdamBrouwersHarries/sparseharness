// [standard includes]
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <typeinfo>
#include <vector>

// [external includes]
// #include "OpenCL_utils.h"
// [local includes]
#include "Executor.h"
#include "common.h"
#include "csds_timer.h"
#include "csv_utils.h"
#include "kernel_config.h"
#include "kernel_utils.h"
#include "options.h"
#include "run.h"
#include "sparse_matrix.h"
#include "vector_generator.h"

#include "harness/harness_spmv.h"

int main(int argc, char *argv[]) {
  start_timer(main, global);
  OptParser op(
      "Harness for SPMV sparse matrix dense vector multiplication benchmarks");

  auto opt_platform = op.addOption<unsigned>(
      {'p', "platform", "OpenCL platform index (default 0).", 0});
  auto opt_device = op.addOption<unsigned>(
      {'d', "device", "OpenCL device index (default 0).", 0});
  auto opt_iterations = op.addOption<unsigned>(
      {'i', "iterations",
       "Execute each kernel 'iterations' times (default 10).", 10});

  //   auto opt_input_file = op.addOption<std::string>({'f', "file", "Input
  //   file"});

  auto opt_matrix_file =
      op.addOption<std::string>({'m', "matrix", "Input matrix"});
  auto opt_kernel_file =
      op.addOption<std::string>({'k', "kernel", "Input kernel"});
  auto opt_run_file =
      op.addOption<std::string>({'r', "runfile", "Run configuration file"});
  auto opt_host_name = op.addOption<std::string>(
      {'h', "hostname", "Host the harness is running on"});

  auto opt_timeout = op.addOption<float>(
      {'t', "timeout", "Timeout to avoid multiple executions (default 100ms).",
       100.0f});

  op.parse(argc, argv);

  using namespace std;

  const std::string matrix_filename = opt_matrix_file->require();
  const std::string kernel_filename = opt_kernel_file->require();
  const std::string runs_filename = opt_run_file->require();
  const std::string hostname = opt_host_name->require();

  std::cerr << "matrix_filename " << matrix_filename << ENDL;
  std::cerr << "kernel_filename " << kernel_filename << ENDL;

  // initialise a matrix, kernel, and set of run parameters from files
  SparseMatrix<float> matrix(matrix_filename);
  KernelConfig<float> kernel(kernel_filename);
  auto csvlines = CSV::load_csv(runs_filename);
  std::vector<Run> runs;
  std::transform(csvlines.begin(), csvlines.end(), std::back_inserter(runs),
                 [](CSV::csv_line line) -> Run { return Run(line); });

  for (auto run : runs)
    std::cerr << run << ENDL;

  // check the matrix
  if (matrix.height() != matrix.width()) {
    std::cout << "Matrix is not square. Failing computation." << ENDL;
    std::cerr << "Matrix is not square. Failing computation." << ENDL;
    std::exit(2);
  }

  // specialise the matrix for the kernel given
  auto cl_matrix = kernel.specialiseMatrix(matrix, 0.0f);
  // extract size variables from it
  int v_Width_cl = cl_matrix.getCLVWidth();
  int v_Height_cl = cl_matrix.getCLVHeight();
  int v_Length_cl = cl_matrix.rows;

  // size args of name/order:
  // v_MWidthC_1, v_MHeight_2, v_VLength_3
  std::vector<int> size_args{v_Width_cl, v_Height_cl, v_Length_cl};

  // initialise the executor
  initExecutor(opt_platform->get(), opt_device->get());

  // generate a vector

  ConstXVectorGenerator<float> tengen(10);
  ConstYVectorGenerator<float> onegen(1);

  auto clkernel = executor::Kernel(kernel.getSource(), "KERNEL", "").build();
  // get some arguments
  auto args = executorEncodeMatrix(kernel, matrix, 0.0f, tengen, onegen,
                                   v_Width_cl, v_Height_cl, v_Length_cl);
  HarnessSpmv harness(clkernel, args);
  for (auto run : runs) {
    start_timer(run_iteration, main);
    std::cout << "Benchmarking run: " << run << ENDL;
    std::vector<double> runtimes =
        harness.benchmark(run, opt_iterations->get(), opt_timeout->get());
    std::cout << "runtimes: [";
    for (auto time : runtimes) {
      std::cout << "," << time;
    }
    std::cout << "]" << ENDL;
  }

  shutdownExecutor();
}
