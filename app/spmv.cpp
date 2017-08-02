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
#include "csds_timer.h"
#include "csv_utils.h"
#include "kernel_config.h"
#include "kernel_utils.h"
#include "options.h"
#include "run.h"
#include "sparse_matrix.h"
#include "vector_generator.h"
// #include "spmv_harness.h"
// #include "spmvrun.h"

#include "Executor.h"

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

  auto opt_timeout = op.addOption<float>(
      {'t', "timeout", "Timeout to avoid multiple executions (default 100ms).",
       100.0f});

  auto opt_double =
      op.addOption<bool>({0, "double", "Use double precision.", false});
  auto opt_threaded = op.addOption<bool>(
      {'t', "threaded",
       "Use a separate thread for compilation and execution (default true).",
       true});

  auto opt_force = op.addOption<bool>(
      {'f', "force", "Override cached cross validation files.", false});
  auto opt_clean = op.addOption<bool>(
      {'c', "clean", "Clean temporary files and exit.", false});
  op.parse(argc, argv);

  using namespace std;

  const std::string matrix_filename = opt_matrix_file->require();
  const std::string kernel_filename = opt_kernel_file->require();
  const std::string runs_filename = opt_run_file->require();

  std::cerr << "matrix_filename " << matrix_filename << std::endl;
  std::cerr << "kernel_filename " << kernel_filename << std::endl;

  // initialise a matrix, kernel, and set of run parameters from files
  SparseMatrix<float> matrix(matrix_filename);
  KernelConfig<float> kernel(kernel_filename);
  auto csvlines = CSV::load_csv(runs_filename);
  std::vector<Run> runs;
  std::transform(csvlines.begin(), csvlines.end(), std::back_inserter(runs),
                 [](CSV::csv_line line) -> Run { return Run(line); });

  for (auto run : runs)
    std::cerr << run << std::endl;

  // check the matrix
  if (matrix.height() != matrix.width()) {
    std::cout << "Matrix is not square. Failing computation." << std::endl;
    std::cerr << "Matrix is not square. Failing computation." << std::endl;
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

  auto clkernel = executor::Kernel(kernel.getSource(), "KERNEL", "");
  // get some arguments
  auto args = executorEncodeMatrix(kernel, matrix, 0.0f, tengen, onegen,
                                   v_Width_cl, v_Height_cl, v_Length_cl);
  for (auto run : runs) {
    start_timer(run_iteration, main);
    std::cout << "Benchmarking run: " << run << std::endl;
    std::vector<double> runtimes;
    benchmark(clkernel, run.local1, run.local2, run.local3, run.global1,
              run.global2, run.global3, args, opt_iterations->get(),
              opt_timeout->get(), runtimes);
    std::cout << "runtimes: [";
    for (auto time : runtimes) {
      std::cout << "," << time;
    }
    std::cout << "]" << std::endl;
  }

  shutdownExecutor();
}
