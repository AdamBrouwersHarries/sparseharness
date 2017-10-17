#include <algorithm>
#include <atomic>
#include <bitset>
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

// [tools]
#include "common.h"
#include "csds_timer.h"
#include "csv_utils.h"
#include "options.h"

// [ocl tools]
#include "harness.h"
#include "kernel_config.h"
#include "kernel_utils.h"

// [application specific]
#include "run.h"
#include "sparse_matrix.h"
#include "vector_generator.h"

// [OpenCL]
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

int main(int argc, char **argv) {
  start_timer(main, global);
  OptParser op("Benchmark program to test the speed of the sparse matrix "
               "parser and specialisation routines without performing any "
               "actual GPU work.");

  // number of times we should try to parse the matrix file
  auto opt_trials = op.addOption<unsigned>(
      {'i', "trials", "Execute each kernel 'trials' times (default 10).", 10});
  auto opt_matrix_file =
      op.addOption<std::string>({'m', "matrix", "Input matrix"});
  auto opt_kernel_file =
      op.addOption<std::string>({'k', "kernel", "Input kernel"});

  auto opt_experiment_id = op.addOption<std::string>(
      {'e', "experiment", "An experiment ID for data reporting",
       "null_experiment"});
  op.parse(argc, argv);

  using namespace std;
  const std::string matrix_filename = opt_matrix_file->require();
  const std::string kernel_filename = opt_kernel_file->require();
  const std::string experiment = opt_experiment_id->get();
  std::cerr << "matrix_filename " << matrix_filename << ENDL;
  std::cerr << "kernel_filename " << kernel_filename << ENDL;

  for (int i = 0; i < opt_trials->require(); i++) {
    SparseMatrix<float> matrix(matrix_filename);
    KernelConfig<float> kernel(kernel_filename);

    if (matrix.height() != matrix.width()) {
      std::cout << "Matrix is not square. Failing computation." << ENDL;
      std::cerr << "Matrix is not square. Failing computation." << ENDL;
      std::exit(2);
    } else {
      std::cout << " Matrix is square - width = " << matrix.width()
                << " and height = " << matrix.height() << "\n";
    }

    // build vector generators
    ConstXVectorGenerator<float> onegen(1.0f);
    ConstYVectorGenerator<float> zerogen(0);

    // finally, build some args
    auto args =
        executorEncodeMatrix(kernel, matrix, 0.0f, onegen, zerogen, 1.0f, 0.0f);
  }
  // finish and return
  return 0;
}