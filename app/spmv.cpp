// [standard includes]
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

class HarnessSPMV : public Harness<SqlStat, float> {
public:
  HarnessSPMV(std::string &kernel_source, unsigned int platform,
              unsigned int device, ArgContainer<float> args)
      : Harness(kernel_source, platform, device, args) {}
  std::vector<SqlStat> benchmark(Run run, int iterations, double timeout,
                                 double delta) {

    start_timer(benchmark, HarnessSPMV);
    allocateBuffers();

    // run the kernel!
    std::vector<SqlStat> runtimes;
    for (int i = 0; i < iterations; i++) {
      start_timer(benchmark_iteration, HarnessSPMV);

      // get pointers to the input + output mem args
      // cl_mem *input_mem_ptr = &(_mem_manager._x_vect);
      // cl_mem *output_mem_ptr = &(_mem_manager._output);
      LOG_DEBUG_INFO("Host vectors before");
      printCharVector<float>("Input ", _mem_manager._input_host_buffer);
      printCharVector<float>("Output ", _mem_manager._output_host_buffer);

      resetTempBuffers();
      // run the kernel
      runtimes.push_back(executeRun(run));

      // copy the output back down
      readFromGlobalArg(_mem_manager._output_host_buffer, _mem_manager._output);

      LOG_DEBUG_INFO("Host vectors after");
      printCharVector<float>("Input ", _mem_manager._input_host_buffer);
      printCharVector<float>("Output ", _mem_manager._output_host_buffer);

      // check to see that we've actually done something with our SPMV!
      assertBuffersNotEqual(_mem_manager._output_host_buffer,
                            _mem_manager._temp_out_buffer);
    }
    // sum the runtimes, and median it and report that
    std::sort(runtimes.begin(), runtimes.end(), SqlStat::compare);
    std::chrono::nanoseconds median_time =
        runtimes[runtimes.size() / 2].getTime();

    runtimes.push_back(SqlStat(median_time, NOT_CHECKED, run.global1,
                               run.local1, MEDIAN_RESULT));

    return runtimes;
  }

  SqlStat executeRun(Run run) {
    // get the runtime from a single kernel run
    std::chrono::nanoseconds time = executeKernel(run);
    return SqlStat(time, NOT_CHECKED, run.global1, run.local1, RAW_RESULT);
  }
};

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
  auto opt_matrix_name =
      op.addOption<std::string>({'f', "matrix_name", "Input matrix name"});
  auto opt_kernel_file =
      op.addOption<std::string>({'k', "kernel", "Input kernel"});
  auto opt_run_file =
      op.addOption<std::string>({'r', "runfile", "Run configuration file"});

  auto opt_host_name = op.addOption<std::string>(
      {'n', "hostname", "Host the harness is running on"});
  auto opt_experiment_id = op.addOption<std::string>(
      {'e', "experiment", "An experiment ID for data reporting"});
  auto opt_float_delta = op.addOption<double>(
      {'t', "delta", "Delta for floating point comparisons", 0.0001});

  auto opt_timeout = op.addOption<float>(
      {'t', "timeout", "Timeout to avoid multiple executions (default 100ms).",
       100.0f});

  op.parse(argc, argv);

  using namespace std;

  const std::string matrix_filename = opt_matrix_file->require();
  const std::string matrix_name = opt_matrix_name->require();
  const std::string kernel_filename = opt_kernel_file->require();
  const std::string runs_filename = opt_run_file->require();
  const std::string hostname = opt_host_name->require();
  const std::string experiment = opt_experiment_id->require();

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
  } else {
    std::cout << " Matrix is square - width = " << matrix.width()
              << " and height = " << matrix.height() << "\n";
  }

  // specialise the matrix for the kernel given
  auto cl_matrix = kernel.specialiseMatrix(matrix, 0.0f);
  // extract size variables from it
  int v_Width_cl = cl_matrix.getCLVWidth();
  int v_Height_cl = cl_matrix.getCLVHeight();
  int v_Length_cl = matrix.width();

  std::cout << "v_Width_cl = " << v_Width_cl << "\n";
  std::cout << "v_Height_cl = " << v_Height_cl << "\n";
  std::cout << "v_Length_cl = " << v_Length_cl << "\n";

  // size args of name/order:
  // v_MWidthC_1, v_MHeight_2, v_VLength_3
  // std::vector<int> size_args{v_Width_cl, v_Width_cl, v_Length_cl};

  // generate a vector

  ConstXVectorGenerator<float> onegen(1.0f);
  ConstYVectorGenerator<float> zerogen(0);

  // auto clkernel = executor::Kernel(kernel.getSource(), "KERNEL", "").build();
  // get some arguments
  auto args =
      executorEncodeMatrix(kernel, matrix, 0.0f, onegen, zerogen, v_Width_cl,
                           v_Height_cl, v_Length_cl, 1.0f, 0.0f);
  HarnessSPMV harness(kernel.getSource(), opt_platform->get(),
                      opt_device->get(), args);
  const std::string &kernel_name = kernel.getName();
  const std::string &host_name = hostname;
  const std::string &device_name = harness.getDeviceName();
  const std::string &experiment_id = experiment;
  for (auto run : runs) {
    start_timer(run_iteration, main);
    std::cout << "Benchmarking run: " << run << ENDL;
    std::vector<SqlStat> runtimes = harness.benchmark(
        run, opt_iterations->get(), opt_timeout->get(), opt_float_delta->get());
    std::cout << "runtimes: [";
    for (auto time : runtimes) {
      std::cout << "\n\t"
                << time.printStat(kernel_name, host_name, device_name,
                                  matrix_name, experiment_id);
    }
    std::cout << "\n]" << ENDL;
    std::string command =
        SqlStat::makeSqlCommand(runtimes, kernel_name, host_name, device_name,
                                matrix_name, experiment_id);
    std::cout << command << "\n";
  }
}
