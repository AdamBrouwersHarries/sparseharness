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

class HarnessSSSP : public IterativeHarness<std::vector<SqlStat>, float> {
public:
  HarnessSSSP(std::string &kernel_source, unsigned int platform,
              unsigned int device, ArgContainer<float> args,
              unsigned int trials, double timeout, double delta)
      : IterativeHarness(kernel_source, platform, device, args, trials, timeout,
                         delta) {}
  virtual std::vector<std::vector<SqlStat>> benchmark(Run run) {
    start_timer(benchmark, HarnessSSSP);
    allocateBuffers();

    // run the kernel!
    std::vector<std::vector<SqlStat>> runtimes;
    for (unsigned int t = 0; t < _trials; t++) {
      start_timer(benchmark_iteration, HarnessSSSP);
      std::vector<SqlStat> run_runtimes = executeRun(run, t);

      // sum the runtimes, and median it and report that
      // collect the median of the runtimes
      std::sort(run_runtimes.begin(), run_runtimes.end(), SqlStat::compare);
      std::chrono::nanoseconds median_time =
          run_runtimes[run_runtimes.size() / 2].getTime();

      run_runtimes.push_back(SqlStat(median_time, NOT_CHECKED, run.global1,
                                     run.local1, MEDIAN_RESULT, t));

      // collect the sum of the runtimes
      std::chrono::nanoseconds total_time = std::accumulate(
          run_runtimes.begin(), run_runtimes.end(),
          std::chrono::nanoseconds(0), // start with first element
          [](std::chrono::nanoseconds time, SqlStat stat) {
            return time + stat.getTime();
          });
      run_runtimes.push_back(SqlStat(total_time, NOT_CHECKED, run.global1,
                                     run.local1, MULTI_ITERATION_SUM));

      // add all the times to the list
      runtimes.push_back(run_runtimes);
      resetInputs();
    }
    return runtimes;
  }

private:
  std::vector<SqlStat> executeRun(Run run, unsigned int trial) {
    start_timer(executeRun, HarnessSSSP);
    std::vector<SqlStat> runtimes;

    // get pointers to the input + output mem args
    cl_mem *input_mem_ptr = &(_mem_manager._x_vect);
    cl_mem *output_mem_ptr = &(_mem_manager._output);

    // and pointers to the input + output host args
    std::vector<char> *input_host_ptr = &(_mem_manager._input_host_buffer);
    std::vector<char> *output_host_ptr = &(_mem_manager._output_host_buffer);

    bool should_terminate = false;
    int iteration = 0;
    do {
      LOG_DEBUG_INFO("Iteration: ", iteration);
      LOG_DEBUG_INFO("Host vectors before");
      printCharVector<float>("Input ", *input_host_ptr);
      printCharVector<float>("Output ", *output_host_ptr);

      // cache the output to check that it's actually changed
      std::copy(output_host_ptr->begin(), output_host_ptr->end(),
                _mem_manager._temp_out_buffer.begin());

      resetTempBuffers();
      // run the kernel
      auto time = executeKernel(run);
      runtimes.push_back(SqlStat(time, NOT_CHECKED, run.global1, run.local1,
                                 RAW_RESULT, trial, iteration));

      // copy the output back down
      readFromGlobalArg(*output_host_ptr, *output_mem_ptr);

      LOG_DEBUG_INFO("Host vectors after");
      printCharVector<float>("Input ", *input_host_ptr);
      printCharVector<float>("Output ", *output_host_ptr);

      assertBuffersNotEqual(*output_host_ptr, _mem_manager._temp_out_buffer);

      should_terminate =
          should_terminate_iteration(*input_host_ptr, *output_host_ptr);
      LOG_DEBUG_INFO("Should terminate iteration: ",
                     should_terminate ? "true" : "false");
      // swap the pointers over

      std::swap(input_mem_ptr, output_mem_ptr);
      std::swap(input_host_ptr, output_host_ptr);

      // set the kernel args
      setGlobalArg(_mem_manager._input_idx, input_mem_ptr);
      setGlobalArg(_mem_manager._output_idx, output_mem_ptr);
      // also set the y vector!
      setGlobalArg(3, input_mem_ptr);

      iteration++;
    } while (!should_terminate);
    return runtimes;
  }

  virtual bool should_terminate_iteration(std::vector<char> &input,
                                          std::vector<char> &output) {
    start_timer(should_terminate_iteration, HarnessSSSP);

    // reinterpret the args as double pointers, and get the lengths
    auto input_ptr = reinterpret_cast<float *>(input.data());
    auto output_ptr = reinterpret_cast<float *>(output.data());
    auto input_length = input.size() / sizeof(float);
    auto output_length = output.size() / sizeof(float);
    // perform a comparison across the two of them, based on pointers
    bool equal = true;
    for (unsigned int i = 0;
         equal == true && i < input_length && i < output_length; i++) {
      equal = fabs(input_ptr[i] - output_ptr[i]) < _delta;
      // std::cout << "Comparing: (" << input_ptr[i] << "," << output_ptr[i]
      //           << "), result: " << equal << "\n";
    }

    return equal;
  }
};

template <typename T>
class InitialDistancesGeneratorX : public XVectorGenerator<T> {
  T value;

public:
  InitialDistancesGeneratorX(T constv) : value(constv) {}

  virtual T generateValue(int ix, SparseMatrix<T> &sm, KernelConfig<T> &kc) {
    if (ix == 0) {
      return 0.0f;
    } else {
      return value;
    }
  }
};

template <typename T>
class InitialDistancesGeneratorY : public YVectorGenerator<T> {
  T value;

public:
  InitialDistancesGeneratorY(T constv) : value(constv) {}

  virtual T generateValue(int ix, SparseMatrix<T> &sm, KernelConfig<T> &kc) {
    if (ix == 0) {
      return 0.0f;
    } else {
      return value;
    }
  }
};

int main(int argc, char *argv[]) {
  COMMON_MAIN_PREAMBLE

  // auto zero =
  // specialise the matrix for the kernel given
  auto cl_matrix =
      kernel.specialiseMatrix(matrix, std::numeric_limits<float>::max());
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

  InitialDistancesGeneratorX<float> inital_distances_x(
      std::numeric_limits<float>::max());
  InitialDistancesGeneratorY<float> inital_distances_y(
      std::numeric_limits<float>::max());

  // get some arguments
  auto args = executorEncodeMatrix(
      kernel, matrix, std::numeric_limits<float>::max(), inital_distances_x,
      inital_distances_y, v_Width_cl, v_Height_cl, v_Length_cl, 0.0f, 0.0f);

  HarnessSSSP harness(kernel.getSource(), opt_platform->get(),
                      opt_device->get(), args, opt_trials->get(),
                      opt_timeout->get(), opt_float_delta->get());

  const std::string &kernel_name = kernel.getName();
  const std::string &host_name = hostname;
  const std::string &device_name = harness.getDeviceName();
  const std::string &experiment_id = experiment;

  for (auto run : runs) {
    start_timer(run_iteration, main);
    std::cout << "Benchmarking run: " << run << ENDL;
    std::vector<std::vector<SqlStat>> runtimes = harness.benchmark(run);
    for (auto statList : runtimes) {
      std::string command =
          SqlStat::makeSqlCommand(statList, kernel_name, host_name, device_name,
                                  matrix_name, experiment_id);
      std::cout << command << "\n";
    }
  }
}
