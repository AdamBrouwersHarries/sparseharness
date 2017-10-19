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
#include "spmv_gold.h"

// [OpenCL]
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

class HarnessSPMV : public Harness<SqlStat, float> {
public:
  HarnessSPMV(std::string &kernel_source, unsigned int platform,
              unsigned int device, ArgContainer<float> args,
              unsigned int trials, std::chrono::milliseconds timeout,
              double delta)
      : Harness(kernel_source, platform, device, args, trials, timeout, delta) {
    allocateBuffers();
  }
  std::vector<SqlStat> benchmark(Run run, std::vector<float> &gold) {

    start_timer(benchmark, HarnessSPMV);

    // run the kernel!
    std::vector<SqlStat> runtimes;
    for (unsigned int t = 0; t < _trials; t++) {
      start_timer(benchmark_iteration, HarnessSPMV);

      // get pointers to the input + output mem args
      // cl_mem *input_mem_ptr = &(_mem_manager._x_vect);
      // cl_mem *output_mem_ptr = &(_mem_manager._output);
      // LOG_DEBUG_INFO("Host vectors before");
      // printCharVector<float>("Input ", _mem_manager._input_host_buffer);
      // printCharVector<float>("Output ", _mem_manager._output_host_buffer);

      resetTempBuffers();
      // run the kernel
      // get the runtime and add it to the list of times
      auto stat = executeRun(run, t, gold);
      runtimes.push_back(stat);
      // check to see if we've breached the timeout
      if (stat.getTime() > _timeout) {
        break;
      }
      // if not, see if we can reduce the timeout (if we have a
      // particularly good parameter set, for example)
      lowerTimeout(stat.getTime());

      // LOG_DEBUG_INFO("Host vectors after");
      // printCharVector<float>("Input ", _mem_manager._input_host_buffer);
      // printCharVector<float>("Output ", _mem_manager._output_host_buffer);

      // check to see that we've actually done something with our SPMV!
      assertBuffersNotEqual(_mem_manager._output_host_buffer,
                            _mem_manager._temp_out_buffer);
    }
    // sum the runtimes, and median it and report that
    std::sort(runtimes.begin(), runtimes.end(), SqlStat::compare);
    std::chrono::nanoseconds median_time =
        runtimes[runtimes.size() / 2].getTime();

    runtimes.push_back(SqlStat(median_time, STATISTIC_VALUE, run.global1,
                               run.local1, MEDIAN_RESULT));

    return runtimes;
  }

private:
  virtual SqlStat executeRun(Run run, unsigned int trial,
                             std::vector<float> &gold) {
    // get the runtime from a single kernel run
    std::chrono::nanoseconds time = executeKernel(run);

    // copy the output back down
    readFromGlobalArg(_mem_manager._output_host_buffer, _mem_manager._output);
    auto correctness = check_result(gold);
    return SqlStat(time, correctness, run.global1, run.local1, RAW_RESULT);
  }
};

int main(int argc, char *argv[]) {
  COMMON_MAIN_PREAMBLE(float)

  // build non-matrix args
  ConstXVectorGenerator<float> x(1.0f);
  ConstYVectorGenerator<float> y(0);
  float alpha = 1.0f;
  float beta = 0.0f;

  // get some arguments
  unsigned long max_alloc =
      512 * 1024 * 1024; // 0.5 GB - a conservative estimate
  max_alloc = deviceGetMaxAllocSize(opt_platform->get(), opt_device->get());
  std::cout << "Got max alloc: " << max_alloc << "\n";

  auto args =
      executorEncodeMatrix(max_alloc, kernel, matrix, 0.0f, x, y, alpha, beta);

  HarnessSPMV harness(kernel.getSource(), opt_platform->get(),
                      opt_device->get(), args, opt_trials->get(),
                      std::chrono::milliseconds(opt_timeout->get()),
                      opt_float_delta->get());

  // calculate the gold value (it's expensive, so do it after
  // the things that might fail)
  auto gold = Gold<float>::spmv(matrix, x, y, alpha, beta, 0.0f);

  const std::string &kernel_name = kernel.getName();
  const std::string &host_name = hostname;
  const std::string &device_name = harness.getDeviceName();
  const std::string &experiment_id = experiment;

  for (auto run : runs) {
    start_timer(run_iteration, main);
    std::cout << "Benchmarking run: " << run << ENDL;
    std::vector<SqlStat> runtimes = harness.benchmark(run, gold);
    std::cout << "runtimes: [";

    // todo: Get the best runtime, and use that to update the "timeout" value
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
