#pragma once
#include "executor/Executor.h"
#include "executor/GlobalArg.h"
#include "executor/LocalArg.h"
#include "executor/ValueArg.h"

#include "run.h"

class Harness {
public:
  Harness(cl::Kernel kernel, std::vector<KernelArg *> args)
      : _kernel(kernel), _args(args) {}

protected:
  cl::Kernel _kernel;
  std::vector<KernelArg *> _args;
};

class HarnessSpmv : Harness {
public:
  HarnessSpmv(cl::Kernel kernel, std::vector<KernelArg *> args)
      : Harness(kernel, args) {}
  std::vector<double> benchmark(Run run, int iterations, double timeout) {
    start_timer(benchmark, HarnessSpmv);

    // kernel setup
    int i = 0;
    for (auto &arg : _args) {
      arg->upload();
      arg->setAsKernelArg(_kernel, i);
      ++i;
    }

    // iterations
    std::vector<double> runtimes(iterations);

    for (int i = 0; i < iterations; i++) {
      start_timer(benchmark_iteration, HarnessSpmv);
      // std::cout << "Iteration: " << i << '\n';

      for (auto &arg : _args) {
        arg->clear();
      }

      double runtime = executeKernel(run);

      runtimes[i] = runtime;

      if (timeout != 0.0 && runtime >= timeout) {
        runtimes.resize(i + 1);
        return runtimes;
      }
    }
    return runtimes;
  }

private:
  double executeKernel(Run run) {
    start_timer(executeKernel, HarnessSpmv);
    auto &devPtr = executor::globalDeviceList.front();
    // get our local and global sizes
    cl_uint localSize1 = run.local1;
    cl_uint localSize2 = run.local2;
    cl_uint localSize3 = run.local3;
    cl_uint globalSize1 = run.global1;
    cl_uint globalSize2 = run.global2;
    cl_uint globalSize3 = run.global3;

    auto event = devPtr->enqueue(
        _kernel, cl::NDRange(globalSize1, globalSize2, globalSize3),
        cl::NDRange(localSize1, localSize2, localSize3));
    // {
    //   start_timer(arg_download, executeKernel);
    //   for (auto &arg : _args) {
    //     start_timer(download_indivdual_arg, arg_download);
    //     arg->download();
    //   }
    // }

    return getRuntimeInMilliseconds(event);
  }
};