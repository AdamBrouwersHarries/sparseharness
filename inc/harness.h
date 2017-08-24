#pragma once
#include "executor/Executor.h"
#include "executor/GlobalArg.h"
#include "executor/KernelArg.h"
#include "executor/LocalArg.h"
#include "executor/ValueArg.h"
#include "kernel_utils.h"
#include "opencl_utils.h"

#include "run.h"

template <typename TimingType, typename SemiRingType> class Harness {
public:
  Harness(std::string &kernel_source, unsigned int platform,
          unsigned int device, ArgContainer<SemiRingType> args)
      : _kernel_source(kernel_source), _args(args) {

    // initialise OpenCL:
    // get the number of platforms
    _platformIdCount = 0;
    clGetPlatformIDs(0, nullptr, &_platformIdCount);

    if (_platformIdCount == 0) {
      std::cerr << "No OpenCL devices found! \n";
    }

    // make a vector of platform ids
    std::vector<cl_platform_id> platformIds(_platformIdCount);
    clGetPlatformIDs(_platformIdCount, platformIds.data(), nullptr);

    // get the number of devices from the platform
    _deviceIdCount = 0;
    clGetDeviceIDs(platformIds[platform], CL_DEVICE_TYPE_ALL, 0, nullptr,
                   &_deviceIdCount);

    // get a list of devices from the platform
    _deviceIds.resize(_deviceIdCount);
    clGetDeviceIDs(platformIds[platform], CL_DEVICE_TYPE_ALL, _deviceIdCount,
                   _deviceIds.data(), nullptr);

    // create a context on that device (with some properties)
    const cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platformIds[platform]), 0, 0};

    _context = clCreateContext(contextProperties, _deviceIdCount,
                               _deviceIds.data(), nullptr, nullptr, &_error);
    checkCLError(_error);

    // create a kernel from the source

    // recast the std::string as a size_t array and char array
    size_t lengths[1] = {_kernel_source.size()};
    const char *sources[1] = {_kernel_source.data()};
    // create the program
    cl_program program =
        clCreateProgramWithSource(_context, 1, sources, lengths, &_error);
    checkCLError(_error);

    // build the program
    _error = clBuildProgram(program, _deviceIdCount, _deviceIds.data(), "",
                            nullptr, nullptr);
    checkCLError(_error);

    // create a kernel from the program
    cl_kernel kernel = clCreateKernel(program, "KERNEL", &_error);
    checkCLError(_error);
    _kernel = kernel;
  }

  virtual std::vector<TimingType> benchmark(Run run, int iterations,
                                            double timeout, double delta) = 0;

  virtual void print_sql_stats(const Run &run, const std::string &kname,
                               const std::string &mname,
                               const std::string &hname,
                               const std::string &experiment_id,
                               std::vector<TimingType> &times)

  {
    auto &devPtr = executor::globalDeviceList.front();
    std::cout << "INSERT INTO table_name (time, correctness, kernel, "
              << "global, local, host, device, matrix, iteration, trial,"
              << "statistic, experiment_id) VALUES ";
    int trial = 0;
    for (auto t : times) {
      if (trial != 0) {
        std::cout << ",";
      }
      std::cout << "(" << t << ",\"notchecked\", \"" << kname << "\", "
                << run.global1 << ", " << run.local1 << ", \"" << hname
                << "\", \"" << devPtr->name() << "\", \"" << mname << "\", 0,"

                << trial << ", \"RAW_RESULT\", \"" << experiment_id << "\")";
      trial++;
    }
    std::cout << ";\n";
  }

protected:
  virtual double executeKernel(Run run) = 0;
  // stateful error code :(
  cl_int _error;
  std::string _kernel_source;
  cl_kernel _kernel;
  cl_uint _platformIdCount;
  cl_uint _deviceIdCount;

  std::vector<cl_device_id> _deviceIds;

  cl_context _context;
  ArgContainer<SemiRingType> _args;
};

// template <typename T> class IterativeHarness : public Harness<std::vector<T>>
// {
template <typename TimingType, typename SemiRingType>
class IterativeHarness : public Harness<TimingType, SemiRingType> {
public:
  IterativeHarness(std::string &kernel_source, unsigned int platform,
                   unsigned int device, ArgContainer<SemiRingType> args)
      : Harness<TimingType, SemiRingType>(kernel_source, platform, device,
                                          args) {}

protected:
  virtual bool should_terminate_iteration(executor::KernelArg *input,
                                          executor::KernelArg *output,
                                          double delta) = 0;
};