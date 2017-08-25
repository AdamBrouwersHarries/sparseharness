#pragma once
#include "executor/Executor.h"
#include "executor/GlobalArg.h"
#include "executor/KernelArg.h"
#include "executor/LocalArg.h"
#include "executor/ValueArg.h"
#include "kernel_utils.h"
#include "opencl_utils.h"

#include "run.h"
#include <chrono>

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

    // finally, create a command queue from the device and context);
    _queue = clCreateCommandQueue(_context, _deviceIds[device],
                                  CL_QUEUE_PROFILING_ENABLE, &_error);
    checkCLError(_error);
  }

  virtual std::vector<TimingType> benchmark(Run run, int iterations,
                                            double timeout, double delta) = 0;

  virtual void
  print_sql_stats(const Run &run, const std::string &kname,
                  const std::string &mname, const std::string &hname,
                  const std::string &experiment_id, std::vector<double> &times)

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
  virtual TimingType executeRun(Run run) = 0;

  std::chrono::milliseconds executeKernel(Run run) {
    cl_event ev;
    const size_t global_range[3] = {run.global1, run.global2, run.global3};
    const size_t local_range[3] = {run.local1, run.local2, run.local3};
    checkCLError(clEnqueueNDRangeKernel(_queue, _kernel, 3, NULL, global_range,
                                        local_range, 0, NULL, &ev));
    clWaitForEvents(1, &ev);

    // check the event:
    cl_int status;
    checkCLError(clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                sizeof(cl_int), &status, NULL));
    switch (status) {
    case CL_QUEUED:
      std::cout << "+++++++ Event CL_QUEUED\n";
      break;
    case CL_SUBMITTED:
      std::cout << "+++++++ Event CL_SUBMITTED\n";
      break;
    case CL_RUNNING:
      std::cout << "+++++++ Event CL_RUNNING\n";
      break;
    case CL_COMPLETE:
      std::cout << "+++++++ Event CL_COMPLETE\n";
      break;
    default:
      std::cout << "+++++++ EVENT FAILED WITH ERROR CODE: " << status << "\n";
    }

    // find how long the kernel took.
    cl_ulong start;
    cl_ulong end;
    checkCLError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), (void *)&start,
                                         NULL));
    checkCLError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), (void *)&end, NULL));

    report_timing(clEnqueueNDRangeKernel, harness, end - start);

    std::chrono::nanoseconds elapsed_ns(end - start);
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_ns);
  }

  cl_mem createAndUploadGlobalArg(std::vector<char> &arg, bool output = false) {
    start_timer(createAndUploadGlobalArg, harness);
    // get a pointer to the underlying arg:
    char *data = arg.data();
    size_t len = arg.size() * sizeof(char);

    std::cout << "creating arg of size: " << len
              << " from pointer: " << static_cast<void *>(data) << "\n";

    // create a mem argument
    cl_mem_flags flags = output ? CL_MEM_READ_WRITE : CL_MEM_READ_ONLY;
    cl_mem buffer =
        clCreateBuffer(_context, CL_MEM_READ_WRITE, (size_t)len, NULL, &_error);
    checkCLError(_error);

    // enqueue a write into that buffer
    cl_event ev; // do something with this event eventually!
    checkCLError(clEnqueueWriteBuffer(_queue, buffer, CL_TRUE, 0, len, data, 0,
                                      NULL, &ev));

    clWaitForEvents(1, &ev);

    // find how long the copy took.
    cl_ulong start;
    cl_ulong end;
    checkCLError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), (void *)&start,
                                         NULL));
    checkCLError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), (void *)&end, NULL));

    report_timing(clEnqueueWriteBuffer, createAndUploadGlobalArg, end - start);
    return buffer;
  }

  void writeToGlobalArg(std::vector<char> &arg, cl_mem buffer) {
    start_timer(writeToGlobalArg, harness);
    // get a pointer to the underlying arg:
    char *data = arg.data();
    size_t len = arg.size() * sizeof(char);

    std::cout << "uploading arg of size: " << len
              << " from pointer: " << static_cast<void *>(data) << "\n";

    // enqueue a write into that buffer
    cl_event ev; // do something with this event eventually!
    checkCLError(clEnqueueWriteBuffer(_queue, buffer, CL_TRUE, 0, len, data, 0,
                                      NULL, &ev));

    clWaitForEvents(1, &ev);

    // find how long the copy took.
    cl_ulong start;
    cl_ulong end;
    checkCLError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), (void *)&start,
                                         NULL));
    checkCLError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), (void *)&end, NULL));

    report_timing(clEnqueueWriteBuffer, writeToGlobalArg, end - start);
  }

  void readFromGlobalArg(std::vector<char> &arg, cl_mem buffer) {
    start_timer(readFromGlobalArg, harness);
    // get a pointer to the underlying arg:
    char *data = arg.data();
    size_t len = arg.size() * sizeof(char);

    std::cout << "downloading arg of size: " << len
              << " into pointer: " << static_cast<void *>(data) << "\n";

    cl_event ev;
    checkCLError(clEnqueueReadBuffer(_queue, buffer, CL_TRUE, 0, len, data, 0,
                                     NULL, &ev));
    clWaitForEvents(1, &ev);

    // find how long the copy took.
    cl_ulong start;
    cl_ulong end;
    checkCLError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), (void *)&start,
                                         NULL));
    checkCLError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), (void *)&end, NULL));

    report_timing(clEnqueueReadBuffer, readFromGlobalArg, end - start);
  }

  cl_mem createGlobalArg(unsigned int size) {
    start_timer(createGlobalArg, harness);
    cl_mem buffer = clCreateBuffer(_context, CL_MEM_READ_WRITE, (size_t)size,
                                   NULL, &_error);
    checkCLError(_error);

    return buffer;
  }

  void setGlobalArg(cl_int arg, cl_mem *mem) {
    start_timer(setGlobalArg, harness);
    std::cout << "setting arg : " << arg << " from memory "
              << static_cast<void *>(mem) << "\n";
    checkCLError(clSetKernelArg(_kernel, arg, sizeof(cl_mem), mem));
  }

  template <typename ValueType> void setValueArg(cl_uint arg, ValueType *val) {
    start_timer(setValueArg, harness);
    checkCLError(clSetKernelArg(_kernel, arg, sizeof(ValueType), val));
  }

  void setLocalArg(cl_uint arg, size_t size) {
    start_timer(setLocalArg, harness);
    checkCLError(clSetKernelArg(_kernel, arg, size, NULL));
  }

  // stateful error code :(
  cl_int _error;
  std::string _kernel_source;
  cl_kernel _kernel;
  cl_uint _platformIdCount;
  cl_uint _deviceIdCount;
  cl_command_queue _queue;

  std::vector<cl_device_id> _deviceIds;

  cl_context _context;
  ArgContainer<SemiRingType> _args;
};

// template <typename T> class IterativeHarness : public
// Harness<std::vector<T>>
// {
template <typename TimingType, typename SemiRingType>
class IterativeHarness : public Harness<TimingType, SemiRingType> {
public:
  IterativeHarness(std::string &kernel_source, unsigned int platform,
                   unsigned int device, ArgContainer<SemiRingType> args)
      : Harness<TimingType, SemiRingType>(kernel_source, platform, device,
                                          args) {}

protected:
  virtual bool should_terminate_iteration(std::vector<char> &input,
                                          std::vector<char> &output,
                                          double delta) = 0;
};