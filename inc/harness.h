#pragma once

#include "cl_memory_manager.h"
#include "cl_device_manager.h"
#include "kernel_utils.h"
#include "opencl_utils.h"
#include "sql_stat.h"

#include "run.h"
#include <chrono>

template <typename TimingType, typename SemiRingType> class Harness {
public:
  Harness(std::string &kernel_source, CLDeviceManager &cldm,
          ArgContainer<SemiRingType> args, unsigned int trials,
          unsigned int timeout, double delta)
      : _kernel_source(kernel_source), _cldm(cldm), _args(args),
        _mem_manager(args), _trials(trials), _timeout(timeout), _delta(delta) {

    // create a kernel from the source

    // recast the std::string as a size_t array and char array
    size_t lengths[1] = {_kernel_source.size()};
    const char *sources[1] = {_kernel_source.data()};
    // create the program
    cl_program program = clCreateProgramWithSource(_cldm._context, 1, sources,
                                                   lengths, &_cldm._error);
    checkCLError(_cldm._error);

    // build the program
    _cldm._error =
        clBuildProgram(program, _cldm._deviceIdCount, _cldm._deviceIds.data(),
                       "", nullptr, nullptr);
    checkCLError(_cldm._error);

    // create a kernel from the program
    cl_kernel kernel = clCreateKernel(program, "KERNEL", &_cldm._error);
    checkCLError(_cldm._error);
    _kernel = kernel;
  }

  virtual std::vector<TimingType> benchmark(Run run) = 0;

  // lower the timeout based on new information about the best time
  // we check (first) to see whether it's within 2x of the new value
  // if it is, we update the timeout to reflect this.
  // the idea is that a 2x difference could be noise (probably), so if it's
  // still within that we should be okay with it
  void lowerTimeout(unsigned int measured_time) {
    if (measured_time * 2 < _timeout) {
      _timeout = measured_time * 2;
    }
  }

  std::string getDeviceName() { return _cldm.getDeviceName(); }

protected:
  virtual TimingType executeRun(Run run, unsigned int trial) = 0;

  std::chrono::nanoseconds executeKernel(Run run) {
    start_timer(executeKernel, harness);

    cl_event ev;
    const size_t global_range[3] = {run.global1, run.global2, run.global3};
    const size_t local_range[3] = {run.local1, run.local2, run.local3};
    checkCLError(clEnqueueNDRangeKernel(_cldm._queue, _kernel, 3, NULL,
                                        global_range, local_range, 0, NULL,
                                        &ev));
    clWaitForEvents(1, &ev);

    // check the event:
    cl_int status;
    checkCLError(clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                sizeof(cl_int), &status, NULL));
    switch (status) {
    case CL_QUEUED:
      LOG_DEBUG_INFO("Event CL_QUEUED");
      break;
    case CL_SUBMITTED:
      LOG_DEBUG_INFO("Event CL_SUBMITTED");
      break;
    case CL_RUNNING:
      LOG_DEBUG_INFO("Event CL_RUNNING");
      break;
    case CL_COMPLETE:
      LOG_DEBUG_INFO("Event CL_COMPLETE");
      break;
    default:
      LOG_WARNING("EVENT FAILED WITH ERROR CODE: ", getErrorString(status));
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
    return elapsed_ns;
  }

  void allocateBuffers() {
    start_timer(allocateBuffers, Harness);
    cl_uint arg_index = 0;
    // build the matrix arguments
    LOG_DEBUG_INFO("setting matrix arguments");
    _mem_manager._matrix_idxs = createAndUploadGlobalArg(_args.m_idxs);
    setGlobalArg(arg_index++, &_mem_manager._matrix_idxs);

    _mem_manager._matrix_vals = createAndUploadGlobalArg(_args.m_vals);
    setGlobalArg(arg_index++, &_mem_manager._matrix_vals);

    // build the vector arguments
    LOG_DEBUG_INFO("setting vector arguments");
    _mem_manager._x_vect = createAndUploadGlobalArg(_args.x_vect, true);
    setGlobalArg(arg_index++, &_mem_manager._x_vect);

    _mem_manager._y_vect = createAndUploadGlobalArg(_args.y_vect, true);
    setGlobalArg(arg_index++, &_mem_manager._y_vect);

    // build the constant arguments
    LOG_DEBUG_INFO("setting constant arguments");
    setValueArg<SemiRingType>(arg_index++, &(_args.alpha));
    setValueArg<SemiRingType>(arg_index++, &(_args.beta));

    // set the output arg
    LOG_DEBUG_INFO("setting the output argument");
    _mem_manager._output_idx = arg_index;
    _mem_manager._output = createGlobalArg(_args.output);
    setGlobalArg(arg_index++, &_mem_manager._output);

    // set the temp globals and write zeros into them
    LOG_DEBUG_INFO("setting ", _args.temp_globals.size(),
                   " temp global arguments");
    int temp_index = 0;
    for (auto size : _args.temp_globals) {
      _mem_manager._temp_global[temp_index] = createGlobalArg(size);
      fillGlobalArg(size, _mem_manager._temp_global[temp_index]);
      setGlobalArg(arg_index++, &(_mem_manager._temp_global[temp_index]));
      temp_index++;
    }

    // build temp locals
    LOG_DEBUG_INFO("setting ", _args.temp_locals.size(),
                   " temp local arguments");
    for (auto size : _args.temp_locals) {
      setLocalArg(arg_index++, size);
    }

    // set the size arguments
    LOG_DEBUG_INFO("setting size arguments");
    for (auto size : _args.size_args) {
      setValueArg<unsigned int>(arg_index++, &(size));
    }
  }

  void resetPointers() {}

  void resetTempBuffers() {
    start_timer(resetTempBuffers, Harness);

    // set the temporary inputs to zero
    int temp_index = 0;
    for (auto arg : _mem_manager._temp_global) {
      start_timer(fillGlobalArg, resetTempBuffers);
      fillGlobalArg(_args.temp_globals[temp_index], arg);
      temp_index++;
    }
  }

  cl_mem createAndUploadGlobalArg(std::vector<char> &arg, bool output = false) {
    start_timer(createAndUploadGlobalArg, harness);
    // get a pointer to the underlying arg:
    char *data = arg.data();
    size_t len = arg.size() * sizeof(char);

    LOG_DEBUG_INFO("Creating arg of size ", len, " from pointer ",
                   static_cast<void *>(data));

    // create a mem argument
    cl_mem_flags flags = output ? CL_MEM_READ_WRITE : CL_MEM_READ_ONLY;
    cl_mem buffer =
        clCreateBuffer(_cldm._context, flags, (size_t)len, NULL, &_cldm._error);
    checkCLError(_cldm._error);

    // enqueue a write into that buffer
    cl_event ev; // do something with this event eventually!
    checkCLError(clEnqueueWriteBuffer(_cldm._queue, buffer, CL_TRUE, 0, len,
                                      data, 0, NULL, &ev));

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

    LOG_DEBUG_INFO("uploading arg of size ", len, " from pointer ",
                   static_cast<void *>(data));

    // enqueue a write into that buffer
    cl_event ev; // do something with this event eventually!
    checkCLError(clEnqueueWriteBuffer(_cldm._queue, buffer, CL_TRUE, 0, len,
                                      data, 0, NULL, &ev));

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

  // NOTE - THIS DEPENDS ON OpenCL 1.2 functionality and therefore may not
  // work on some NVIDIA platforms. See problems such as this:
  // https://stackoverflow.com/questions/32145522/compiling-opencl-1-2-codes-on-nvidia-gpus
  // This code therfore may need to be rewritten at some point!
  void fillGlobalArg(size_t buffer_size, cl_mem buffer) {
    start_timer(fillGlobalArg, harness);
    LOG_DEBUG_INFO("filling buffer with ", buffer_size, " bytes of zeros");

    char pattern = 0;

    cl_event ev;
    checkCLError(clEnqueueFillBuffer(_cldm._queue, buffer, &pattern,
                                     sizeof(char), 0, buffer_size, 0, NULL,
                                     &ev));

    clWaitForEvents(1, &ev);

    // find how long the copy took.
    cl_ulong start;
    cl_ulong end;
    checkCLError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), (void *)&start,
                                         NULL));
    checkCLError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), (void *)&end, NULL));

    report_timing(clEnqueueFillBuffer, fillGlobalArg, end - start);
  }

  void readFromGlobalArg(std::vector<char> &arg, cl_mem buffer) {
    start_timer(readFromGlobalArg, harness);
    // get a pointer to the underlying arg:
    char *data = arg.data();
    size_t len = arg.size() * sizeof(char);

    LOG_DEBUG_INFO("downloading arg of size: ", len, " into pointer ",
                   static_cast<void *>(data));

    cl_event ev;
    checkCLError(clEnqueueReadBuffer(_cldm._queue, buffer, CL_TRUE, 0, len,
                                     data, 0, NULL, &ev));
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
    LOG_DEBUG_INFO("creating global arg of size ", size);

    cl_mem buffer = clCreateBuffer(_cldm._context, CL_MEM_READ_WRITE,
                                   (size_t)size, NULL, &_cldm._error);
    checkCLError(_cldm._error);

    return buffer;
  }

  void setGlobalArg(cl_int arg, cl_mem *mem) {
    start_timer(setGlobalArg, harness);
    LOG_DEBUG_INFO("setting global arg ", arg, " from memory ",
                   static_cast<void *>(mem), "with size: ", sizeof(cl_mem));
    checkCLError(clSetKernelArg(_kernel, arg, sizeof(cl_mem), mem));
  }

  template <typename ValueType> void setValueArg(cl_uint arg, ValueType *val) {
    start_timer(setValueArg, harness);
    LOG_DEBUG_INFO("setting value arg of with value ", *val);
    checkCLError(clSetKernelArg(_kernel, arg, sizeof(ValueType), val));
  }

  void setLocalArg(cl_uint arg, size_t size) {
    start_timer(setLocalArg, harness);
    LOG_DEBUG_INFO("setting local arg of size ", size);

    checkCLError(clSetKernelArg(_kernel, arg, size, NULL));
  }

  // stateful error code :(
  std::string _kernel_source;
  cl_kernel _kernel;

  ArgContainer<SemiRingType> _args;

  CLMemoryManager<SemiRingType> _mem_manager;

  CLDeviceManager &_cldm;

  unsigned int _trials;
  unsigned int _timeout;
  double _delta;
};

// template <typename T> class IterativeHarness : public
// Harness<std::vector<T>>
// {
template <typename TimingType, typename SemiRingType>
class IterativeHarness : public Harness<TimingType, SemiRingType> {
public:
  IterativeHarness(std::string &kernel_source, CLDeviceManager cldm,
                   ArgContainer<SemiRingType> args, unsigned int trials,
                   unsigned int timeout, double delta)
      : Harness<TimingType, SemiRingType>(kernel_source, cldm, args, trials,
                                          timeout, delta) {}

protected:
  virtual bool should_terminate_iteration(std::vector<char> &input,
                                          std::vector<char> &output) = 0;

  void resetInputs() {
    start_timer(allocateBuffers, Harness);
    // build the matrix arguments
    LOG_DEBUG_INFO("setting matrix arguments");
    // _mem_manager._matrix_idxs = createAndUploadGlobalArg(_args.m_idxs);
    // setGlobalArg(arg_index++, &_mem_manager._matrix_idxs);
    this->writeToGlobalArg(this->_args.m_idxs, this->_mem_manager._matrix_idxs);

    // this->_mem_manager._matrix_vals =
    // createAndUploadGlobalArg(Harness<TimingType,
    // SemiRingType>::_args.m_vals);
    // setGlobalArg(arg_index++, &Harness<TimingType,
    // SemiRingType>::_mem_manager._matrix_vals);
    this->writeToGlobalArg(this->_args.m_vals, this->_mem_manager._matrix_vals);

    // build the vector arguments
    LOG_DEBUG_INFO("setting vector arguments");
    // this->_mem_manager._x_vect =
    // createAndUploadGlobalArg(Harness<TimingType,
    // SemiRingType>::_args.x_vect, true);
    this->setGlobalArg(2, &this->_mem_manager._x_vect);
    this->writeToGlobalArg(this->_args.x_vect, this->_mem_manager._x_vect);

    // this->_mem_manager._y_vect =
    // createAndUploadGlobalArg(Harness<TimingType,
    // SemiRingType>::_args.y_vect, true);
    this->setGlobalArg(3, &this->_mem_manager._y_vect);
    this->writeToGlobalArg(this->_args.y_vect, this->_mem_manager._y_vect);

    // build the constant arguments
    // LOG_DEBUG_INFO("setting constant arguments");
    // setValueArg<float>(arg_index++, &(Harness<TimingType,
    // SemiRingType>::_args.alpha));
    // setValueArg<float>(arg_index++, &(Harness<TimingType,
    // SemiRingType>::_args.beta));

    // set the output arg
    LOG_DEBUG_INFO("setting the output argument");
    // this->_mem_manager._output_idx = arg_index;
    // this->_mem_manager._output =
    // createGlobalArg(Harness<TimingType,
    // SemiRingType>::_args.output);
    this->setGlobalArg(6, &this->_mem_manager._output);
    this->fillGlobalArg(this->_args.output, this->_mem_manager._output);

    this->resetTempBuffers();
  }
};