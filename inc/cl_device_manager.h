#pragma once

#include "kernel_utils.h"
#include "opencl_utils.h"

// manage a device, context and queue in a transparent way, that is easily moved
class CLDeviceManager {
public:
  CLDeviceManager(unsigned int platform, unsigned int device)
      : _device(device) {
    // initialise OpenCL:
    // get the number of platforms
    _platformIdCount = 0;
    clGetPlatformIDs(0, nullptr, &_platformIdCount);

    if (_platformIdCount == 0) {
      LOG_ERROR("No OpenCL devices found!");
      exit(1);
    }

    LOG_DEBUG_INFO("Found ", _platformIdCount, " platforms");

    // make a vector of platform ids
    std::vector<cl_platform_id> platformIds(_platformIdCount);
    clGetPlatformIDs(_platformIdCount, platformIds.data(), nullptr);

    // get the number of devices from the platform
    _deviceIdCount = 0;
    clGetDeviceIDs(platformIds[platform], CL_DEVICE_TYPE_ALL, 0, nullptr,
                   &_deviceIdCount);

    LOG_DEBUG_INFO("Found ", _deviceIdCount, " devices on the chosen platform");

    // get a list of devices from the platform
    _deviceIds.resize(_deviceIdCount);
    clGetDeviceIDs(platformIds[platform], CL_DEVICE_TYPE_ALL, _deviceIdCount,
                   _deviceIds.data(), nullptr);

    LOG_INFO("Running on OpenCL device: ", getDeviceName());

    // create a context on that device (with some properties)
    const cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platformIds[platform]), 0, 0};

    _context = clCreateContext(contextProperties, _deviceIdCount,
                               _deviceIds.data(), nullptr, nullptr, &_error);
    checkCLError(_error);

    // finally, create a command queue from the device and context);
    _device_id = _deviceIds[_device];
    _queue = clCreateCommandQueue(_context, _deviceIds[_device],
                                  CL_QUEUE_PROFILING_ENABLE, &_error);
    checkCLError(_error);
  }

  std::string getDeviceName() {
    char name[10240];
    LOG_DEBUG_INFO("Getting device name from device ", _device_id);
    _error = clGetDeviceInfo(_deviceIds[_device], CL_DEVICE_NAME, sizeof(name),
                             name, NULL);
    checkCLError(_error);
    return std::string(name);
  }

  unsigned long getMaxMemAllocSize() {
    cl_ulong size;
    LOG_DEBUG_INFO("Getting device max alloc size from device", _device_id);
    _error = clGetDeviceInfo(_deviceIds[_device], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                             sizeof(size), &size, NULL);
    checkCLError(_error);
    return size;
  }

  cl_int _error;

  cl_uint _platformIdCount;
  cl_uint _deviceIdCount;
  cl_uint _device;
  cl_command_queue _queue;

  cl_device_id _device_id;
  std::vector<cl_device_id> _deviceIds;

  cl_context _context;
};