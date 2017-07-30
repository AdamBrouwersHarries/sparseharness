#pragma once

#include <cassert>
#include <iostream>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "run.h"

class OpenCL {

  static cl::Context context;
  static cl::CommandQueue queue;
  static std::vector<cl::Device> devices;
  static cl::Device &device;

  static std::size_t device_max_workgroup_size;
  static std::vector<size_t> device_max_work_item_sizes;

  static cl_ulong device_local_mem_size;

  static int iterations;

public:
  // Initialise the opencl status and default devices
  static void init(const unsigned platform_idx, const unsigned device_idx,
                   const unsigned iters = 10) {
    iterations = iters;
    std::vector<cl::Platform> platform;
    cl::Platform::get(&platform);

    assert(platform.size() >= platform_idx + 1 && "platform not found");

    platform[platform_idx].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    assert(devices.size() >= device_idx + 1 && "Device not found");

    devices = {devices[device_idx]};
    context = cl::Context(devices);
    auto &device = devices.front();
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    device_local_mem_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    device_max_workgroup_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    device_max_work_item_sizes =
        device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

    std::cout << "Executing on " << device.getInfo<CL_DEVICE_NAME>()
              << std::endl;
  }

  // Create an opencl buffer object
  static cl::Buffer alloc(cl_mem_flags flags, std::size_t size,
                          void *data = nullptr) {
    return cl::Buffer(context, flags, size, data);
  }

  // Compile a string into an OpenCL kernel object
  static cl::Kernel compileSource(const std::string &source) {
    cl::Program program;
    cl::Kernel kernel;
    try {
      program = cl::Program(
          context, cl::Program::Sources(
                       1, std::make_pair(source.data(), source.size())));
      program.build(devices);
      kernel = cl::Kernel(program, "KERNEL");
    } catch (const cl::Error &err) {
      try {
        // should this really be devices.front?
        const std::string what =
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        // Nvidia doesn't compile the code if it uses too much memory (ptxas
        // error)
        if (what.find("uses too much shared data") != std::string::npos) {
          std::cerr << "[INCOMPATIBLE] Compatibility check failed\n";
        } else {
          std::cerr << "[COMPILE-ERROR] Compilation failed \n ";
        }
      } // the getBuildInfo might also fail
      catch (const cl::Error &err) {
        std::cerr << "[COMPILE-ERROR] getBuildInfo failed\n";
      }
    }
    return kernel;
  }

  // compatiblity checks for a Kernel
  enum KernelCompatibility {
    COMPATIBLE,
    INVALID_WORKGROUP_SIZE,
    INVALID_MEM_SIZE
  };

  static KernelCompatibility
  check_compatibility(cl::Kernel kernel, size_t num_work_items,
                      size_t sum_local_mem_kernel_args = 0) {
    size_t wg_size = 0;
    cl_ulong local_size = 0;

    kernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &wg_size);
    kernel.getWorkGroupInfo(device, CL_KERNEL_LOCAL_MEM_SIZE, &local_size);

    if (wg_size == 0 || num_work_items > device_max_workgroup_size)
      return INVALID_WORKGROUP_SIZE;

    if (local_size > device_local_mem_size)
      return INVALID_MEM_SIZE;

    // OpenCL API does not seem to return the correct size for local memory
    // specified
    // as kernel args. Manually implement check here:
    if (sum_local_mem_kernel_args > device_local_mem_size)
      return INVALID_MEM_SIZE;
    return COMPATIBLE;
  }

  static void executeKernel(cl::Kernel kernel, Run &run) {
    // check compatibility
    auto cmpt = OpenCL::check_compatibility(kernel, run.num_work_items());
    if (cmpt != COMPATIBLE) {
      // report the error, and fail
      std::cout << "Invalid kernel!" << std::endl;
    }
    static double time;
    // auto llocals
  }
};