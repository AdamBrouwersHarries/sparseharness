#pragma once
#include "CL/cl.h"
#include <string>
#include <vector>

#include "Assert.h"
#include "Logger.h"

// give the function names
std::string getErrorString(cl_int);
void checkCLError(cl_int);

//  Check to see if an error code is correct etc
void checkCLError(cl_int error) {
  if (error != CL_SUCCESS) {
    LOG_ERROR("OpenCL call failed with error ", getErrorString(error));
    std::exit(1);
  }
}
// get the string associated with an OpenCL error code
std::string getErrorString(cl_int err) {
  std::ostringstream ostr;
  switch (err) {
  case CL_SUCCESS:
    ostr << "CL_SUCCESS";
    break;
  case CL_DEVICE_NOT_FOUND:
    ostr << "CL_DEVICE_NOT_FOUND";
    break;
  case CL_DEVICE_NOT_AVAILABLE:
    ostr << "CL_DEVICE_NOT_AVAILABLE";
    break;
  case CL_COMPILER_NOT_AVAILABLE:
    ostr << "CL_COMPILER_NOT_AVAILABLE";
    break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    ostr << "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    break;
  case CL_OUT_OF_RESOURCES:
    ostr << "CL_OUT_OF_RESOURCES";
    break;
  case CL_OUT_OF_HOST_MEMORY:
    ostr << "CL_OUT_OF_HOST_MEMORY";
    break;
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    ostr << "CL_PROFILING_INFO_NOT_AVAILABLE";
    break;
  case CL_MEM_COPY_OVERLAP:
    ostr << "CL_MEM_COPY_OVERLAP";
    break;
  case CL_IMAGE_FORMAT_MISMATCH:
    ostr << "CL_IMAGE_FORMAT_MISMATCH";
    break;
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    ostr << "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    break;
  case CL_BUILD_PROGRAM_FAILURE:
    ostr << "CL_BUILD_PROGRAM_FAILURE";
    break;
  case CL_MAP_FAILURE:
    ostr << "CL_MAP_FAILURE";
    break;
  case CL_INVALID_VALUE:
    ostr << "CL_INVALID_VALUE";
    break;
  case CL_INVALID_DEVICE_TYPE:
    ostr << "CL_INVALID_DEVICE_TYPE";
    break;
  case CL_INVALID_PLATFORM:
    ostr << "CL_INVALID_PLATFORM";
    break;
  case CL_INVALID_DEVICE:
    ostr << "CL_INVALID_DEVICE";
    break;
  case CL_INVALID_CONTEXT:
    ostr << "CL_INVALID_CONTEXT";
    break;
  case CL_INVALID_QUEUE_PROPERTIES:
    ostr << "CL_INVALID_QUEUE_PROPERTIES";
    break;
  case CL_INVALID_COMMAND_QUEUE:
    ostr << "CL_INVALID_COMMAND_QUEUE";
    break;
  case CL_INVALID_HOST_PTR:
    ostr << "CL_INVALID_HOST_PTR";
    break;
  case CL_INVALID_MEM_OBJECT:
    ostr << "CL_INVALID_MEM_OBJECT";
    break;
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    ostr << "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    break;
  case CL_INVALID_IMAGE_SIZE:
    ostr << "CL_INVALID_IMAGE_SIZE";
    break;
  case CL_INVALID_SAMPLER:
    ostr << "CL_INVALID_SAMPLER";
    break;
  case CL_INVALID_BINARY:
    ostr << "CL_INVALID_BINARY";
    break;
  case CL_INVALID_BUILD_OPTIONS:
    ostr << "CL_INVALID_BUILD_OPTIONS";
    break;
  case CL_INVALID_PROGRAM:
    ostr << "CL_INVALID_PROGRAM";
    break;
  case CL_INVALID_PROGRAM_EXECUTABLE:
    ostr << "CL_INVALID_PROGRAM_EXECUTABLE";
    break;
  case CL_INVALID_KERNEL_NAME:
    ostr << "CL_INVALID_KERNEL_NAME";
    break;
  case CL_INVALID_KERNEL_DEFINITION:
    ostr << "CL_INVALID_KERNEL_DEFINITION";
    break;
  case CL_INVALID_KERNEL:
    ostr << "CL_INVALID_KERNEL";
    break;
  case CL_INVALID_ARG_INDEX:
    ostr << "CL_INVALID_ARG_INDEX";
    break;
  case CL_INVALID_ARG_VALUE:
    ostr << "CL_INVALID_ARG_VALUE";
    break;
  case CL_INVALID_ARG_SIZE:
    ostr << "CL_INVALID_ARG_SIZE";
    break;
  case CL_INVALID_KERNEL_ARGS:
    ostr << "CL_INVALID_KERNEL_ARGS";
    break;
  case CL_INVALID_WORK_DIMENSION:
    ostr << "CL_INVALID_WORK_DIMENSION";
    break;
  case CL_INVALID_WORK_GROUP_SIZE:
    ostr << "CL_INVALID_WORK_GROUP_SIZE";
    break;
  case CL_INVALID_WORK_ITEM_SIZE:
    ostr << "CL_INVALID_WORK_ITEM_SIZE";
    break;
  case CL_INVALID_GLOBAL_OFFSET:
    ostr << "CL_INVALID_GLOBAL_OFFSET";
    break;
  case CL_INVALID_EVENT_WAIT_LIST:
    ostr << "CL_INVALID_EVENT_WAIT_LIST";
    break;
  case CL_INVALID_EVENT:
    ostr << "CL_INVALID_EVENT";
    break;
  case CL_INVALID_OPERATION:
    ostr << "CL_INVALID_OPERATION";
    break;
  case CL_INVALID_GL_OBJECT:
    ostr << "CL_INVALID_GL_OBJECT";
    break;
  case CL_INVALID_BUFFER_SIZE:
    ostr << "CL_INVALID_BUFFER_SIZE";
    break;
  case CL_INVALID_MIP_LEVEL:
    ostr << "CL_INVALID_MIP_LEVEL";
    break;
  default:
    ostr << "unknown error";
  }
  ostr << " (code: " << err << ")";
  return ostr.str();
}

template <typename T>
void printCharVector(const std::string &name, std::vector<char> &v) {
  // get the underlying pointer, and the length in terms of t
  // then recast in terms of T
  T *data = reinterpret_cast<T *>(v.data());
  unsigned int length = (v.size() * sizeof(char)) / sizeof(T);
  // print it out
  std::ostringstream ostr;
  ostr << "[";
  for (int i = 0; i < length; i++) {
    if (i != 0) {
      ostr << ",";
    }
    ostr << data[i];
  }
  ostr << "]";
  LOG_DEBUG_INFO("Buffer ", name, ostr.str());
}

void assertBuffersNotEqual(std::vector<char> &v1, std::vector<char> &v2) {
  bool different_found = false;
  for (int i = 0; i < v1.size(); i++) {
    if (v1[i] != v2[i]) {
      different_found = true;
      break;
    }
  }
  if (!different_found) {
    LOG_ERROR("Buffers are identical - kernel has probably failed...");
  }
}