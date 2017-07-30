#pragma once

#include "OpenCL_utils.h"

class PreparedKernel {
  cl::Kernel compiled_kernel;
  cl::Buffer output;
};
