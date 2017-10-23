#pragma once

#include "kernel_utils.h"
#include "opencl_utils.h"

template <typename SemiringType> class CLMemoryManager {
public:
  CLMemoryManager(ArgContainer<SemiringType> &args)
      : _args(args), _temp_global(_args.temp_globals.size()),
        _input_host_buffer(args.x_vect.begin(), args.x_vect.end()),
        _output_host_buffer(_args.output, 0),
        _temp_out_buffer(_args.output, 0) {}

  ArgContainer<SemiringType> &_args;
  cl_mem _matrix_idxs;
  cl_mem _matrix_vals;
  cl_mem _x_vect;
  cl_mem _y_vect;
  cl_mem _output;
  std::vector<cl_mem> _temp_global;

  cl_uint _arg_index = 0;
  cl_uint _input_idx = 2;
  cl_uint _output_idx = 0;

  std::vector<char> _input_host_buffer;
  std::vector<char> _output_host_buffer;
  std::vector<char> _temp_out_buffer;
};