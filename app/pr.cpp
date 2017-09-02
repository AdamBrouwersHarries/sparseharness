// // [standard includes]
// #include <algorithm>
// #include <atomic>
// #include <bitset>
// #include <cmath>
// #include <condition_variable>
// #include <cstdio>
// #include <fstream>
// #include <iostream>
// #include <memory>
// #include <mutex>
// #include <queue>
// #include <set>
// #include <sstream>
// #include <thread>
// #include <typeinfo>
// #include <vector>

// // [tools]
// #include "common.h"
// #include "csds_timer.h"
// #include "csv_utils.h"
// #include "options.h"

// // [ocl tools]
// #include "harness.h"
// #include "kernel_config.h"
// #include "kernel_utils.h"

// // [application specific]
// #include "run.h"
// #include "sparse_matrix.h"
// #include "vector_generator.h"

// // [OpenCL]
// #ifdef __APPLE__
// #include "OpenCL/opencl.h"
// #else
// #include "CL/cl.h"
// #endif

// class HarnessPR : public IterativeHarness<std::chrono::nanoseconds, float> {
// public:
//   HarnessPR(std::string &kernel_source, unsigned int platform,
//             unsigned int device, ArgContainer<float> args)
//       : IterativeHarness(kernel_source, platform, device, args) {}
//   std::vector<std::chrono::nanoseconds>
//   benchmark(Run run, int trials, double timeout, double delta) {

//     start_timer(benchmark, HarnessPR);

//     // set up the kernel
//     cl_uint arg_index = 0;
//     cl_uint input_idx = 2;
//     cl_uint output_idx = 0;

//     // build the matrix arguments
//     cl_mem matrix_idxs_ma = createAndUploadGlobalArg(_args.m_idxs);
//     setGlobalArg(arg_index++, &matrix_idxs_ma);

//     cl_mem matrix_vals_ma = createAndUploadGlobalArg(_args.m_vals);
//     setGlobalArg(arg_index++, &matrix_vals_ma);

//     // build the vector arguments
//     cl_mem x_vect_ma = createAndUploadGlobalArg(_args.x_vect, true);
//     setGlobalArg(arg_index++, &x_vect_ma);

//     cl_mem y_vect_ma = createAndUploadGlobalArg(_args.y_vect, true);
//     setGlobalArg(arg_index++, &y_vect_ma);

//     // build the constant arguments
//     setValueArg<float>(arg_index++, &(_args.alpha));
//     setValueArg<float>(arg_index++, &(_args.beta));

//     // build temp globals
//     // do we need to save them at all???
//     // std::vector<cl_mem> temp_globals(_args.temp_globals.size());
//     // THIS MIGHT SEGFAULT AWFULLY :D
//     for (auto size : _args.temp_globals) {
//       cl_mem temp_arg = createGlobalArg(size);
//       setGlobalArg(arg_index++, &temp_arg);
//     }

//     // set the output arg
//     output_idx = arg_index;
//     cl_mem output_mem = createGlobalArg(_args.output);
//     setGlobalArg(arg_index++, &output_mem);

//     // build temp locals
//     for (auto size : _args.temp_locals) {
//       setLocalArg(arg_index++, size);
//     }

//     // set the size arguments
//     for (auto size : _args.size_args) {
//       setValueArg<unsigned int>(arg_index++, &(size));
//     }

//     // finally, create buffers to copy the input and output into
//     std::vector<char> input_host_buffer(_args.x_vect.begin(),
//                                         _args.x_vect.end());
//     std::vector<char> output_host_buffer(_args.output, 0);

//     std::vector<char> blank_output_buffer(_args.output, 0);

//     // run the kernel!
//     std::vector<std::chrono::nanoseconds> runtimes;
//     for (int t = 0; t < trials; t++) {
//       start_timer(benchmark_iteration, HarnessPR);

//       // get pointers to the input + output mem args
//       cl_mem *input_mem_ptr = &x_vect_ma;
//       cl_mem *output_mem_ptr = &output_mem;

//       // and pointers to the input + output host args
//       std::vector<char> *input_host_ptr = &input_host_buffer;
//       std::vector<char> *output_host_ptr = &output_host_buffer;

//       bool should_terminate = false;
//       int itcnt = 0;
//       do {
//         LOG_DEBUG_INFO("Host vectors before");
//         printCharVector<float>("Input ", *input_host_ptr);
//         printCharVector<float>("Output ", *output_host_ptr);

//         // run the kernel
//         runtimes.push_back(executeRun(run, t));

//         // copy the output back down
//         readFromGlobalArg(*output_host_ptr, *output_mem_ptr);

//         LOG_DEBUG_INFO("Host vectors after");
//         printCharVector<float>("Input ", *input_host_ptr);
//         printCharVector<float>("Output ", *output_host_ptr);

//         should_terminate = should_terminate_iteration(*input_host_ptr,
//                                                       *output_host_ptr,
//                                                       delta);
//         // swap the pointers over
//         std::swap(input_mem_ptr, output_mem_ptr);
//         std::swap(input_host_ptr, output_host_ptr);

//         // reupload the input
//         // writeToGlobalArg(*input_host_ptr, *input_mem_ptr);
//         // writeToGlobalArg(blank_output_buffer, *output_mem_ptr);

//         // set the kernel args
//         setGlobalArg(input_idx, input_mem_ptr);
//         setGlobalArg(output_idx, output_mem_ptr);

//         itcnt++;
//       } while (!should_terminate && itcnt < 300);
//     }
//     return runtimes;
//   }

// private:
//   virtual std::chrono::nanoseconds executeRun(Run run, unsigned int trial) {
//     return executeKernel(run);
//   }

//   virtual bool should_terminate_iteration(std::vector<char> &input,
//                                           std::vector<char> &output,
//                                           double delta) {
//     start_timer(should_terminate_iteration, HarnessPR);

//     // reinterpret the args as double pointers, and get the lengths
//     auto input_ptr = reinterpret_cast<float *>(input.data());
//     auto output_ptr = reinterpret_cast<float *>(output.data());
//     auto input_length = input.size() / sizeof(float);
//     auto output_length = output.size() / sizeof(float);
//     // perform a comparison across the two of them, based on pointers
//     bool equal = true;
//     for (unsigned int i = 0;
//          equal == true && i < input_length && i < output_length; i++) {
//       equal = fabs(input_ptr[i] - output_ptr[i]) < delta;
//       std::cout << "Comparing: (" << input_ptr[i] << "," << output_ptr[i]
//                 << "), result: " << equal << "\n";
//     }

//     return equal;
//   }
// };

// int main(int argc, char *argv[]) {
//   start_timer(main, global);
//   OptParser op(
//       "Harness for SPMV sparse matrix dense vector multiplication
//       benchmarks");

//   auto opt_platform = op.addOption<unsigned>(
//       {'p', "platform", "OpenCL platform index (default 0).", 0});
//   auto opt_device = op.addOption<unsigned>(
//       {'d', "device", "OpenCL device index (default 0).", 0});
//   auto opt_trials = op.addOption<unsigned>(
//       {'i', "trials", "Execute each kernel 'trials' times (default 10).",
//       10});

//   //   auto opt_input_file = op.addOption<std::string>({'f', "file", "Input
//   //   file"});

//   auto opt_matrix_file =
//       op.addOption<std::string>({'m', "matrix", "Input matrix"});
//   auto opt_kernel_file =
//       op.addOption<std::string>({'k', "kernel", "Input kernel"});
//   auto opt_run_file =
//       op.addOption<std::string>({'r', "runfile", "Run configuration file"});
//   auto opt_host_name = op.addOption<std::string>(
//       {'n', "hostname", "Host the harness is running on"});
//   auto opt_experiment_id = op.addOption<std::string>(
//       {'e', "experiment", "An experiment ID for data reporting"});
//   auto opt_float_delta = op.addOption<double>(
//       {'t', "delta", "Delta for floating point comparisons", 0.0001});

//   auto opt_timeout = op.addOption<float>(
//       {'t', "timeout", "Timeout to avoid multiple executions (default
//       100ms).",
//        100.0f});

//   op.parse(argc, argv);

//   using namespace std;

//   const std::string matrix_filename = opt_matrix_file->require();
//   const std::string kernel_filename = opt_kernel_file->require();
//   const std::string runs_filename = opt_run_file->require();
//   const std::string hostname = opt_host_name->require();
//   const std::string experiment = opt_experiment_id->require();

//   std::cerr << "matrix_filename " << matrix_filename << ENDL;
//   std::cerr << "kernel_filename " << kernel_filename << ENDL;

//   // initialise a matrix, kernel, and set of run parameters from files
//   SparseMatrix<float> matrix(matrix_filename);
//   KernelConfig<float> kernel(kernel_filename);
//   auto csvlines = CSV::load_csv(runs_filename);
//   std::vector<Run> runs;
//   std::transform(csvlines.begin(), csvlines.end(), std::back_inserter(runs),
//                  [](CSV::csv_line line) -> Run { return Run(line); });

//   for (auto run : runs)
//     std::cerr << run << ENDL;

//   // check the matrix
//   if (matrix.height() != matrix.width()) {
//     std::cout << "Matrix is not square. Failing computation." << ENDL;
//     std::cerr << "Matrix is not square. Failing computation." << ENDL;
//     std::exit(2);
//   }

//   // specialise the matrix for the kernel given
//   auto cl_matrix = kernel.specialiseMatrix(matrix, 0.0f);
//   // extract size variables from it
//   int v_Width_cl = cl_matrix.getCLVWidth();
//   int v_Height_cl = cl_matrix.getCLVHeight();
//   int v_Length_cl = cl_matrix.rows;

//   // size args of name/order:
//   // v_MWidthC_1, v_MHeight_2, v_VLength_3
//   std::vector<int> size_args{v_Width_cl, v_Height_cl, v_Length_cl};

//   // generate a vector

//   ConstXVectorGenerator<float> tengen(1000.0f);
//   ConstYVectorGenerator<float> zerogen(0);

//   // auto clkernel = executor::Kernel(kernel.getSource(), "KERNEL",
//   "").build();
//   // get some arguments
//   auto args = executorEncodeMatrix(kernel, matrix, 0.0f, tengen, zerogen,
//                                    v_Width_cl, v_Height_cl, v_Length_cl);
//   HarnessPR harness(kernel.getSource(), opt_platform->get(),
//   opt_device->get(),
//                     args);
//   for (auto run : runs) {
//     start_timer(run_iteration, main);
//     std::cout << "Benchmarking run: " << run << ENDL;
//     std::vector<std::chrono::nanoseconds> runtimes = harness.benchmark(
//         run, opt_trials->get(), opt_timeout->get(), opt_float_delta->get());
//     std::cout << "runtimes: [";
//     for (auto time : runtimes) {
//       std::cout << "," << time.count();
//     }
//     std::cout << "]" << ENDL;
//   }
// }
int main() { return 0; }