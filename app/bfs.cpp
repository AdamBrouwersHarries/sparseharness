// [standard includes]
#include <algorithm>
#include <atomic>
#include <bitset>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <typeinfo>
#include <vector>

// [external includes]
#include "Executor.h"

// [tools]
#include "common.h"
#include "csds_timer.h"
#include "csv_utils.h"
#include "options.h"

// [ocl tools]
#include "harness.h"
#include "kernel_config.h"
#include "kernel_utils.h"

// [application specific]
#include "run.h"
#include "sparse_matrix.h"
#include "vector_generator.h"

class HarnessBFS : public IterativeHarness<double> {
public:
  HarnessBFS(cl::Kernel kernel, ArgConfig args)
      : IterativeHarness(kernel, args) {}
  std::vector<double> benchmark(Run run, int iterations, double timeout,
                                double delta) {
    start_timer(benchmark, HarnessBFS);

    // kernel setup
    int i = 0;
    for (auto &arg : _args.args) {
      arg->upload();
      arg->setAsKernelArg(_kernel, i);
      ++i;
    }

    // iterations
    std::vector<double> runtimes(iterations);

    // we need a cache for the input vector
    // this seems _super_ hacky. I don't like casting around like this, even if
    // it is legal. I'm worried.
    std::vector<char> input_cache;
    copy_from_arg(_args.args[_args.input], input_cache);

    // run the benchmark for that many iterations
    for (int i = 0; i < iterations; i++) {
      start_timer(benchmark_iteration, HarnessBFS);
      // std::cout << "Iteration: " << i << '\n';

      // Run the algorithm
      double runtime = 0.0f;
      { // copy the cached input into the input arg:
        copy_into_arg(input_cache, _args.args[_args.input]);
        // _args.args[_args.input]->clear();
        // _args.args[_args.input]->upload();

        bool should_terminate = false;
        // run the kernel
        do {
          std::cout << " ------------------- VALUES BEFORE RUN\n";
          // print_arg<float>(_args.args[_args.input]);
          // print_arg<float>(_args.args[_args.output]);
          std::cout << "--------------- EXECUTING KERNEL\n";
          runtime += executeKernel(run);
          std::cout << " ------------------- VALUES after RUN\n";
          // print_arg<float>(_args.args[_args.input]);
          // print_arg<float>(_args.args[_args.output]);
          std::cout << "--------------- PERFORMING CHECK\n";
          should_terminate = should_terminate_iteration(
              _args.args[_args.input], _args.args[_args.output], delta);
          // swap the pointers in the arg list
          std::cout << "---------------- SWAPPING \n";

          executor::KernelArg *tmp = _args.args[_args.input];
          _args.args[_args.input] = _args.args[_args.output];
          _args.args[_args.output] = tmp;
          // std::cout << "preswap: in: " << _args.input
          //           << " out: " << _args.output << "\n";
          // auto tmp = _args.input;
          // _args.input = _args.output;
          // _args.output = tmp;
          // std::cout << "postswap: in: " << _args.input
          //           << " out: " << _args.output << "\n";

          // copy the output buffer into the input
          // copy_args(_args.args[_args.output], _args.args[_args.input]);

          // reset the kernel args
          // _args.args[_args.output]->clear();

          // _args.args[_args.input]->upload();
          // _args.args[_args.output]->upload();

          _args.args[_args.input]->setAsKernelArg(_kernel, _args.input);
          _args.args[_args.output]->setAsKernelArg(_kernel, _args.output);
        } while (!should_terminate);
        // get the underlying vectors from the args that we care about
      }

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
    start_timer(executeKernel, HarnessBFS);
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

    return getRuntimeInMilliseconds(event);
  }

  virtual bool should_terminate_iteration(executor::KernelArg *input,
                                          executor::KernelArg *output,
                                          double delta) {
    start_timer(should_terminate_iteration, HarnessBFS);
    {
      start_timer(arg_download, should_terminate_iteration);
      // input->download();
      output->download();
    }
    // get the host vectors from the arguments
    std::vector<char> &input_vector =
        static_cast<executor::GlobalArg *>(input)->data().hostBuffer();
    std::vector<char> &output_vector =
        static_cast<executor::GlobalArg *>(output)->data().hostBuffer();
    // reinterpret it as double pointers, and get the lengths
    auto input_ptr = reinterpret_cast<float *>(input_vector.data());
    auto output_ptr = reinterpret_cast<float *>(output_vector.data());
    auto input_length = input_vector.size() / sizeof(float);
    auto output_length = output_vector.size() / sizeof(float);
    // perform a comparison across the two of them, based on pointers
    bool equal = true;
    for (unsigned int i = 0;
         equal == true && i < input_length && i < output_length; i++) {
      equal = fabs(input_ptr[i] - output_ptr[i]) < delta;
    }

    return equal;
  }
};

int main(int argc, char *argv[]) {
  start_timer(main, global);
  OptParser op(
      "Harness for SPMV sparse matrix dense vector multiplication benchmarks");

  auto opt_platform = op.addOption<unsigned>(
      {'p', "platform", "OpenCL platform index (default 0).", 0});
  auto opt_device = op.addOption<unsigned>(
      {'d', "device", "OpenCL device index (default 0).", 0});
  auto opt_iterations = op.addOption<unsigned>(
      {'i', "iterations",
       "Execute each kernel 'iterations' times (default 10).", 10});

  //   auto opt_input_file = op.addOption<std::string>({'f', "file", "Input
  //   file"});

  auto opt_matrix_file =
      op.addOption<std::string>({'m', "matrix", "Input matrix"});
  auto opt_kernel_file =
      op.addOption<std::string>({'k', "kernel", "Input kernel"});
  auto opt_run_file =
      op.addOption<std::string>({'r', "runfile", "Run configuration file"});
  auto opt_host_name = op.addOption<std::string>(
      {'n', "hostname", "Host the harness is running on"});
  auto opt_experiment_id = op.addOption<std::string>(
      {'e', "experiment", "An experiment ID for data reporting"});
  auto opt_float_delta = op.addOption<double>(
      {'t', "delta", "Delta for floating point comparisons", 0.0001});

  auto opt_timeout = op.addOption<float>(
      {'t', "timeout", "Timeout to avoid multiple executions (default 100ms).",
       100.0f});

  op.parse(argc, argv);

  using namespace std;

  const std::string matrix_filename = opt_matrix_file->require();
  const std::string kernel_filename = opt_kernel_file->require();
  const std::string runs_filename = opt_run_file->require();
  const std::string hostname = opt_host_name->require();
  const std::string experiment = opt_experiment_id->require();

  std::cerr << "matrix_filename " << matrix_filename << ENDL;
  std::cerr << "kernel_filename " << kernel_filename << ENDL;

  // initialise a matrix, kernel, and set of run parameters from files
  SparseMatrix<float> matrix(matrix_filename);
  KernelConfig<float> kernel(kernel_filename);
  auto csvlines = CSV::load_csv(runs_filename);
  std::vector<Run> runs;
  std::transform(csvlines.begin(), csvlines.end(), std::back_inserter(runs),
                 [](CSV::csv_line line) -> Run { return Run(line); });

  for (auto run : runs)
    std::cerr << run << ENDL;

  // check the matrix
  if (matrix.height() != matrix.width()) {
    std::cout << "Matrix is not square. Failing computation." << ENDL;
    std::cerr << "Matrix is not square. Failing computation." << ENDL;
    std::exit(2);
  }

  // specialise the matrix for the kernel given
  auto cl_matrix = kernel.specialiseMatrix(matrix, 0.0f);
  // extract size variables from it
  int v_Width_cl = cl_matrix.getCLVWidth();
  int v_Height_cl = cl_matrix.getCLVHeight();
  int v_Length_cl = cl_matrix.rows;

  // size args of name/order:
  // v_MWidthC_1, v_MHeight_2, v_VLength_3
  std::vector<int> size_args{v_Width_cl, v_Height_cl, v_Length_cl};

  // initialise the executor
  initExecutor(opt_platform->get(), opt_device->get());

  // generate a vector

  ConstXVectorGenerator<float> tengen(1000.0f);
  ConstYVectorGenerator<float> zerogen(0);

  auto clkernel = executor::Kernel(kernel.getSource(), "KERNEL", "").build();
  // get some arguments
  auto args = executorEncodeMatrix(kernel, matrix, 0.0f, tengen, zerogen,
                                   v_Width_cl, v_Height_cl, v_Length_cl);
  HarnessBFS harness(clkernel, args);
  for (auto run : runs) {
    start_timer(run_iteration, main);
    std::cout << "Benchmarking run: " << run << ENDL;
    std::vector<double> runtimes = harness.benchmark(
        run, opt_iterations->get(), opt_timeout->get(), opt_float_delta->get());
    std::cout << "runtimes: [";
    for (auto time : runtimes) {
      std::cout << "," << time;
    }
    std::cout << "]" << ENDL;
  }

  shutdownExecutor();
}
