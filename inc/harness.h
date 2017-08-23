#pragma once
#include "executor/Executor.h"
#include "executor/GlobalArg.h"
#include "executor/KernelArg.h"
#include "executor/LocalArg.h"
#include "executor/ValueArg.h"
#include "kernel_utils.h"
#include "run.h"

template <typename T> class Harness {
public:
  Harness(cl::Kernel kernel, ArgConfig args) : _kernel(kernel), _args(args) {}

  virtual std::vector<T> benchmark(Run run, int iterations, double timeout,
                                   double delta) = 0;
  virtual void
  print_sql_stats(const Run &run, const std::string &kname,
                  const std::string &mname, const std::string &hname,
                  const std::string &experiment_id, std::vector<T> &times)

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
  // copy data from a global arg's host memory into another container
  void copy_from_arg(executor::KernelArg *arg,
                     std::vector<char> &newcontainer) {
    // get the arg as a global arg
    executor::GlobalArg *global_arg = static_cast<executor::GlobalArg *>(arg);
    std::copy(global_arg->data().begin(), global_arg->data().end(),
              std::back_insert_iterator<std::vector<char>>(newcontainer));
  }

  // copy data from a container into a global arg's host memory
  void copy_into_arg(std::vector<char> &data, executor::KernelArg *arg) {
    // get the arg as a global arg - let's hope this is valid!
    executor::GlobalArg *global_arg = static_cast<executor::GlobalArg *>(arg);
    global_arg->assign(data);
  }

  template <typename U> void print_arg(executor::KernelArg *arg) {
    executor::GlobalArg *global_arg = static_cast<executor::GlobalArg *>(arg);
    global_arg->download();
    auto vectdata = global_arg->data().hostBuffer();
    auto data = reinterpret_cast<U *>(vectdata.data());
    auto len = vectdata.size() / sizeof(U);
    std::cout << "[";
    for (int i = 0; i < len; i++) {
      std::cout << data[i] << ",";
    }
    std::cout << "]\n";
  }

  virtual double executeKernel(Run run) = 0;
  cl::Kernel _kernel;
  ArgConfig _args;
};

// template <typename T> class IterativeHarness : public Harness<std::vector<T>>
// {
template <typename T> class IterativeHarness : public Harness<T> {
public:
  IterativeHarness(cl::Kernel kernel, ArgConfig args)
      : Harness<T>(kernel, args) {}

protected:
  virtual bool should_terminate_iteration(executor::KernelArg *input,
                                          executor::KernelArg *output,
                                          double delta) = 0;
};