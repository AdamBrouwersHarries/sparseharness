#pragma once
#include "executor/Executor.h"
#include "executor/GlobalArg.h"
#include "executor/KernelArg.h"
#include "executor/LocalArg.h"
#include "executor/ValueArg.h"
#include "run.h"

class Harness {
public:
  Harness(cl::Kernel kernel, std::vector<executor::KernelArg *> args)
      : _kernel(kernel), _args(args) {}

  virtual std::vector<double> benchmark(Run run, int iterations,
                                        double timeout) = 0;
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
  virtual double executeKernel(Run run) = 0;
  cl::Kernel _kernel;
  std::vector<executor::KernelArg *> _args;
};