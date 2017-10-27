#pragma once

#define ENDL "\n"

#define COMMON_MAIN_PREAMBLE(mtype)                                            \
  start_timer(main, global);                                                   \
  OptParser op("Harness for SPMV sparse matrix dense vector multiplication "   \
               "benchmarks");                                                  \
  auto opt_platform = op.addOption<unsigned>(                                  \
      {'p', "platform", "OpenCL platform index (default 0).", 0});             \
  auto opt_device = op.addOption<unsigned>(                                    \
      {'d', "device", "OpenCL device index (default 0).", 0});                 \
  auto opt_trials = op.addOption<unsigned>(                                    \
      {'i', "trials", "Execute each kernel 'trials' times (default 10).",      \
       10});                                                                   \
  auto opt_matrix_file =                                                       \
      op.addOption<std::string>({'m', "matrix", "Input matrix"});              \
  auto opt_matrix_name =                                                       \
      op.addOption<std::string>({'f', "matrix_name", "Input matrix name"});    \
  auto opt_kernel_file =                                                       \
      op.addOption<std::string>({'k', "kernel", "Input kernel"});              \
  auto opt_run_file =                                                          \
      op.addOption<std::string>({'r', "runfile", "Run configuration file"});   \
  auto opt_host_name = op.addOption<std::string>(                              \
      {'n', "hostname", "Host the harness is running on"});                    \
  auto opt_experiment_id = op.addOption<std::string>(                          \
      {'e', "experiment", "An experiment ID for data reporting"});             \
  auto opt_float_delta = op.addOption<double>(                                 \
      {'c', "delta", "Delta for floating point comparisons", 0.0001});         \
  auto opt_timeout = op.addOption<unsigned int>(                               \
      {'t', "timeout",                                                         \
       "Timeout to avoid multiple executions (default 100ms).", 100});         \
  op.parse(argc, argv);                                                        \
  using namespace std;                                                         \
  const std::string matrix_filename = opt_matrix_file->require();              \
  const std::string matrix_name = opt_matrix_name->require();                  \
  const std::string kernel_filename = opt_kernel_file->require();              \
  const std::string runs_filename = opt_run_file->require();                   \
  const std::string hostname = opt_host_name->require();                       \
  const std::string experiment = opt_experiment_id->require();                 \
  std::cerr << "matrix_filename " << matrix_filename << ENDL;                  \
  std::cerr << "kernel_filename " << kernel_filename << ENDL;                  \
  SparseMatrix<mtype> matrix(matrix_filename);                                 \
  KernelConfig<mtype> kernel(kernel_filename);                                 \
  auto csvlines = CSV::load_csv(runs_filename);                                \
  std::vector<Run> runs;                                                       \
  std::transform(csvlines.begin(), csvlines.end(), std::back_inserter(runs),   \
                 [](CSV::csv_line line) -> Run { return Run(line); });         \
  if (matrix.height() != matrix.width()) {                                     \
    std::cout << "Matrix is not square. Failing computation." << ENDL;         \
    std::cerr << "Matrix is not square. Failing computation." << ENDL;         \
    std::exit(2);                                                              \
  } else {                                                                     \
    std::cout << " Matrix is square - width = " << matrix.width()              \
              << " and height = " << matrix.height() << "\n";                  \
  }
