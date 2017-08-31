#pragma once

#include <chrono>
#include <sstream>
#include <string>

// virtual void print_sql_stats(const Run &run, const std::string &kname,
//                              const std::string &mname, const std::string
//                              &hname,
//                              const std::string &experiment_id,
//                              std::vector<double> &times)

// {
//   // auto &devPtr = executor::globalDeviceList.front();
//   std::cout << "INSERT INTO table_name (time, correctness, kernel, "
//             << "global, local, host, device, matrix, iteration, trial,"
//             << "statistic, experiment_id) VALUES ";
//   int trial = 0;
//   for (auto t : times) {
//     if (trial != 0) {
//       std::cout << ",";
//     }
//     std::cout << "(" << t << ",\"notchecked\", \"" << kname << "\", "
//               << run.global1 << ", " << run.local1 << ", \"" << hname
//               << "\", \"" << getDeviceName() << "\", \"" << mname << "\", 0,"

//               << trial << ", \"RAW_RESULT\", \"" << experiment_id << "\")";
//     trial++;
//   }
//   std::cout << ";\n";
// }

enum Correctness {
  CORRECT,
  NOT_CHECKED,
  GENERIC_FAILURE,
  BAD_LENGTH,
  BAD_VALUES,
  GENERIC_BAD_VALUES
};

enum TrialType { RAW_RESULT, MULTI_ITERATION_SUM, MEDIAN_RESULT };

class SqlStat {

public:
  SqlStat(std::chrono::milliseconds time, Correctness correctness,
          unsigned int global, unsigned int local, TrialType trial_type)
      : _time(time), _correctness(correctness), _global(global), _local(local),
        _trial_type(trial_type) {}

  std::string printStat(const std::string &kernel_name,
                        const std::string &host_name,
                        const std::string &device_name,
                        const std::string &matrix_name,
                        const std::string &experiment_id) {
    std::ostringstream out;
    out << "(" << _time.count() << ", \"notchecked\", \"" << kernel_name
        << "\", " << _global << ", " << _local << ", \"" << host_name
        << "\", \"" << device_name << "\", \"" << matrix_name << "\", 0, \""

        << trialType() << "\", \"" << experiment_id << "\")";
    return out.str();
  }

  static std::string printHeader() {
    std::ostringstream out;
    out << "INSERT INTO table_name (time, correctness, kernel, "
        << "global, local, host, device, matrix, iteration, trial,"
        << "statistic, experiment_id) VALUES ";
    return out.str();
  }

  static bool compare(SqlStat a, SqlStat b) {
    return a.getTime() < b.getTime();
  }

  static std::chrono::milliseconds add(SqlStat a, SqlStat b) {
    return a.getTime() + b.getTime();
  }

  static std::string makeSqlCommand(std::vector<SqlStat> stats,
                                    const std::string &kernel_name,
                                    const std::string &host_name,
                                    const std::string &device_name,
                                    const std::string &matrix_name,
                                    const std::string &experiment_id) {
    std::ostringstream out;
    out << printHeader();
    int i = 0;
    for (auto stat : stats) {
      if (i != 0) {
        out << ", ";
      }
      out << stat.printStat(kernel_name, host_name, device_name, matrix_name,
                            experiment_id);
      i++;
    }
    out << ";";
    return out.str();
  }

  std::chrono::milliseconds getTime() { return _time; }

private:
  std::string trialType() {
    switch (_trial_type) {
    case RAW_RESULT:
      return "RAW_RESULT";
      break;
    case MULTI_ITERATION_SUM:
      return "MULTI_ITERATION_SUM";
      break;
    case MEDIAN_RESULT:
      return "MEDIAN_RESULT";
      break;
    default:
      return "ERROR";
    };
  }

  std::string trialCorrectness() {
    switch (_correctness) {
    case CORRECT:
      return "correct";
      break;
    case NOT_CHECKED:
      return "notchecked";
      break;
    case GENERIC_FAILURE:
      return "genericfailure";
      break;
    case BAD_LENGTH:
      return "badlength";
      break;
    case BAD_VALUES:
      return "badvalues";
      break;
    case GENERIC_BAD_VALUES:
      return "badvalues";
      break;
    default:
      return "ERROR";
    }
  }
  std::chrono::milliseconds _time;
  Correctness _correctness;
  unsigned int _global;
  unsigned int _local;
  TrialType _trial_type;
};