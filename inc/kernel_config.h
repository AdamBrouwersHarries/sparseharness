#ifndef KERNEL_H
#define KERNEL_H

#include "common.h"
#include "csds_timer.h"

class ArgDescr {
public:
  const std::string variable;
  const std::string addressSpace;
  const std::string size;
  ArgDescr(){};
  ArgDescr(std::string var, std::string addrspace, std::string sz)
      : variable(var), addressSpace(addrspace), size(sz){};
  // need a copy constructor?
};

class KernelProperties {
public:
  // Constructor
  KernelProperties();
  KernelProperties(std::string kname);
  KernelProperties(std::string outerMap, std::string innerMap,
                   std::string innerMap2, std::string arrayType, int splitSize,
                   int chunkSize);
  std::string outerMap;
  std::string innerMap;
  std::string innerMap2;
  std::string arrayType;
  int splitSize;
  int chunkSize;

private:
  std::string argcache;
};

template <typename T> class KernelConfig {
public:
  // Constructors
  KernelConfig(std::string filename);

  // Destuctor
  ~KernelConfig(){};

  // Getters
  std::string &getSource();
  std::string &getName();
  std::vector<ArgDescr> getArgs();
  std::vector<ArgDescr> getTempGlobals();
  std::vector<ArgDescr> getTempLocals();
  std::vector<std::string> getParamVars();
  ArgDescr *getOutputArg();
  KernelProperties getProperties();

private:
  std::string source;
  std::string name;
  std::vector<ArgDescr> inputArgs;
  std::vector<ArgDescr> tempGlobals;
  std::vector<ArgDescr> tempLocals;
  std::vector<std::string> paramVars;
  ArgDescr *outputArg;
  KernelProperties kprops;
};

#endif // KERNEL_H