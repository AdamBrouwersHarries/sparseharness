#ifndef KERNEL_H
#define KERNEL_H

#include "sparse_matrix.h"

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
                   std::string innerMap2, int splitSize, int chunkSize);
  std::string outerMap;
  std::string innerMap;
  std::string innerMap2;
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
  std::string getSource();
  std::string getName();
  std::vector<ArgDescr> getArgs();
  KernelProperties getProperties();

  // Specialiser for a matrix - makes more sense here than in the matrix,
  // as it's kernel, not matrix dependent

  OpenCLSparseMatrix<T> specialiseMatrix(SparseMatrix<T> matrix, T zero);

private:
  std::string source;
  std::string name;
  std::vector<ArgDescr> inputArgs;
  std::vector<ArgDescr> tempArgs;
  ArgDescr *outputArg;
  KernelProperties kprops;
};

#endif // KERNEL_H