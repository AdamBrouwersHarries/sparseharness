#pragma once

#include "kernel_config.h"
#include "sparse_matrix.h"

// Generators for various vector types

// superclass - completely abstract vector generator
template <typename T> class VectorGenerator {
public:
  // VectorGenerator(SparseMatrix<T> &sm, KernelConfig<T> &kc)
  //     : _sm(sm), _kc(kc) {}

  virtual T get(int ix) = 0;

  std::vector<T> generate(int length) {
    start_timer(generate, VectorGenerator);
    std::vector<T> v(length);
    int n = 0;
    std::generate(v.begin(), v.end(), [&] {
      T v = get(n++);
      // std::cout << "Generated: " << v << "\n";
      return v;
    });
    return v;
  };
};

//////// More specific generators:

// generator that generates a vector of the same size as the _actual_ height of
// the matrix, i.e. the "x" vector in a standard gemv computation
template <typename T> class XVectorGenerator : public VectorGenerator<T> {};

// generator that generates a vector the same size as the _padded_ height of the
// matrix, i.e. the "y" vector in a standard gemv computation (we need the data
// sizes to match up in OpenCL, hence why this is not quite "correct", but gives
// us something that works for the actual memory we use)
template <typename T> class YVectorGenerator : public VectorGenerator<T> {};

// a custom constant x vector generator, which (given a value) will generate a
// vector filled with that value
template <typename T> class ConstXVectorGenerator : public XVectorGenerator<T> {
  T value;

public:
  ConstXVectorGenerator(T constv) : value(constv) {}

  virtual T get(int ix) { return value; }
};

// a custom constant y vector generator, which (given a value) will generate a
// vector filled with that value
template <typename T> class ConstYVectorGenerator : public YVectorGenerator<T> {
  T value;

public:
  ConstYVectorGenerator(T constv) : value(constv) {}

  virtual T get(int ix) { return value; }
};