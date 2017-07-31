#pragma once

#include "executor/Executor.h"
#include "kernel_config.h"
#include "sparse_matrix.h"

template <typename T>
std::vector<KernelArg> specialseMatrix(KernelConfig<T> kernel,
                                       SparseMatrix<T> matrix, T zero) {}