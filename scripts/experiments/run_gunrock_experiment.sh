#!/bin/bash
cd ~/projects/sparseharness/
mkdir -p build
pushd build 
cmake .. 
make -j64
popd
cd scripts/experiments
# ./run_all.sh ~/mdatasets/original/ ../../build/spmv_harness ~/kdatasets/kernels/e9d0b5b53766f985811eb7a0e6b92b33c757b864/spmv/ ../../example/runfile2.csv 0 0 

./run_all.sh ~/scratch/mdatasets/gunrock/ ../../build/spmv_harness ~/scratch/kdatasets/kernels/a88d68e00740611dcb61dcab0f66906e8f0bbd1a/spmv/ ../../example/runfile2.csv 1 0 ~/scratch/s1467120
