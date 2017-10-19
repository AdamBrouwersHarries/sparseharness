#!/bin/bash
$dataset=$1

cd ~/projects/sparseharness/
mkdir -p build
pushd build 
cmake .. 
make -j64
popd
cd scripts/experiments
# ./run_all.sh ~/mdatasets/original/ ../../build/spmv_harness ~/kdatasets/kernels/e9d0b5b53766f985811eb7a0e6b92b33c757b864/spmv/ ../../example/runfile2.csv 0 0 

./run_all.sh $dataset ../../build/spmv_harness ~/scratch/kdatasets/kernels/head/spmv/ ../../example/runfile2.csv 1 0 ~/scratch/s1467120
