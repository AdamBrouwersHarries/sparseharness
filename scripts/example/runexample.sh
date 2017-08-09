#!/bin/sh

# This script runs an example kernel using the harness 
# The harness executable
harness=../../build/spmv_harness
# A kernel to run with the harness
kernel=../../example/kernel.json
# The matrix to process
matrix=../../example/matrix2.mtx
# The run parameters - i.e. local and global sizes
runfile=../../example/runfile2.csv
# our hostname
hname=$HOSTNAME

# run it all!
$harness -n $hname -m $matrix -k $kernel -r $runfile -i 10 -t 200 

