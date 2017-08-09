#!/bin/sh

projroot=$1
# This script runs an example kernel using the harness 
# The harness executable
harness=$projroot/build/spmv_harness
# A kernel to run with the harness
kernel=$projroot/example/kernel.json
# The matrix to process
matrix=$projroot/example/matrix2.mtx
# The run parameters - i.e. local and global sizes
runfile=$projroot/example/runfile2.csv
# our hostname
hname=$HOSTNAME

# run it all!
$harness -n $hname -m $matrix -k $kernel -d 1 -r $runfile -i 10 -t 200 

