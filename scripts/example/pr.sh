#!/bin/sh

projroot=$1
# This script runs an example kernel using the harness 
# The harness executable
harness=$projroot/build/pr_harness
# A kernel to run with the harness
kernel=$projroot/example/pr/kernel5.json
# The matrix to process
matrix=$projroot/example/matrix.mtx
# The run parameters - i.e. local and global sizes
runfile=$projroot/example/shortrunfile.csv
# our hostname
hname=$HOSTNAME
# a random experiemnt id
exid=example_experiment

# run it all!
oclgrind --check-api --data-races $harness \
    -e $exid \
    -n $hname \
    -m $matrix \
    -f matrix \
    -k $kernel \
    -d 0 \
    -r $runfile \
    -i 10 \
    -t 200 
    -c 0.01
