#!/bin/sh

projroot=$1
harness=$2
# This script runs an example kernel using the harness 
# The harness executable
# harness=$projroot/build/bfs_harness
# A kernel to run with the harness
kernel=$projroot/example/kernel.json
# The matrix to process
matrix=$projroot/example/matrix3.mtx
# The run parameters - i.e. local and global sizes
runfile=$projroot/example/shortrunfile.csv
# our hostname
hname=$HOSTNAME
# a random experiemnt id
exid=example_experiment

# run it all!
mkdir -p results
for f in $(ls $projroot/example/ | grep json); do
	kernel=$projroot/example/$f
	oclgrind $harness -e $exid -n $hname -m $matrix -k $kernel -d 0 -r $runfile -i 1 -t 0.5 &> results/output-$f.cpp
done 
