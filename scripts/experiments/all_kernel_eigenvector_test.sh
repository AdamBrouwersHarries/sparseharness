#!/bin/sh
# This script runs the "eigenvector" harness with all generated SPMV kernels using oclgrind, and searches for reported errors


projroot=$1
echo "Project root: $projroot" 
harness=$2 
echo "Harness executable: $harness"
kernelfolder=$3
echo "KernelFolder: $kernelfolder"
# The matrix to process
matrix=$projroot/example/matrix3.mtx
# The run parameters - i.e. local and global sizes
runfile=$projroot/example/shortrunfile.csv
# our hostname
hname=$HOSTNAME
# a random experiemnt id
exid=kernel_oclgrind_test
rdir=~/scratch/results/$exid
mkdir -p $rdir

kernelcount=$(ls $kernelfolder | wc -l)
i=1
for k in $(ls $kernelfolder);
do
	echo "Running kernel $k - $i/$kernelcount"
	oclgrind --log $rdir/oclgrind-log-$k.log --check-api $harness -e $exid -n $hname -m $matrix -k $kernelfolder/$k -d 0 -r $runfile -i 1 -t 0.5 &> $rdir/output-$k.cpp

	# Report warp divergence
	if grep -Fq "Work-group divergence detected" $rdir/oclgrind-log-$k.log
	then
		echo "	Work-group divergence in kernel $k"
	fi

	# Report other errors
	errorLines=$(cat $rdir/oclgrind-log-$k.log | wc -l )
	echo "	$errorLines lines of errors"

	i=$(($i + 1))
done