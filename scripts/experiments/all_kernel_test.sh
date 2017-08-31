#!/bin/sh
# This script runs the a given harness with all generated kernels using 
# oclgrind, and searches for reported errors


projroot=$1
echo "Project root: $projroot" 
harness=$2 
echo "Harness executable: $harness"
kernelfolder=$3
echo "KernelFolder: $kernelfolder"
# The matrix to process
matrix=$projroot/example/matrix3.mtx
# The run parameters - i.e. local and global sizes
runfile=$projroot/example/runfile2.csv
# our hostname
hname=$HOSTNAME
# a random experiemnt id
exid=kernel_oclgrind_test
rdir=~/scratch/results/$exid
mkdir -p $rdir
mkdir -p $rdir/oclgrind/

kernelcount=$(ls $kernelfolder | wc -l)
i=1
failed_kernels=0
errored_kernels=0
oclgrind_failed_kernels=0
for k in $(ls $kernelfolder);
do
	echo "Running kernel $k - $i/$kernelcount"
	echo "" > $rdir/oclgrind/log-$k.log
	oclgrind --data-races --log $rdir/oclgrind/log-$k.log --check-api \
		$harness -e $exid \
		-n $hname \
		-m $matrix \
		-f "matrix3" \
		-k $kernelfolder/$k \
		-d 0 \
		-r $runfile \
		-i 5 \
		-t 0.5 &> $rdir/output-$k.cpp

	# Report warp divergence
	if grep -Fq "Work-group divergence detected" $rdir/oclgrind/log-$k.log 
		then
		echo "  Work-group divergence in kernel $k"
	fi 

	if grep -Fq "kernel has probably failed" $rdir/output-$k.cpp 
		then 
		echo "  Kernel has probably failed $f"
		failed_kernels=$(($failed_kernels + 1))
	fi

	if grep -Fq "code: -9999" $rdir/output-$k.cpp 
		then
		echo "  OpenCL call failed with error unknown error (code: -9999)"
		errored_kernels=$(($errored_kernels + 1))
	fi

	if [[ $(wc -l <$rdir/oclgrind/log-$k.log) -ge 2 ]] 
		then 
		echo "  OCLGrind found errors with this kernel!"
		oclgrind_failed_kernels=$(($oclgrind_failed_kernels + 1))
	else
		rm -rf $rdir/oclgrind/log-$k.log
	fi

	# # Report other errors
	# errorLines=$(cat $rdir/oclgrind-log-$k.log | wc -l )
	# if [[ errorLines -ge 0 ]] then 
	# else 

	# fi

	# echo "	$errorLines lines of errors"

	echo "	Current tally: $failed_kernels/$i have probably failed"
	echo "	Current tally: $errored_kernels/$i have given -9999 errors"
	echo "	Current tally: $oclgrind_failed_kernels/$i have had OCLGrind errors"

	i=$(($i + 1))
done

echo "$failed_kernels/$i have probably failed"
echo "$errored_kernels/$i have given -9999 errors"
echo "$oclgrind_failed_kernels/$i have had OCLGrind errors"