#!/bin/sh

projroot=$1
# This script runs an example kernel using the harness 
# The harness executable
harness=$projroot/build/spmv_harness
# A kernel to run with the harness
kernel=$projroot/example/kernel6.json
# The matrix to process
matrix=$projroot/example/matrix3.mtx
# The run parameters - i.e. local and global sizes
runfile=$projroot/example/runfile.csv
# our hostname
hname=$HOSTNAME
# a random experiemnt id
exid=example_experiment

# run it all!
oclgrind --data-races --uninitialized $harness -e $exid -n $hname -m $matrix -f matrix3 -k $kernel -d 0 -r $runfile -i 1 -t 200 


# $spmv -p $platform \
#               -d $device \
#               -i 5 \
#               -m $datasetf/$m/$m.mtx \
#               -f $m \
#               -k $kernelfolder/$k \
#               -r $runfile \
#               -n $host \
#               -t 20 \
#               -e $exID &>$rdir/result_$kname.txt
