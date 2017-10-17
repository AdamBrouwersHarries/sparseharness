#!/bin/sh

kernel=$1
matrix=$2
root=$3

# make a folder for temp data
dataf=~/scratch/parser_prof/
mkdir -p $dataf

# Run the script, and cat it all to a file called "parser_result.txt"
$root/build/just_parser_harness -i 10 -f mname -k $kernel -m $matrix &> $dataf/parser_result.txt

pushd $dataf

~/tools/analyse.sh $dataf/parser_result.txt

popd 