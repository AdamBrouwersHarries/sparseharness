#!/bin/bash

# Build a query from a folder containing results
# get the directory containing (recursively) the results
resdir=$1
# get the name of a table to insert into
table=$2


# make a temporary folder for the results
bdir=$(basename $resdir)
tdir="/tmp/sparseharness/$bdir"
echo "Making temp directory: $tdir"
mkdir -p $tdir

# build the file we will write to
qf=$tdir/query.sql
touch $qf
echo "" > $qf

for f in $(find $resdir -name "*.tar.gz");
do
	# unzip it to stdout, into grp, then sed, then to the result file
	tar -xOzf $f | grep "INSERT" | sed "s/.*INSERT/INSERT/g" | sed "s/table_name/$table/g" >> $qf
done
