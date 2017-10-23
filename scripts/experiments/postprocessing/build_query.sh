#!/bin/bash

# Build a query from a folder containing results
# get the directory containing (recursively) the results
resdir=$1
# get the name of a table to insert into
table=$2


# make a temporary folder for the results
brdir=$(basename $resdir)
tdir="/tmp/scratchharness/$bdir"
mkdir -p $tdir

# grep the results from the folder recursively, 
# and subsitute various parts of it

grep -r "INSERT" . | sed "s/.*INSERT/INSERT/g" | sed "s/table_name/$table/g" > $tdir/query.sql