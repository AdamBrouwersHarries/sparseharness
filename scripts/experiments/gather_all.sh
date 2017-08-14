#!/bin/sh
resultfolder=$1

files=$(find $resultfolder -name "*.txt")
touch aggregate_result.txt

for f in $files; do
	echo "reading data from $f"
	cat $f >> aggregate_result.txt
done