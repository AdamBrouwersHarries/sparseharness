#!/bin/sh
resultfolder=$1

files=$(find $resultfolder -name "*.txt")
touch aggregate_result.txt

count=$(echo $files | wc -l)
i=0
for f in $files; do
	echo "reading data from $f - $i/$count"
	i=$((i++))
	cat $f >> aggregate_result.txt
done