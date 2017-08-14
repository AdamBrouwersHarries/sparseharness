#!/bin/sh
resultfolder=$1

find $resultfolder -name "*.txt" > files.txt
echo "" > aggregate_result.txt

count=$(cat files.txt | wc -l)
i=0
for f in $(cat files.txt); do
	echo "reading data from $f - $i/$count"
	i=$(($i + 1))
	cat $f >> aggregate_result.txt
done