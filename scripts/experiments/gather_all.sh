#!/bin/sh
resultfolder=$1

find $resultfolder -name "*.txt" > files.txt
# echo "" > aggregate_result.txt
echo "" > aggregate_profiling_data.txt
echo "" > aggregate_sql_data.txt

count=$(cat files.txt | wc -l)
i=0
for f in $(cat files.txt); do
	echo "reading data from $f - $i/$count"
	i=$(($i + 1))
	grep "PROFILING_DATUM" $f >> aggregate_profiling_data.txt
	grep "INSERT INTO" $f >> aggregate_sql_data.txt
	# cat $f >> aggregate_result.txt
done