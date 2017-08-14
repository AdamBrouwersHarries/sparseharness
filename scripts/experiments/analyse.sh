#!/bin/sh
resultfile=$1
echo "Reading data from $resultfile, and writing to: "

profilefile=$(echo $resultfile | sed -e "s/result/profiling_data/g")
summaryfile=$(echo $resultfile | sed -e "s/result/profile_summary/g")
readablesummary=$(echo $resultfile | sed -e "s/result/profile_summary_readable/g")

echo "  ++ profile file:     $profilefile"
echo "  ++ summary file:     $summaryfile"
echo "  ++ readable summary: $readablesummary"

touch $profilefile

echo "  .. extracting profile data"
cat $resultfile | grep "PROFILING_DATUM" > $profilefile 

echo "  .. removing git data and spurious spaces"
echo "  -- (running sed on $profilefile)"

# I'm not sure why this line isn't useful...
# sed -i '' -e "s/\[====.*\] //g" $profilefile
# remove any lines that come with with debug info. (might miss some profiling :( )
sed -i '.bakup' -e "/.*DINFO.*/d" $profilefile
sed -i '.bakup' -e "/^PROFILING_DATUM/!d" $profilefile
sed -i '.bakup' -e "s/PROFILING_DATUM(//g" $profilefile
sed -i '.bakup' -e "s/)//g" $profilefile
sed -i '.bakup' -e "s/ //g" $profilefile

echo "  .. running query over summary"

q -O -d , "select c1 as method, c2 as context, c4 as language, count(c3) as calls, min(c3) as minimum, avg(c3) as mean, max(c3) as maximum, sum(c3) as total from $profilefile group by c1, c2, c4 order by total desc" > $summaryfile

echo "  .. formatting!"

# cat $summaryfile | column -t  -s ,

cat $summaryfile | column -t  -s , > $readablesummary

