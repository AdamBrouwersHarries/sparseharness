#/bin/bash
echo "" > runstatus.txt

datasetf=$1
echo "Dataset folder: $datasetf"
echo "Dataset folder: $datasetf" >> runstatus.txt

spmv=$2
echo "SparseMatrixDenseVector excutable: $spmv"
echo "SparseMatrixDenseVector excutable: $spmv" >> runstatus.txt

kernelfolder=$3
echo "KernelFolder: $kernelfolder"
echo "KernelFolder: $kernelfolder" >> runstatus.txt

runfile=$4
echo "runfile: $runfile"
echo "runfile: $runfile" >> runstatus.txt

platform=$5
echo "Platform: $platform"
echo "Platform: $platform" >> runstatus.txt

device=$6
echo "Device: $device"
echo "Device: $device" >> runstatus.txt

host=$(hostname)

# Get some unique data for the experiment ID
now=$(date +"%Y-%m-%dT%H-%M-%S%z")
hsh=$(git rev-parse HEAD)
exID="$hsh-$now"

start=$(date +%s)

mkdir -p .gold_results
mkdir -p .gold
mkdir -p "results/results-$exID"

kernelcount=$(ls $kernelfolder | wc -l)
matrixcount=$(ls $datasetf | wc -l)
taskcount=$((kernelcount*matrixcount)) 
echo "taskcount: $taskcount"
echo "taskcount: $taskcount" >> runstatus.txt

i=0
for m in $(cat $datasetf/datasets.txt);
do
	rdir="results/results-$exID/$m/"
	mkdir -p $rdir

	for k in $(ls $kernelfolder);
	do
		echo "Processing matrix: $m - $i/$taskcount" 
		echo "Processing matrix: $m - $i/$taskcount"  >> runstatus.txt
		kname=$(basename $k .json)
		echo "Using kernel: $kname"
		echo "Using kernel: $kname" >> runstatus.txt

		runstart=$(date +%s)
		$spmv -p $platform \
			  -d $device \
			  -i 5 \
			  -m $datasetf/$m/$m.mtx \
			  -f $m \
			  -k $kernelfolder/$k \
			  -r $runfile \
			  -n $host \
			  -t 20 \
			  -e $exID &>$rdir/result_$kname.txt
			  # -p $platform \
			 # 2>results-$exID/result_$m_$kname.txt
		# Compress the result
		tar czvf $rdir/result_$kname.tar.gz $rdir/result_$kname.txt
		# remove the original file
		rm -rf $rdir/result_$kname.txt
		runend=$(date +%s)
		runtime=$((runend-runstart))

		scripttime=$((runend-start))

		echo "Run took $runtime seconds, total time of $scripttime seconds"
		echo "Run took $runtime seconds, total time of $scripttime seconds" >> runstatus.txt
		i=$(($i + 1))
	done
done

echo "finished experiments"
echo "finished experiments" >> runstatus.txt