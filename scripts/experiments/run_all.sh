#/bin/bash

datasetf=$1
echo "Dataset folder: $datasetf"

spmv=$2
echo "SparseMatrixDenseVector excutable: $spmv"

kernelfolder=$3
echo "KernelFolder: $kernelfolder"

runfile=$4
echo "runfile: $runfile"

platform=$5
echo "Platform: $platform"

device=$6
echo "Device: $device"

# Get some unique data for the experiment ID
now=$(date -Iminutes)
hsh=$(git rev-parse HEAD)
exID="$hsh-$now"

start=$(date +%s)

mkdir -p .gold_results
mkdir -p .gold
mkdir -p "results/results-$exID"

# build the kernels first
# firstMatrix=$(tail -n 1 $datasetf/datasets.txt)
# echo $firstMatrix
# $spmv --save-kernels --all -g 32768 -l 128 --platform $platform $datasetf/$firstMatrix/$firstMatrix.mtx

# Make gold output for each dataset
# for f in $(cat $datasetf/datasets.txt);
# do
#     echo "matrix: $f"
  
#     echo "$spmv --load-kernels -g 32768 -l 128 -i 1 --check --saveOutput .gold/spmv-$f.gold --platform $platform $datasetf/$f/$f.mtx >> .gold_results/$f-results.txt"
#     $spmv --load-kernels -g 32768 -l 128 -i 1 --check --saveOutput .gold/spmv-$f.gold  --platform $platform $datasetf/$f/$f.mtx >> .gold_results/$f-results.txt
# done

for m in $(cat $datasetf/datasets.txt);
do
	for k in $(ls $kernelfolder);
	do
		echo "Processing matrix: $m" 
		kname=$(basename $k .json)
		echo "Using kernel: $kname"
		$spmv -m $datasetf/$m/$m.mtx \
			  -i 5 \
			  -t 20 \
			  -k $kernelfolder/$k \
			  -p $platform \
			  -d $device \
			  -r $runfile &>results/results-$exID/result_$m_$kname.txt
			  # -p $platform \
			 # 2>results-$exID/result_$m_$kname.txt
		cptime=$(date +%s)
		runtime=$((cptime-start))

		echo "finished at $(date -Iminutes), took $runtime seconds"
	done
done


# for l in ${l_sizes[@]};
# do
# 	echo "Local: $l"
# 	for m in ${mult[@]}; 
# 	do
# 		echo "\t Mult: $m"
# 		echo "\t $(($l * $m))"
# 		global=$(($l * $m))
# 		echo "Global: $global"
# 		if [ $global -ge 16384 ] && [ $global -le 524288 ];
# 		then
# 			for f in $(cat $datasetf/datasets.txt);
# 			do
# 				echo "matrix: $f"
# 				echo "global: $global"
# 				echo "local: $local"
# 				echo "Resultfile: result_$f-$global-$l.txt"
# 				$spmv --experimentId $exID --load-kernels --loadOutput .gold/spmv-$f.gold 
# 				-g $global -l $l -i 20 -t 25 --all --check $datasetf/$f/$f.mtx &>                 
# 				 results-$exID/result_$f-$global-$l.txt
# 			done
# 		fi
# 	done
# done
