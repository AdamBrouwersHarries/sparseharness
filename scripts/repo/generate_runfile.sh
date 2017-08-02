#/bin/sh

# Declare a range of local sizes to try
declare -a l_sizes=(128 64 256 32 512 16 8)
# declare some multipliers to get global sizes
declare -a mult=(8192 4096 2048 1024 512 256 128 64 32)

for l in ${l_sizes[@]};
do
	for m in ${mult[@]}; 
	do
		g=$(($l * $m))
		if [ $g -ge 16384 ] && [ $g -le 524288 ];
		then
			echo "$g,1,1,$l,1,1,"
		fi
	done
done
