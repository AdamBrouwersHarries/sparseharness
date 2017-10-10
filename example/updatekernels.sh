#/bin/sh

rm *.json

d=/home/adam/projects/lift/scripts/generated_programs/spmv/
cp -f $d/swrg-slcl-pmdp.json kernel.json
cp -f $d/awrg-alcl-alcl-edp-split-512.json kernel2.json
cp -f $d/swrg-slcl-sdp-chunk-128.json kernel3.json
cp -f $d/awrg-alcl-alcl-edp-split-8.json kernel4.json
cp -f $d/glb-sdp.json kernel5.json
cp -f $d/glb-sdp-rsa.json kernel6.json

algorithms=( sssp spmv bfs pr)
for alg in "${algorithms[@]}"
do 
	echo "Copying algorithm: $alg"
	rm -rf $alg
	ad=/home/adam/projects/lift/scripts/generated_programs/$alg/
	mkdir -p $alg
	cp -f $ad/swrg-slcl-pmdp.json $alg/kernel.json
	cp -f $ad/awrg-alcl-alcl-edp-split-512.json $alg/kernel2.json
	cp -f $ad/swrg-slcl-sdp-chunk-128.json $alg/kernel3.json
	cp -f $ad/awrg-alcl-alcl-edp-split-8.json $alg/kernel4.json
	cp -f $ad/glb-sdp.json $alg/kernel5.json
	cp -f $ad/glb-sdp-rsa.json $alg/kernel6.json
done