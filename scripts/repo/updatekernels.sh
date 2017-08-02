#/bin/sh
d=/home/adam/projects/lift/scripts/generated_programs/spmv/
r=/home/adam/projects/spmvharness/
cp -f $d/swrg-slcl-pmdp.json $r/example/kernel.json
cp -f $d/awrg-alcl-alcl-edp-split-512.json $r/example/kernel2.json
cp -f $d/swrg-slcl-sdp-chunk-128.json $r/example/kernel3.json
cp -f $d/awrg-alcl-alcl-edp-split-8.json $r/example/kernel4.json