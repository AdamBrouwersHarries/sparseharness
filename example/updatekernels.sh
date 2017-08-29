#/bin/sh
d=/home/adam/projects/lift/scripts/generated_programs/spmv/
cp -f $d/swrg-slcl-pmdp.json kernel.json
cp -f $d/awrg-alcl-alcl-edp-split-512.json kernel2.json
cp -f $d/swrg-slcl-sdp-chunk-128.json kernel3.json
cp -f $d/awrg-alcl-alcl-edp-split-8.json kernel4.json
cp -f $d/glb-sdp.json kernel5.json