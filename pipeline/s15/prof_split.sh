#!/bin/bash
# S1.5 step 5 (Orin): per-layer profile of full engine vs the 4 segment engines (trtexec, no data transfers, spin wait).
# Usage: prof_split.sh <eng_dir> <tag>   -> ~/h26/s15/prof/<tag>_{full,seg1..4}.{prof,layers}.json + .log
TE=/usr/src/tensorrt/bin/trtexec; ED=$1; TAG=$2; OD=~/h26/s15/prof; mkdir -p $OD
for e in full seg1 seg2 seg3 seg4; do
  $TE --loadEngine=$ED/${e}_fp16.plan --noDataTransfers --useSpinWait --warmUp=2000 --duration=10 --iterations=2000 \
      --dumpProfile --separateProfileRun --profilingVerbosity=detailed \
      --exportProfile=$OD/${TAG}_${e}.prof.json --exportLayerInfo=$OD/${TAG}_${e}.layers.json > $OD/${TAG}_${e}.log 2>&1
  echo "$TAG $e rc=$? $(grep -m1 'GPU Compute Time' $OD/${TAG}_${e}.log)"
done
