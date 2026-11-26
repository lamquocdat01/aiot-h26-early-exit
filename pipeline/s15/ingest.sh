#!/bin/bash
: "${JETSON_HOST:?set JETSON_HOST=user@host of the Jetson board}"
# S1.5 (PC): for each Kaggle run dir <net>_s<seed> under $1: export ONNX (+ORT check) -> copy to Jetson -> enqueue job.
# Mode: 9201 meas, 9202 check2, else logits. (Nets below threshold are still processed; prereg excludes them only from H3.)
set -u; K=$1; OUT=D:/data/aiot-h26/s15/onnx; LOG=D:/data/aiot-h26/s15/ingest.log; mkdir -p $OUT
for d in $K/n?_s92??; do
  b=$(basename $d); n=${b%_s*}; s=${b#*_s}; [ -f $d/weights.pt ] || { echo "$(date +%T) $b no weights" >> $LOG; continue; }
  [ -f $OUT/$b/export_report.json ] && { echo "$(date +%T) $b already" >> $LOG; continue; }
  python D:/repos/aiot-h26/s15/export_trained_s15.py $n $d $OUT/$b > $OUT/$b.export.log 2>&1 || { echo "$(date +%T) $b EXPORT FAIL" >> $LOG; continue; }
  ssh $JETSON_HOST "mkdir -p ~/h26/s15/onnx/$b" && scp -q $OUT/$b/seg?_fp32.onnx $OUT/$b/export_report.json $JETSON_HOST:~/h26/s15/onnx/$b/
  m=logits; [ $s = 9201 ] && m=meas; [ $s = 9202 ] && m=check2
  ssh $JETSON_HOST "echo '$n $s $m' >> ~/h26/s15/queue.txt"; echo "$(date +%T) $b enqueued $m $(grep -o 'ORTvsKaggle.*' $OUT/$b.export.log)" >> $LOG
done
