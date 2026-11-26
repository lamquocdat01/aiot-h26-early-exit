#!/bin/bash
# S1.5 Jetson job queue: runs "net seed mode" lines from ~/h26/s15/queue.txt in order (append-only), one at a time.
# Before every job: stop any running aicamera_* container (never start). Stops at a line "END". Status: ~/h26/s15/queue_status.txt
Q=~/h26/s15/queue.txt; D=~/h26/s15/queue_done.txt; S=~/h26/s15/queue_status.txt; touch $Q $D
while true; do
  line=$(grep -vxF -f $D $Q | head -1)
  if [ -z "$line" ]; then echo "$(date '+%F %T') idle" > $S; sleep 30; continue; fi
  if [ "$line" = "END" ]; then echo "$(date '+%F %T') END" > $S; echo END >> $D; exit 0; fi
  r=$(docker ps --format '{{.Names}}' | grep '^aicamera_' || true); [ -n "$r" ] && docker stop $r >> ~/h26/s15/out/docker_stop.log 2>&1
  echo "$(date '+%F %T') running $line" > $S
  ~/h26/s15/code/run_s15.sh $line
  echo "$line" >> $D
done
