#!/bin/bash
# S1.5 step 6 (Orin), one (net, seed): compose -> build (PIN for n5) -> check -> Orin FP16 logits (hold-out + test) -> [measure]
# Usage: run_s15.sh <net> <seed> <mode: meas|check2|logits|flip1k>
#   script: <name> 0 script -> runs ~/h26/s15/code/job_<name>.sh (GPU side jobs kept inside the queue so they never overlap a measurement)
#   suffix P on a mode (measP, check2P, logitsP): engines in eng/<n>_s<s>P, measurements in meas_<n>_s<s>P (N2 fix)
#   bridge: 1 batch fp16_full_p4 on the existing engines (S1.5b step 2), no rebuild
#   suffix R: re-run of a mode already in queue_done (S1.5b option b: N1/N2 re-measured with N3-N5)
#   flip1k: only the full/trunc/If logits on the first 1,000 test images (prereg sec. 2); meas also runs it at the end
#   meas   (seed 9201): compose + all forms; measure full, trunc1-3, chain1-4, cond p1-4 (10 reps, idle before each batch)
#   check2 (seed 9202): compose; measure full + chain1-4 (2 % cost check)
#   logits (other seeds): segment engines only
set -u; n=$1; s=$2; mode=$3; mode=${mode%R}; P=""; case $mode in *P) P=P; mode=${mode%P};; esac;   # suffix R = re-run (S1.5b option b), same job
 H=~/h26/s15; L=$H/out/run_s15.log; V=~/h26/venv/bin/python; C=~/h26/s14b/code
T=$H/onnx/${n}_s$s; E=$H/eng/${n}_s$s$P; O=$H/out; mkdir -p $E $O/logits
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
norm=imagenet; [ $n = n1 ] && norm=cifar
PIN=""; PIN_ENG=""; [ $n = n5 ] && PIN="te:ELEMENTWISE:attn/qkv/Add"
# step-4 fix N2 exit 1 (mode suffix P): pin layer1 (body.4) of segment 1 to FP32 in seg1/full/trunc/cond (not seg3, whose block 4 shares the name)
[ $n = n2 ] && [ -n "$P" ] && PIN="te:*:^(s1_)?/body/body\.4/" && PIN_ENG="^(seg1|full|trunc|cond)"; export PIN PIN_ENG
[ $n = n4 ] && [ -n "$P" ] && PIN="te:*:^(s1_)?/body/body\.[0-2]/" && PIN_ENG="^(seg1|full|trunc|cond)"; export PIN PIN_ENG
idx=${n#n}
if { [ $mode = meas ] || [ $mode = check2 ] || [ "${mode#bridge}" != "$mode" ]; } && [ -f $H/POWER_FAIL ]; then log "$n s$s $mode SKIPPED: POWER_FAIL present"; exit 3; fi
flip1k() { log "$n s$s flip1k"; (cd $C && $V $H/code/engine_flip1k.py $E ~/h26/data/x_test_uint8.npy $norm $O/flip1k_${n}_s$s$P.npz >> $L 2>&1); }
if [ $mode = script ]; then log "script job_$n"; bash $H/code/job_$n.sh >> $O/job_$n.log 2>&1; log "end script job_$n"; exit 0; fi
if [ "${mode#bridge}" != "$mode" ]; then X=$T/x_pool_f32.npy; [ $n = n1 ] && X=~/h26/data/x_test_uint8.npy
  log "$n s$s bridge"; (cd $C && $V $H/code/measure_s15.py $n $E $X $O/${mode}_${n}_s$s --precs fp16 --reps 1 --n_min 2000 --seed 9590 --only fp16_full_p4 --idle_each 5 --idle 30 >> $L 2>&1); log "$n s$s end bridge"; exit 0; fi
if [ $mode = flip1k ]; then flip1k; log "$n s$s end flip1k"; exit 0; fi
if [ $mode != logits ]; then
  log "$n s$s compose"; (cd $C && python3 compose_s14b_v2.py $T ~/h26/data/x_test_uint8.npy ~/h26/s14b/onnx/ptq_idx.npy ~/h26/s14b/onnx/pool_idx.npy --int8 0 --norm $norm > $O/compose_${n}_s$s.log 2>&1)
fi
log "$n s$s build PIN=$PIN"; (cd $C && $V $H/code/build_s15.py $T $E > $O/build_${n}_s$s$P.log 2>&1)
if [ $mode != logits ]; then
  log "$n s$s check"; (cd $C && $V check_s14b.py $T $E $O/check_${n}_s${s}${P}_fp16.json --prec fp16 --tol 0.15 > $O/check_${n}_s$s$P.log 2>&1)
fi
log "$n s$s logits"; (cd $C && $V orin_logits.py $E ~/h26/s14b/onnx/holdout_uint8.npy $norm $O/logits/${n}_s${s}${P}_holdout >> $L 2>&1 && \
                       $V orin_logits.py $E ~/h26/data/x_test_uint8.npy $norm $O/logits/${n}_s${s}${P}_test >> $L 2>&1)
X=$T/x_pool_f32.npy; [ $n = n1 ] && X=~/h26/data/x_test_uint8.npy
if [ $mode = meas ]; then
  ONLY=fp16_full_p4,fp16_trunc_p1,fp16_trunc_p2,fp16_trunc_p3,fp16_chain_p1,fp16_chain_p2,fp16_chain_p3,fp16_chain_p4,fp16_cond_p1,fp16_cond_p2,fp16_cond_p3,fp16_cond_p4
  log "$n s$s measure"; (cd $C && $V $H/code/measure_s15.py $n $E $X $O/meas_${n}_s$s$P --precs fp16 --reps 10 --n_min 2000 --seed $((9500+idx)) --only $ONLY --idle_each 5 --idle 30 >> $L 2>&1)
elif [ $mode = check2 ]; then
  log "$n s$s measure(check2)"; (cd $C && $V $H/code/measure_s15.py $n $E $X $O/meas_${n}_s$s$P --precs fp16 --reps 10 --n_min 2000 --seed $((9510+idx)) --only fp16_full_p4,fp16_chain_p1,fp16_chain_p2,fp16_chain_p3,fp16_chain_p4 --idle_each 5 --idle 30 >> $L 2>&1)
fi
[ $mode = meas ] && flip1k
log "$n s$s end $mode"
