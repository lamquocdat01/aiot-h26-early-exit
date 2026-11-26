#!/bin/bash
# S1.5 step 4 for N2 exit 1 (flip 1.02 % on test 10k, seed 9205): bisect seg1 FP32 pins vs Kaggle FP32 on the first 1,000 test images.
H=~/h26/s15; V=~/h26/venv/bin/python; F=$H/fix; mkdir -p $F
$V - <<'P'
import numpy as np
XU = np.load("/home/user/h26/data/x_test_uint8.npy")[:1000].astype(np.float32) / 255
def bil(n_in, n_out):
    W = np.zeros((n_out, n_in), np.float32); sc = n_in / n_out
    for d in range(n_out):
        s = max((d + 0.5) * sc - 0.5, 0.0); i0 = min(int(np.floor(s)), n_in - 1); i1 = min(i0 + 1, n_in - 1); w = s - i0
        W[d, i0] += 1 - w; W[d, i1] += w
    return W
W = bil(32, 224); x = np.matmul(np.matmul(W, XU), W.T)
M = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1); S = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
np.save("/home/user/h26/s15/fix/x_im_1k.npy", ((x - M) / S).astype(np.float32))
P
cd ~/h26/s14b/code && $V $H/code/fp16_bisect.py $H/onnx/n2_s9205/seg1_fp32.onnx $F/ref_n2_s9205_e1_1k.npz z $F/x_im_1k.npy $F/bisect_n2_s9205_seg1.json \
  none all "re:/head/" "re:body\.4" "re:body\.[0-3]/" "re:body\.4\.2" "re:body\.4\.2,re:/head/" "type:CONVOLUTION" "type:POOLING+CONCATENATION+MATRIX_MULTIPLY+FULLY_CONNECTED"
rm -f $F/x_im_1k.npy; echo FIXN2_DONE
