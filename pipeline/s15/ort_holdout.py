"""S1.5b step 3 (PC): ORT FP32 per-exit logits of the exported segment chain on the 5,000 hold-out images
(preprocessing identical to orin_logits.py), for the flip check of Orin FP16 logits (unpinned and pinned N2).
Usage: python ort_holdout.py <onnx_dir with seg1..4_fp32.onnx> <holdout_uint8.npy> <norm cifar|imagenet> <out_prefix>
"""
import sys, numpy as np, onnxruntime as ort

od, xp, norm, outp = sys.argv[1:5]; XU = np.load(xp)


def bil(n_in, n_out):
    W = np.zeros((n_out, n_in), np.float32); sc = n_in / n_out
    for d in range(n_out):
        s = max((d + 0.5) * sc - 0.5, 0.0); i0 = min(int(np.floor(s)), n_in - 1); i1 = min(i0 + 1, n_in - 1); w = s - i0
        W[d, i0] += 1 - w; W[d, i1] += w
    return W


if norm == "cifar":
    M, S, Wm = np.array([0.5070, 0.4865, 0.4409], np.float32), np.array([0.2673, 0.2564, 0.2761], np.float32), None
else:
    M, S, Wm = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32), bil(32, 224)
so = ort.SessionOptions(); so.intra_op_num_threads = 8
segs = [ort.InferenceSession(f"{od}/seg{k}_fp32.onnx", so, providers=["CPUExecutionProvider"]) for k in range(1, 5)]
Z = [[] for _ in range(4)]
for i0 in range(0, len(XU), 50):
    x = XU[i0:i0 + 50].astype(np.float32) / 255
    if Wm is not None:
        x = np.matmul(np.matmul(Wm, x), Wm.T)
    h = ((x - M.reshape(1, 3, 1, 1)) / S.reshape(1, 3, 1, 1)).astype(np.float32)
    for j in range(len(h)):                      # exported graphs have batch 1
        f = h[j:j + 1]
        for k, s in enumerate(segs):
            r = dict(zip([o.name for o in s.get_outputs()], s.run(None, {"x": f})))
            Z[k].append(r["z"]); f = r.get("f")
for k in range(4):
    np.save(f"{outp}_e{k+1}.npy", np.concatenate(Z[k]).astype(np.float32))
print("ORT_DONE", outp)
