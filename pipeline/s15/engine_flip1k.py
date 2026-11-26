"""S1.5 prereg sec. 2 (Orin): logits of the full, trunc1-3 and If (forced path 1-4) FP16 engines on the first 1,000 test images
(same preprocessing as orin_logits.py), for the argmax-flip check vs Kaggle FP32 (done on PC by flip_check.py --engines).
Usage: python engine_flip1k.py <eng_dir> <x_test_uint8.npy> <norm cifar|imagenet> <out.npz> [--n 1000]
"""
import sys, numpy as np
sys.path.insert(0, "/home/user/h26/s14b/code")
from cuda import cudart
from trt_s14b import Engine, Form, ck

ed, xp, norm, outp = sys.argv[1:5]; N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1000
XU = np.load(xp)[:N]


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
x = XU.astype(np.float32) / 255
if Wm is not None:
    x = np.matmul(np.matmul(Wm, x), Wm.T)
X = ((x - M.reshape(1, 3, 1, 1)) / S.reshape(1, 3, 1, 1)).astype(np.float32)
ck(cudart.cudaSetDevice(0)); st = ck(cudart.cudaStreamCreate()); out = {}
forms = [("full", 4, "full")] + [(f"trunc{k}", k, "trunc") for k in (1, 2, 3)] + [(f"cond_p{k}", k, "cond") for k in (1, 2, 3, 4)]
eng = {}
for name, k, kind in forms:
    f = name.split("_")[0]
    try:
        if f not in eng:
            eng[f] = Engine(f"{ed}/{f}_fp16.plan")
    except Exception as ex:
        out[f"{name}_error"] = np.array(str(ex)); continue
    F = Form(kind, k, [eng[f]], st)
    out[name] = np.stack([F.once(X[i:i + 1]).reshape(-1).astype(np.float32).copy() for i in range(N)])
np.savez_compressed(outp, **out); print("FLIP1K_DONE", outp, [k for k in out])
