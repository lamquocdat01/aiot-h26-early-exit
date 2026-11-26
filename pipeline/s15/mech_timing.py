"""S1.5 step 5 (Orin): wall-time decomposition of chain4 vs full (batch 1, FP16).
A full   : H2D + full engine + D2H + sync                                (= Form full)
B chain4 : H2D + per segment {run, D2H z, sync, host softmax}           (= Form chain, path 4)
C chain4-nosync: H2D + seg1..4 back-to-back + D2H z4 + one sync          (no per-exit D2H / sync / host check)
Usage: python mech_timing.py <eng_dir> <x.npy float32 or uint8 CIFAR> <out.json> [--iters 3000]
"""
import sys, json, time
import numpy as np
sys.path.insert(0, "/home/user/h26/s14b/code")
from cuda import cudart
from trt_s14b import Engine, Form, chain, ck

ed, xp, outp = sys.argv[1:4]; it = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 3000
X = np.load(xp)
if X.dtype == np.uint8:
    m = np.array([0.5070, 0.4865, 0.4409], np.float32).reshape(1, 3, 1, 1); s = np.array([0.2673, 0.2564, 0.2761], np.float32).reshape(1, 3, 1, 1)
    X = ((X[:500].astype(np.float32) / 255 - m) / s).astype(np.float32)
ck(cudart.cudaSetDevice(0)); st = ck(cudart.cudaStreamCreate())
full = Engine(f"{ed}/full_fp16.plan"); segs = [Engine(f"{ed}/seg{k}_fp16.plan") for k in range(1, 5)]; chain(segs)
FA = Form("full", 4, [full], st); FB = Form("chain", 4, segs, st)


def nosync(x):
    segs[0].h2d("x", x, st)
    for e in segs:
        e.run(st)
    segs[3].d2h("z", st); ck(cudart.cudaStreamSynchronize(st))


res = {}
for name, f in (("A_full", FA.once), ("B_chain4", FB.once), ("C_chain4_nosync", nosync)) * 2:      # two interleaved rounds
    for i in range(300):
        f(X[i % len(X)][None])
    lat = np.empty(it)
    for i in range(it):
        q = time.perf_counter(); f(X[i % len(X)][None]); lat[i] = time.perf_counter() - q
    res.setdefault(name, []).append(float(np.median(lat) * 1e3))
res = {k: dict(p50_ms_rounds=v, p50_ms=float(np.mean(v))) for k, v in res.items()}
res["B_minus_A_ms"] = res["B_chain4"]["p50_ms"] - res["A_full"]["p50_ms"]
res["B_minus_C_sync_d2h_check_ms"] = res["B_chain4"]["p50_ms"] - res["C_chain4_nosync"]["p50_ms"]
res["C_minus_A_gpu_side_ms"] = res["C_chain4_nosync"]["p50_ms"] - res["A_full"]["p50_ms"]
json.dump(res, open(outp, "w"), indent=1); print(json.dumps(res))
