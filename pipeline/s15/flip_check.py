"""S1.5 prereg sec. 2 (PC): argmax flip rate of Orin FP16 segment-chain logits vs Kaggle PyTorch FP32 logits, per exit, test 10,000
(+ hold-out reported). PASS if every exit <= 1 %.  Usage: python flip_check.py <net> <seed> <orin_dir> <kaggle_run_dir> <out.json> [--engines flip1k.npz]"""
import sys, json, numpy as np
net, seed, od, kd, out = sys.argv[1:6]; r = dict(net=net, seed=int(seed))
for sp in ("test", "holdout"):
    f = [float((np.load(f"{od}/{net}_s{seed}_{sp}_e{k}.npy").argmax(1) != np.load(f"{kd}/logits_{sp}_e{k}.npy").argmax(1)).mean()) for k in range(1, 5)]
    r[f"flip_{sp}"] = f
r["pass_test_le_1pct"] = max(r["flip_test"]) <= 0.01
if "--engines" in sys.argv:          # full/trunc/If engines on the first 1,000 test images (engine_flip1k.py npz)
    Z = np.load(sys.argv[sys.argv.index("--engines") + 1]); ex = dict(full=4, trunc1=1, trunc2=2, trunc3=3, cond_p1=1, cond_p2=2, cond_p3=3, cond_p4=4); fe = {}
    for nm, k in ex.items():
        if nm in Z.files:
            ref = np.load(f"{kd}/logits_test_e{k}.npy")[:len(Z[nm])]; fe[nm] = float((Z[nm].argmax(1) != ref.argmax(1)).mean())
    r["flip_engines_1k"] = fe; r["pass_engines_le_1pct"] = bool(fe) and max(fe.values()) <= 0.01
json.dump(r, open(out, "w"), indent=1); print(json.dumps(r))
