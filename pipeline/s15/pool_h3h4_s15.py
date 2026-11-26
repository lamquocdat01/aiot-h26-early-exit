"""S1.5 (PC): H3 and H4 per seed and pooled (PREREG_S15 sec. 4) from proc_A_s15.py splits_<net>_s<seed>.npz.
Columns per split: out, viol_r, viol_s, flops_loss, sLB, s_eval, f_s_eval.
Usage: python pool_h3h4_s15.py <procA_dir> <out.json> [--exclude n5,...]   (excluded = below threshold / failed flip: not in H3)
"""
import sys, json, glob, re, os
import numpy as np
from scipy.stats import beta as betad

pd, outp = sys.argv[1:3]; EXC = sys.argv[sys.argv.index("--exclude") + 1].split(",") if "--exclude" in sys.argv else []
ALPHAS = ("0.01", "0.02", "0.05"); NS = ("500", "2000", "5000"); ETA = "0.05"; DELTA = 0.1
S = {}
for f in glob.glob(f"{pd}/splits_n?_s92??.npz"):
    net, seed = re.search(r"splits_(n\d)_s(\d+)", f).groups(); S.setdefault(net, {})[int(seed)] = dict(np.load(f))


def cp(k, m):
    return [0.0 if k == 0 else float(betad.ppf(0.025, k, m - k + 1)), 1.0 if k == m else float(betad.ppf(0.975, k + 1, m - k))]


def key(n, a):
    return f"{ETA}_{n}_{a}"


def stats(arrs, n, a):
    A = np.concatenate([x[key(n, a)] for x in arrs]); m = len(A)
    out = A[:, 0] > 0; vr, vs, fl = int(A[:, 1].sum()), int(A[:, 2].sum()), int(A[:, 3].sum())
    return dict(splits=m, out_rate=float(out.mean()), out_pos_rate=float((out & (A[:, 4] > 0)).mean()), none_rate=float(1 - out.mean()),
                viol_r=vr / m, viol_r_ci=cp(vr, m), viol_s=vs / m, viol_s_ci=cp(vs, m), flops_loss_rate=fl / m,
                s_eval_mean_mJ=float(np.nanmean(A[out, 5])) if out.any() else None)


def evaluate(sel):          # sel: net -> list of split dicts
    r = dict(cells={}, H4_fail_cells=[])
    for net, arrs in sel.items():
        for n in NS:
            for a in ALPHAS:
                c = stats(arrs, n, a); r["cells"][f"{net}|n{n}|a{a}"] = c
                if c["viol_r"] > DELTA or c["viol_s"] > DELTA:
                    r["H4_fail_cells"].append(f"{net}|n{n}|a{a}")
    r["H4_PASS"] = not r["H4_fail_cells"]
    g = lambda net, a: r["cells"][f"{net}|n2000|a{a}"]
    tr = [n for n in ("n3", "n5") if n in sel and n not in EXC]; cn = [n for n in ("n1", "n2", "n4") if n in sel and n not in EXC]
    ha = {n: sum(g(n, a)["out_pos_rate"] >= 0.5 for a in ALPHAS) for n in tr}
    hb = {n: sum(g(n, a)["none_rate"] >= 0.5 for a in ALPHAS) for n in cn}
    hc = {n: sum(g(n, a)["flops_loss_rate"] >= 0.5 for a in ALPHAS) for n in cn}
    r["H3"] = dict(transformers=tr, cnns=cn, a_alpha_counts=ha, b_alpha_counts=hb, c_alpha_counts=hc,
                   a=bool(tr) and all(v >= 2 for v in ha.values()), b=bool(cn) and all(v == 3 for v in hb.values()),
                   c=sum(v >= 2 for v in hc.values()) >= 2)
    r["H3"]["PASS"] = r["H3"]["a"] and r["H3"]["b"] and r["H3"]["c"]
    return r


res = dict(excluded_from_H3=EXC, seeds={n: sorted(v) for n, v in S.items()})
res["pooled"] = evaluate({n: [v[s] for s in sorted(v)] for n, v in S.items()})
seeds = sorted({s for v in S.values() for s in v}); res["per_seed"] = {}
for s in seeds:
    res["per_seed"][s] = evaluate({n: [v[s]] for n, v in S.items() if s in v})
res["per_seed_counts"] = dict(H3_pass=sum(r["H3"]["PASS"] for r in res["per_seed"].values()), H4_pass=sum(r["H4_PASS"] for r in res["per_seed"].values()), seeds=len(seeds))
json.dump(res, open(outp, "w"), indent=1)
P = res["pooled"]; print("POOLED H3", P["H3"]); print("POOLED H4", P["H4_PASS"], P["H4_fail_cells"]); print("per-seed", res["per_seed_counts"])
for net in sorted(S):
    print(net, [(a, round(P["cells"][f"{net}|n2000|a{a}"]["out_rate"], 3), round(P["cells"][f"{net}|n2000|a{a}"]["flops_loss_rate"], 3)) for a in ALPHAS])
