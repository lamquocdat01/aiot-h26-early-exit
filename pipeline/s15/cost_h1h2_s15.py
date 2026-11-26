"""S1.5 (PC): H1, H2 and the seed-9202 2 % check from the per-net summary csvs (fit_predict.py summarize; per-batch idle subtracted).
PREREG_S15 sec. 3-4: CI 90 % by batch bootstrap (10,000, seed 9601), the two configs' batches resampled independently.
Usage: python cost_h1h2_s15.py <summary_dir> <out.json>    (summary_dir holds sum_<net>_s<seed>.csv)
"""
import sys, json, csv, os
import numpy as np

sd, outp = sys.argv[1:3]; B = 10000


def load(p):
    return {(r["kind"], int(r["k"])): np.array([float(v) for v in r["per_rep_net"].split(";")]) for r in csv.DictReader(open(p))
            if r["prec"] == "fp16" and not r["extra"]}


def boot(f, arrays, rng):
    out = np.empty(B)
    for b in range(B):
        out[b] = f(*[a[rng.integers(0, len(a), len(a))] for a in arrays])
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))]


res = {}
for net in ("n1", "n2", "n3", "n4", "n5"):
    p = f"{sd}/sum_{net}_s9201.csv"
    if not os.path.exists(p):
        res[net] = dict(missing=True); continue
    C = load(p); rng = np.random.default_rng(9601); r = dict(cbar={f"{k[0]}{k[1]}": float(v.mean()) for k, v in C.items()}, n_batches={f"{k[0]}{k[1]}": len(v) for k, v in C.items()})
    f, c4 = C[("full", 4)], C[("chain", 4)]
    r["R"] = float(c4.mean() / f.mean()); r["R_ci90"] = boot(lambda a, b: a.mean() / b.mean(), [c4, f], rng)
    r["R_k"] = {k: float(C[("chain", k)].mean() / C[("trunc", k)].mean()) if k < 4 else r["R"] for k in (1, 2, 3, 4) if ("trunc", k) in C or k == 4}
    if all(("cond", k) in C for k in (1, 2, 3, 4)):
        A = [C[("cond", k)] for k in (1, 2, 3, 4)]
        D = lambda *a: (max(x.mean() for x in a) - min(x.mean() for x in a)) / np.mean([x.mean() for x in a])
        r["D"] = float(D(*A)); r["D_ci90"] = boot(D, A, np.random.default_rng(9601)); r["cond_over_full"] = float(np.mean([x.mean() for x in A]) / f.mean())
    else:
        r["D"] = None
    p2 = f"{sd}/sum_{net}_s9202.csv"
    if os.path.exists(p2):
        C2 = load(p2); r["check2_dev_pct"] = {f"{k[0]}{k[1]}": float(100 * (C2[k].mean() / C[k].mean() - 1)) for k in [("full", 4)] + [("chain", k) for k in (1, 2, 3, 4)] if k in C2}
        r["check2_pass"] = all(abs(v) <= 2 for v in r["check2_dev_pct"].values())
        r["R_s9202"] = float(C2[("chain", 4)].mean() / C2[("full", 4)].mean())
    res[net] = r
H1 = dict(transformers={n: res[n].get("R_ci90") for n in ("n3", "n5")}, cnns={n: res[n].get("R_ci90") for n in ("n1", "n2", "n4")})
try:
    h1 = all(res[n]["R_ci90"][1] <= 1.10 for n in ("n3", "n5")) and all(res[n]["R_ci90"][0] >= 1.20 for n in ("n1", "n2", "n4"))
except KeyError:
    h1 = None
built = [n for n in res if res[n].get("D") is not None]
res["H1_PASS"] = h1; res["H2_PASS"] = (all(res[n]["D"] <= 0.05 for n in built) if built else None); res["H2_nets"] = built
json.dump(res, open(outp, "w"), indent=1)
for n in ("n1", "n2", "n3", "n4", "n5"):
    r = res[n]
    if r.get("missing"):
        print(n, "missing"); continue
    print(n, "R", round(r["R"], 3), [round(v, 3) for v in r["R_ci90"]], "D", r["D"] and round(r["D"], 4), r.get("D_ci90"), "chk2", r.get("check2_dev_pct"))
print("H1", res["H1_PASS"], "H2", res["H2_PASS"])
