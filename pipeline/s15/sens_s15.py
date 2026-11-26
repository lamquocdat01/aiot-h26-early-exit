"""S1.5 analysis addendum (anh Dat 25-11; NOT used for scoring, prereg unchanged):
1. per ratio used by H1/H2/H3: n batches, mean, 95 % CI (batch bootstrap 10,000, seed 9601) + "CI crosses threshold".
2. sensitivity (secondary): main = per-batch idle_pre subtracted (as fit_predict summarize);
   a = sandwich baseline: idle of batch k = mean(idle_pre_k, idle_pre_{k+1}) in wall-clock order (last batch: idle_pre_k
       and the 'after' idle); b = gross energy (no idle subtracted); c = latency p50.
3. latency CV across batches of the same config.
Usage: python sens_s15.py <meas_root> <out.json>   (meas_root/meas_<net>_s9201[P]/measure_raw.json)
"""
import sys, json, os
import numpy as np

root, outp = sys.argv[1:3]; B = 10000
NETS = {"n1": "meas_n1_s9201", "n2": "meas_n2_s9201P", "n2_unpinned": "meas_n2_s9201", "n3": "meas_n3_s9201",
        "n4": "meas_n4_s9201P", "n4_unpinned": "meas_n4_s9201", "n5": "meas_n5_s9201"}
CLS = {"n1": "cnn", "n2": "cnn", "n2_unpinned": "cnn", "n4": "cnn", "n4_unpinned": "cnn", "n3": "tr", "n5": "tr"}


def load(p):
    R = json.load(open(p)); runs = {}
    for r in R["runs"]:                                  # same selection as summarize: prefer remeasure / unflagged
        k = (r["cfg"], r["rep"])
        if r["tag"] == "remeasure" or k not in runs or (runs[k]["flag"] and not r["flag"]):
            runs[k] = r
    after = [i["mean_mW"][0] for i in R["idle"] if i["tag"] == "after"]
    chrono = sorted(R["runs"], key=lambda r: r["wall"])  # all batches incl. flagged: neighbours in time
    nxt = {}
    for i, r in enumerate(chrono):
        nxt[id(r)] = chrono[i + 1]["idle_pre"]["mean_mW"][0] if i + 1 < len(chrono) else (after[0] if after else r["idle_pre"]["mean_mW"][0])
    M = {}
    for (cfg, rep), r in runs.items():
        if not cfg.startswith("fp16_") or len(cfg.split("_")) > 3:
            continue
        ip = r["idle_pre"]["mean_mW"][0] / 1000; isw = (ip + nxt[id(r)] / 1000) / 2
        v = dict(main=(r["E_J"][0] - ip * r["sec"]) / r["n"] * 1e3, a=(r["E_J"][0] - isw * r["sec"]) / r["n"] * 1e3,
                 b=r["E_J"][0] / r["n"] * 1e3, c=r["lat_p50_ms"])
        kind, pk = cfg.split("_")[1:3]
        for m, x in v.items():
            M.setdefault(m, {}).setdefault(f"{kind}{pk[1]}", []).append(x)
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in M.items()}


def boot(f, arrs, seed=9601):
    rng = np.random.default_rng(seed); o = np.array([f(*[a[rng.integers(0, len(a), len(a))] for a in arrs]) for _ in range(B)])
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))], [float(np.percentile(o, 5)), float(np.percentile(o, 95))]


Dfun = lambda *a: (max(x.mean() for x in a) - min(x.mean() for x in a)) / np.mean([x.mean() for x in a])
res = dict(note="secondary analysis, not used for H1-H4 scoring", nets={})
for net, d in NETS.items():
    p = f"{root}/{d}/measure_raw.json"
    if not os.path.exists(p):
        res["nets"][net] = "missing"; continue
    M = load(p); r = {}
    for m in ("main", "a", "b", "c"):
        C = M[m]; q = {}
        ci95, ci90 = boot(lambda x, y: x.mean() / y.mean(), [C["chain4"], C["full4"]])
        thr = 1.10 if CLS[net] == "tr" else 1.20
        q["H1_R"] = dict(n_batches=[len(C["chain4"]), len(C["full4"])], mean=float(C["chain4"].mean() / C["full4"].mean()), ci95=ci95, ci90=ci90,
                         threshold=thr, pass_ci90=(ci90[1] <= thr) if CLS[net] == "tr" else (ci90[0] >= thr), ci95_crosses=ci95[0] < thr < ci95[1])
        if all(f"cond{k}" in C for k in (1, 2, 3, 4)):
            A = [C[f"cond{k}"] for k in (1, 2, 3, 4)]; ci95, ci90 = boot(Dfun, A)
            q["H2_D"] = dict(n_batches=[len(x) for x in A], mean=float(Dfun(*A)), ci95=ci95, ci90=ci90, threshold=0.05,
                             pass_point=float(Dfun(*A)) <= 0.05, ci95_crosses=ci95[0] < 0.05 < ci95[1])
        for k in (1, 2, 3, 4):                           # H3 inputs: per-path cost ratio vs full (threshold 1 = no saving)
            ci95, _ = boot(lambda x, y: x.mean() / y.mean(), [C[f"chain{k}"], C["full4"]])
            q[f"H3_chain{k}_over_full"] = dict(n_batches=[len(C[f"chain{k}"]), len(C["full4"])], mean=float(C[f"chain{k}"].mean() / C["full4"].mean()),
                                               ci95=ci95, threshold=1.0, ci95_crosses=ci95[0] < 1.0 < ci95[1])
        r[m] = q
    r["lat_cv_pct"] = {k: float(100 * v.std(ddof=1) / v.mean()) for k, v in M["c"].items()}
    r["energy_cv_pct_main"] = {k: float(100 * v.std(ddof=1) / v.mean()) for k, v in M["main"].items()}
    r["H1_flip_vs_main"] = {m: r[m]["H1_R"]["pass_ci90"] != r["main"]["H1_R"]["pass_ci90"] for m in ("a", "b", "c")}
    if "H2_D" in r["main"]:
        r["H2_flip_vs_main"] = {m: r[m]["H2_D"]["pass_point"] != r["main"]["H2_D"]["pass_point"] for m in ("a", "b", "c")}
    res["nets"][net] = r
json.dump(res, open(outp, "w"), indent=1)
for net, r in res["nets"].items():
    if r == "missing":
        print(net, "missing"); continue
    for m in ("main", "a", "b", "c"):
        h = r[m]["H1_R"]; d = r[m].get("H2_D", {})
        print(net, m, "R", round(h["mean"], 3), [round(x, 3) for x in h["ci95"]], "pass90", h["pass_ci90"], "x95", h["ci95_crosses"],
              "| D", d and round(d["mean"], 4), d and [round(x, 4) for x in d["ci95"]], d and d["ci95_crosses"])
    print(net, "latCV%", {k: round(v, 2) for k, v in r["lat_cv_pct"].items()})
