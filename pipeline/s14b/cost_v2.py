import os
"""S1.4b wave 2, step 4-5 (PC): cost model v2 (prereg/COST_MODEL_V2_PREREG.md).
  path cost  c = e_net * F_path + kappa_d * calls + beta * bytes_crossed + h * host_checks     (mJ/inference, idle subtracted)
  e_net = c_full_meas(net) / F_full(net)   [measured full engine of THAT net]
  trunc : calls 1, bytes 0, checks 0               -> c = e F + kappa_trunc
  chain : calls k, bytes = sum of FP16 boundary tensors crossed (k-1), checks k -> c = e F + theta_chain*k + beta*B_k
          (kappa_chain and h are collinear because calls == checks == k; only theta_chain = kappa_chain + h is fitted)
  cond  : (step 3: TensorRT executes every branch) -> c = e F_allheads + kappa_cond, path-independent; rho*_cond = never profitable
Shared parameters (all nets): kappa_trunc, theta_chain, beta, kappa_cond. Fit = least squares on relative residuals, wave-1 FP16 N1-N3.
Usage:
  python cost_v2.py fit <out.json>
  python cost_v2.py predict <fit.json> <net> <export_report.json> <full_summary.csv> <out_prediction.json>
  python cost_v2.py score <prediction.json> <summary.csv> <out.json>
"""
import sys, json, csv
import numpy as np
from scipy.optimize import least_squares

R1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "s14b_n1")
SHAPES = {"n1": [[16, 32, 32], [16, 32, 32], [32, 16, 16]]}        # N1 export report predates seg_input_shapes


def geom(rep, net):
    fl = rep["flops"]
    sh = SHAPES.get(net) or [s[1:] for s in rep["seg_input_shapes"][1:]]
    b = [2 * int(np.prod(s)) / 1e6 for s in sh]                      # MB crossed at boundary j (FP16)
    F = {("full", 4): fl["full"]}
    for k in (1, 2, 3):
        F[("trunc", k)] = fl["trunc"][k - 1]
    for k in (1, 2, 3, 4):
        F[("chain", k)] = sum(fl["seg"][:k])
    F_cond = sum(fl["seg"])
    B = {k: sum(b[:k - 1]) for k in (1, 2, 3, 4)}
    return {kk: v / 1e9 for kk, v in F.items()}, F_cond / 1e9, B


def pred(p, kind, k, e, F, Fc, B):
    kt, th, be, kc = p
    if kind == "full":
        return e * F[("full", 4)]
    if kind == "trunc":
        return e * F[("trunc", k)] + kt
    if kind == "chain":
        return e * F[("chain", k)] + th * k + be * B[k]
    return e * Fc + kc


def load(path, prec="fp16"):
    return {(r["kind"], int(r["k"])): float(r["mJ_net"]) for r in csv.DictReader(open(path)) if r["prec"] == prec and not r["extra"]}


def rho(c):
    out = {}
    for j in (1, 2, 3):
        num, den = c[("chain", 4)] - c[("full", 4)], c[("chain", 4)] - c[("chain", j)]
        out[f"chain_{j}"] = float(num / den) if den > 0 else None
    return out


def fit(out):
    data = []
    for n in ("n1", "n2", "n3"):
        rep = json.load(open(f"{R1}/prep/export_report_{n}.json")); F, Fc, B = geom(rep, n); m = load(f"{R1}/summary_{n}.csv")
        e = m[("full", 4)] / F[("full", 4)]
        for kk, y in m.items():
            if kk[0] in ("trunc", "chain"):
                data.append((n, kk, y, e, F, Fc, B))
        data.append((n, ("cond", 0), np.mean([m[("cond", k)] for k in (1, 2, 3, 4)]), e, F, Fc, B))
    res = lambda p: np.array([(y - pred(p, kk[0], kk[1], e, F, Fc, B)) / y for (n, kk, y, e, F, Fc, B) in data])
    sol = least_squares(res, np.array([0.5, 0.3, 1.0, 0.5])); p = sol.x
    rows = [dict(net=n, cfg=f"{kk[0]}_{kk[1]}", meas=y, fit=pred(p, kk[0], kk[1], e, F, Fc, B), rel=(y - pred(p, kk[0], kk[1], e, F, Fc, B)) / y) for (n, kk, y, e, F, Fc, B) in data]
    pc = [abs(r["rel"]) for r in rows if not r["cfg"].startswith("cond")]
    o = dict(params=dict(kappa_trunc=p[0], theta_chain=p[1], beta_mJ_per_MB=p[2], kappa_cond=p[3]), rows=rows,
             in_sample_median_abs_rel_trunc_chain=float(np.median(pc)), in_sample_max=float(np.max(pc)), n_points=len(data))
    json.dump(o, open(out, "w"), indent=1)
    print(json.dumps(o["params"], indent=1), "median", round(o["in_sample_median_abs_rel_trunc_chain"], 4), "max", round(o["in_sample_max"], 4))
    for r in rows:
        print(r["net"], r["cfg"], round(r["meas"], 3), round(r["fit"], 3), f"{100*r['rel']:+.1f}%")


def predict(fitp, net, repp, fullcsv, out):
    P = json.load(open(fitp))["params"]; p = [P["kappa_trunc"], P["theta_chain"], P["beta_mJ_per_MB"], P["kappa_cond"]]
    rep = json.load(open(repp)); F, Fc, B = geom(rep, net); cf = float([r for r in csv.DictReader(open(fullcsv)) if r["cfg"] == "fp16_full_p4"][0]["mJ_net"])
    e = cf / F[("full", 4)]; c = {("full", 4): cf}
    for k in (1, 2, 3):
        c[("trunc", k)] = pred(p, "trunc", k, e, F, Fc, B)
    for k in (1, 2, 3, 4):
        c[("chain", k)] = pred(p, "chain", k, e, F, Fc, B); c[("cond", k)] = pred(p, "cond", k, e, F, Fc, B)
    o = dict(net=net, e_net_mJ_per_GFLOP=e, c_full_meas=cf, params=P, prediction_mJ={f"fp16_{a}_p{b}": v for (a, b), v in c.items()}, rho_star_pred=rho(c),
             rho_cond="never profitable (path-independent cost)", bytes_MB=B, F_GFLOP={f"{a}_{b}": v for (a, b), v in F.items()})
    json.dump(o, open(out, "w"), indent=1); print(json.dumps({k: v for k, v in o.items() if k in ("net", "e_net_mJ_per_GFLOP", "prediction_mJ", "rho_star_pred")}, indent=1))


def score(predp, sump, out):
    P = json.load(open(predp)); m = load(sump); res = {}
    for key, v in P["prediction_mJ"].items():
        kind, k = key.split("_")[1], int(key.split("_")[2][1])
        if (kind, k) in m and kind != "full":
            res[key] = abs(v - m[(kind, k)]) / m[(kind, k)]
    tc = [v for kk, v in res.items() if "_trunc_" in kk or "_chain_" in kk]
    rm, rp = rho(m), P["rho_star_pred"]; rr = {}
    for j in rp:
        if rp[j] is None or rm[j] is None:
            rr[j] = dict(pred=rp[j], meas=rm[j], ok=False)
        else:
            d = abs(rp[j] - rm[j]); rr[j] = dict(pred=rp[j], meas=rm[j], abs=d, rel=d / abs(rm[j]) if rm[j] else None, ok=bool(d <= 0.05 or (rm[j] and d / abs(rm[j]) <= 0.20)))
    cond_meas = [m[("cond", k)] for k in (1, 2, 3, 4) if ("cond", k) in m]
    o = dict(net=P["net"], cost_rel_err=res, median_trunc_chain=float(np.median(tc)), max_trunc_chain=float(np.max(tc)), rho=rr,
             cond_path_spread=(max(cond_meas) - min(cond_meas)) / np.mean(cond_meas) if cond_meas else None,
             pass_cost=bool(np.median(tc) <= 0.10), pass_rho=bool(all(v["ok"] for v in rr.values())))
    json.dump(o, open(out, "w"), indent=1); print(json.dumps(o, indent=1))


if __name__ == "__main__":
    {"fit": lambda: fit(sys.argv[2]), "predict": lambda: predict(*sys.argv[2:7]), "score": lambda: score(*sys.argv[2:5])}[sys.argv[1]]()
