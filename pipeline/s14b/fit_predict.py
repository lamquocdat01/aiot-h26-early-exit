"""S1.4b night 1, steps 3-4 (PC): summarise measurements, fit the frozen cost model (prereg/COST_MODEL_N1_PREREG.md)
on N1+N2, predict N3, and (with --score) score the N3 prediction.
Usage:
  python fit_predict.py summarize <measure_raw.json> <export_report.json> <out.csv>
  python fit_predict.py fit <prec> <sumN1.csv> <sumN2.csv> <n3_export_report.json> <out_prediction.json> <n1_export_report.json> <n2_export_report.json>
  python fit_predict.py score <prediction.json> <sumN3.csv> <out.json>
"""
import sys, json, csv
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as tdist

PN = ["e", "phi_trunc", "phi_chain", "phi_cond", "tau1", "theta_chain", "h_cond"]


def feats(kind, k, fl):
    G = {"full": fl["full"], "trunc": fl["trunc"][k - 1] if k < 4 else fl["full"]}.get(kind, sum(fl["seg"][:k])) / 1e9
    calls = k if kind == "chain" else 1
    checks = k if kind == "chain" else (min(k, 3) if kind == "cond" else 0)
    return G, calls, checks


def predict(p, kind, k, fl):
    e, ft, fc, fd, t1, th, hc = p; G, calls, checks = feats(kind, k, fl)
    phi = dict(full=0.0, trunc=ft, chain=fc, cond=fd)[kind]
    if kind == "chain":
        return e * (1 + phi) * G + th * k
    return e * (1 + phi) * G + t1 + (hc * checks if kind == "cond" else 0.0)


def summarize(raw, rep, out):
    R = json.load(open(raw)); fl = json.load(open(rep))["flops"]
    idle = np.mean([i["mean_mW"][0] for i in R["idle"]]) / 1000; idle_g = np.mean([i["mean_mW"][1] for i in R["idle"]]) / 1000
    runs = {}
    for r in R["runs"]:
        key = (r["cfg"], r["rep"])
        if r["tag"] == "remeasure" or key not in runs or (runs[key]["flag"] and not r["flag"]):
            runs[key] = r
    rows = []
    for cfg in sorted({c for c, _ in runs}):
        rr = [runs[(c, q)] for c, q in sorted(runs) if c == cfg]
        net = np.array([(r["E_J"][0] - (r["idle_pre"]["mean_mW"][0] / 1000 if r.get("idle_pre") else idle) * r["sec"]) / r["n"] * 1e3 for r in rr]); tot = np.array([r["E_J"][0] / r["n"] * 1e3 for r in rr])
        gnet = np.array([(r["E_J"][1] - (r["idle_pre"]["mean_mW"][1] / 1000 if r.get("idle_pre") else idle_g) * r["sec"]) / r["n"] * 1e3 for r in rr]); n = len(net); tq = tdist.ppf(0.95, n - 1)
        P, kind, pk = cfg.split("_")[:3]; k = int(pk[1]); G, calls, checks = feats(kind, k, fl)
        rows.append(dict(cfg=cfg, prec=P, kind=kind, k=k, extra="_".join(cfg.split("_")[3:]), n_reps=n, flagged=sum(r["flag"] for r in rr),
                         mJ_net=net.mean(), mJ_net_lo=net.mean() - tq * net.std(ddof=1) / np.sqrt(n), mJ_net_hi=net.mean() + tq * net.std(ddof=1) / np.sqrt(n),
                         mJ_tot=tot.mean(), mJ_gpu_rail_net=gnet.mean(), lat_p50_ms=np.mean([r["lat_p50_ms"] for r in rr]), lat_p99_ms=np.mean([r["lat_p99_ms"] for r in rr]),
                         G_gflop=G, calls=calls, checks=checks, per_rep_net=";".join(f"{v:.5f}" for v in net), idle_W=idle))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r["cfg"], round(r["mJ_net"], 4), [round(r["mJ_net_lo"], 4), round(r["mJ_net_hi"], 4)], round(r["lat_p50_ms"], 3), r["flagged"])


def load_sum(p, prec):
    return [r for r in csv.DictReader(open(p)) if r["prec"] == prec and not r["extra"]]


def rho(c):   # c: dict (kind,k)->cost ; returns rho*_{d,j}
    out = {}
    for d in ("chain", "cond"):
        if (d, 4) not in c or ("full", 4) not in c:
            continue
        for j in (1, 2, 3):
            num, den = c[(d, 4)] - c[("full", 4)], c[(d, 4)] - c[(d, j)]
            out[f"{d}_{j}"] = float(num / den) if den > 0 else None
    return out


def fit(prec, s1, s2, rep3, out):
    S = [load_sum(s1, prec), load_sum(s2, prec)]
    FL = [json.load(open(p))["flops"] for p in sys.argv[7:9]]
    def resid(p, reps=None):
        res = []
        for rows, fl in zip(S, FL):
            for r in rows:
                y = float(r["mJ_net"]) if reps is None else np.mean(np.array([float(v) for v in r["per_rep_net"].split(";")])[reps])
                res.append((y - predict(p, r["kind"], int(r["k"]), fl)) / y)
        return np.array(res)
    p0 = np.array([0.3, 0.0, 0.5, 0.2, 0.1, 0.1, 0.1])
    sol = least_squares(resid, p0); p = sol.x
    nrep = min(len(r["per_rep_net"].split(";")) for rows in S for r in rows); rng = np.random.default_rng(9404); boot = []
    for _ in range(2000):
        idx = rng.integers(0, nrep, nrep); boot.append(least_squares(lambda q: resid(q, idx), p).x)
    boot = np.array(boot); ci = {n: [float(p[i]), float(np.percentile(boot[:, i], 5)), float(np.percentile(boot[:, i], 95))] for i, n in enumerate(PN)}
    rows_out = []; ys, yh = [], []
    for name, rows, fl in zip(("N1", "N2"), S, FL):
        for r in rows:
            y = float(r["mJ_net"]); h = predict(p, r["kind"], int(r["k"]), fl); ys.append(y); yh.append(h)
            rows_out.append(dict(net=name, cfg=r["cfg"], meas=y, fit=h, rel_resid=(y - h) / y))
    ys, yh = np.array(ys), np.array(yh); R2 = float(1 - ((ys - yh) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())
    fl3 = json.load(open(rep3))["flops"]; pred = {}
    for kind, ks in (("full", [4]), ("trunc", [1, 2, 3]), ("chain", [1, 2, 3, 4]), ("cond", [1, 2, 3, 4])):
        for k in ks:
            pred[f"{prec}_{kind}_p{k}"] = float(predict(p, kind, k, fl3))
    pboot = [{f"{prec}_{kind}_p{k}": predict(q, kind, k, fl3) for kind, ks in (("full", [4]), ("trunc", [1, 2, 3]), ("chain", [1, 2, 3, 4]), ("cond", [1, 2, 3, 4])) for k in ks} for q in boot[:500]]
    pred_ci = {k: [float(np.percentile([b[k] for b in pboot], 5)), float(np.percentile([b[k] for b in pboot], 95))] for k in pred}
    c = {(k.split("_")[1], int(k.split("_")[2][1])): v for k, v in pred.items()}
    # secondary (not gate): anchored on the measured N3 full engine -> prediction stated as a function of c_full_meas
    G3 = feats("full", 4, fl3)[0]
    anchored = {k: dict(slope_on_cfull_meas=float((1 + dict(full=0, trunc=p[1], chain=p[2], cond=p[3])[k.split("_")[1]]) * feats(k.split("_")[1], int(k.split("_")[2][1]), fl3)[0] / G3),
                        offset_mJ=float(predict(p, k.split("_")[1], int(k.split("_")[2][1]), fl3) - p[0] * (1 + dict(full=0, trunc=p[1], chain=p[2], cond=p[3])[k.split("_")[1]]) * feats(k.split("_")[1], int(k.split("_")[2][1]), fl3)[0]))
                for k in pred}
    res = dict(prec=prec, params=ci, R2=R2, fit_rows=rows_out, median_abs_rel_resid=float(np.median(np.abs([r["rel_resid"] for r in rows_out]))),
               max_abs_rel_resid=float(np.max(np.abs([r["rel_resid"] for r in rows_out]))),
               N3_prediction_mJ=pred, N3_prediction_ci90=pred_ci, N3_rho_star_pred=rho(c), N3_secondary_anchored=anchored,
               prereg="prereg/COST_MODEL_N1_PREREG.md", n_boot=2000)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("fit_rows", "N3_secondary_anchored", "N3_prediction_ci90")}, indent=1))
    for r in rows_out:
        print(r["net"], r["cfg"], round(r["meas"], 4), round(r["fit"], 4), f"{100*r['rel_resid']:+.1f}%")


def score(predp, s3, out):
    P = json.load(open(predp)); prec = P["prec"]; rows = load_sum(s3, prec); meas = {r["cfg"]: float(r["mJ_net"]) for r in rows}
    err = {k: abs(P["N3_prediction_mJ"][k] - meas[k]) / meas[k] for k in P["N3_prediction_mJ"] if k in meas}
    cm = {(k.split("_")[1], int(k.split("_")[2][1])): v for k, v in meas.items()}; rm = rho(cm); rp = P["N3_rho_star_pred"]; rerr = {}
    for k in rp:
        if rp[k] is None or rm.get(k) is None:
            rerr[k] = None
        elif abs(rm[k]) < 0.05:
            rerr[k] = dict(abs_err=abs(rp[k] - rm[k]), ok=abs(rp[k] - rm[k]) <= 0.05)
        else:
            rerr[k] = dict(rel_err=abs(rp[k] - rm[k]) / abs(rm[k]), ok=abs(rp[k] - rm[k]) / abs(rm[k]) <= 0.10)
    rel = [v["rel_err"] if "rel_err" in v else (0.0 if v["ok"] else 1.0) for v in rerr.values() if v]
    full = meas.get(f"{prec}_full_p4"); anch = {k: v["slope_on_cfull_meas"] * full + v["offset_mJ"] for k, v in P["N3_secondary_anchored"].items()} if full else {}
    res = dict(prec=prec, cost_rel_err=err, cost_median=float(np.median(list(err.values()))), cost_max=float(np.max(list(err.values()))),
               rho_meas=rm, rho_pred=rp, rho_err=rerr, rho_median=float(np.median(rel)) if rel else None,
               G_C1=bool(np.median(list(err.values())) <= 0.10 and rel and np.median(rel) <= 0.10),
               secondary_anchored_err={k: abs(v - meas[k]) / meas[k] for k, v in anch.items() if k in meas})
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "summarize":
        summarize(*sys.argv[2:5])
    elif m == "fit":
        fit(*sys.argv[2:7])
    else:
        score(*sys.argv[2:5])
