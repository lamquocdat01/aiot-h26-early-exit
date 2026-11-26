"""S1.4b wave 2, step 6 (PC): procedure A (Theorem A1) on real Orin logits + measured joule costs (prereg/PROC_A_PREREG.md).
Usage: python proc_A.py <net> <logits_dir (holdout/test Orin FP16: logits_{split}_e{k}.npy, y_{split}.npy)> <cost_summary.csv>
                        <export_report.json> <out_dir> [--draws 200] [--eta 0.05]
"""
import argparse, json, csv, os, itertools
import numpy as np
from scipy.stats import beta as betad, t as tdist

ap = argparse.ArgumentParser(); ap.add_argument("net"); ap.add_argument("ld"); ap.add_argument("costs"); ap.add_argument("rep"); ap.add_argument("out")
ap.add_argument("--draws", type=int, default=200); ap.add_argument("--etas", default="0.05,0.0,0.10")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
D, DR, DPI, DC = 0.10, 0.05, 0.03, 0.02
LAMS = np.linspace(0.99, 0.50, 20); ALPHAS = (0.01, 0.02, 0.05); NS = (500, 2000, 5000)


def load(split):
    Z = [np.load(f"{a.ld}/logits_{split}_e{k}.npy").astype(np.float64) for k in range(1, 5)]; y = np.load(f"{a.ld}/y_{split}.npy")
    conf = np.stack([(np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)).max(1) for z in Z], 1)
    corr = np.stack([z.argmax(1) == y for z in Z], 1); return conf, corr


confH, corrH = load("holdout"); confT, corrT = load("test")
fl = json.load(open(a.rep))["flops"]; Fpath = np.array([sum(fl["seg"][:k]) for k in (1, 2, 3, 4)]) / 1e9; Ftr = np.array(fl["trunc"]) / 1e9


def paths(conf, E, lam):
    ex = np.full(len(conf), 3)
    for j in sorted(E, reverse=True):
        ex = np.where(conf[:, j] >= lam, j, ex)
    return ex


thetas = []
for r in (1, 2, 3):
    for E in itertools.combinations((0, 1, 2), r):
        for lam in LAMS:
            thetas.append(dict(kind="chain", E=E, lam=float(lam), chain=("chain", E)))
for j in range(3):
    thetas.append(dict(kind="trunc", j=j, chain=("trunc", j)))
for th in thetas:
    for tag, conf, corr in (("H", confH, corrH), ("T", confT, corrT)):
        ex = paths(conf, th["E"], th["lam"]) if th["kind"] == "chain" else np.full(len(conf), th["j"])
        th["ex" + tag] = ex; th["loss" + tag] = (~corr[np.arange(len(ex)), ex] & corr[:, 3]).astype(float)
    th["r_test"] = th["lossT"].mean()
LOSSH = np.stack([th["lossH"] for th in thetas]); CHAINS = {}
for i, th in enumerate(thetas):
    CHAINS.setdefault(th["chain"], []).append(i)

# ---------------- measured costs: per-rep batch means
C = {}
for r in csv.DictReader(open(a.costs)):
    if r["prec"] == "fp16" and not r["extra"]:
        C[(r["kind"], int(r["k"]))] = np.array([float(v) for v in r["per_rep_net"].split(";")])
KEYS = [("full", 4)] + [("chain", k) for k in (1, 2, 3, 4)] + [("trunc", k) for k in (1, 2, 3)]
cbar = {k: C[k].mean() for k in KEYS}


def cost_bounds(eta):
    nc = min(len(C[k]) for k in KEYS); tq = tdist.ppf(1 - DC / len(KEYS), nc - 1); U = {}
    for k in KEYS:
        h = tq * C[k].std(ddof=1) / np.sqrt(len(C[k])); U[k] = max(C[k].mean() + h + eta * cbar[k], 0.0)
    kf = ("full", 4); Lf = C[kf].mean() - tq * C[kf].std(ddof=1) / np.sqrt(len(C[kf])) - eta * cbar[kf]
    return U, Lf


def vec(th, cmap):        # per-sample cost given exit path
    if th["kind"] == "trunc":
        return np.full(len(th["exH"]), cmap[("trunc", th["j"] + 1)]), None
    lut = np.array([cmap[("chain", k)] for k in (1, 2, 3, 4)]); return lut[th["exH"]], lut[th["exT"]]


s_true = {}
for i, th in enumerate(thetas):
    if th["kind"] == "trunc":
        s_true[i] = cbar[("full", 4)] - cbar[("trunc", th["j"] + 1)]
    else:
        s_true[i] = cbar[("full", 4)] - np.array([cbar[("chain", k)] for k in (1, 2, 3, 4)])[th["exT"]].mean()
flops_H = [Ftr[th["j"]] * np.ones(len(th["exH"])) if th["kind"] == "trunc" else Fpath[th["exH"]] for th in thetas]


def cp(k, n, d):
    return 1.0 if k >= n else float(betad.ppf(1 - d, k + 1, n - k))


def run(n, alpha, eta, rng):
    U, Lf = cost_bounds(eta); R = max(U.values()); w = np.bincount(rng.integers(0, len(confH), n), minlength=len(confH)).astype(float)
    sums = LOSSH @ w; lvl = DR / len(CHAINS); cert = []
    for ids in CHAINS.values():
        for i in ids:
            if cp(int(round(sums[i])), n, lvl) <= alpha:
                cert.append(i)
            else:
                break
    best, bLB = None, 0.0; dq = DPI / len(thetas)
    for i in cert:
        v = vec(thetas[i], U)[0]; m = float(v @ w) / n; var = max(float((v * v) @ w) - n * m * m, 0) / (n - 1)
        ub = m + np.sqrt(2 * var * np.log(2 / dq) / n) + 7 * R * np.log(2 / dq) / (3 * (n - 1)); sLB = Lf - ub
        if sLB > bLB:
            best, bLB = i, sLB
    fl_best = min(cert, key=lambda i: float(flops_H[i] @ w) / n) if cert else None      # FLOPs-based selector (same certified set)
    o = dict(out=best is not None, n_cert=len(cert))
    if best is not None:
        th = thetas[best]; o.update(theta=f"{th['kind']}:{th.get('E', th.get('j'))}:{th.get('lam')}", sLB=bLB, s_test=s_true[best], r_test=th["r_test"],
                                   viol_r=bool(th["r_test"] > alpha), viol_s=bool(s_true[best] < bLB))
    if fl_best is not None:
        th = thetas[fl_best]; o.update(flops_theta=f"{th['kind']}:{th.get('E', th.get('j'))}:{th.get('lam')}", flops_s_test=s_true[fl_best], flops_r_test=th["r_test"])
    return o


rows = []
for eta in map(float, a.etas.split(",")):
    for n in NS:
        for alpha in ALPHAS:
            rng = np.random.default_rng(9601 + int(alpha * 1000) + n); res = [run(n, alpha, eta, rng) for _ in range(a.draws)]; O = [r for r in res if r["out"]]
            F = [r for r in res if "flops_s_test" in r]; cf = cbar[("full", 4)]
            modal = max({r["theta"] for r in O}, key=[r["theta"] for r in O].count) if O else None
            row = dict(net=a.net, eta=eta, n=n, alpha=alpha, out_rate=len(O) / a.draws, none_rate=1 - len(O) / a.draws, modal=modal,
                       sLB_mJ=np.mean([r["sLB"] for r in O]) if O else None, sLB_pct=100 * np.mean([r["sLB"] for r in O]) / cf if O else None,
                       s_test_mJ=np.mean([r["s_test"] for r in O]) if O else None, s_test_pct=100 * np.mean([r["s_test"] for r in O]) / cf if O else None,
                       viol_r=sum(r.get("viol_r", False) for r in res) / a.draws, viol_s=sum(r.get("viol_s", False) for r in res) / a.draws,
                       cert_rate=sum(r["n_cert"] > 0 for r in res) / a.draws,
                       flops_s_test_pct=100 * np.mean([r["flops_s_test"] for r in F]) / cf if F else None,
                       flops_loses_joule_rate=sum(r["flops_s_test"] < 0 for r in F) / a.draws if F else None,
                       flops_modal=max({r["flops_theta"] for r in F}, key=[r["flops_theta"] for r in F].count) if F else None, c_full=cf)
            rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
with open(f"{a.out}/procA_{a.net}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
best_test = max(((s_true[i], i) for i, th in enumerate(thetas) if th["r_test"] <= 0.05), default=None)
json.dump(dict(net=a.net, rows=rows, cbar=dict((f"{k[0]}_{k[1]}", v) for k, v in cbar.items()), n_theta=len(thetas),
               oracle_best_saving_test_at_r_le_5pct=best_test[0] if best_test else None,
               acc_test=corrT.mean(0).tolist(), acc_holdout=corrH.mean(0).tolist()), open(f"{a.out}/procA_{a.net}.json", "w"), indent=1, default=float)
