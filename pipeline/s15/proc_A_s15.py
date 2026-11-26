"""S1.5 step 7 (PC): procedure A (PROC_A_PREREG.md, unchanged) under the PREREG_S15 sec. 5 data protocol:
pool hold-out 5,000 + test 10,000 Orin FP16 logits (15,000); 500 independent splits (seed 9301+i): first n of a random
permutation = calibration, the rest = evaluation. Also the hold-out-vs-test two-sample checks (sec. 6).
Usage: python proc_A_s15.py <net> <seed> <logits_prefix_dir> <cost_summary.csv> <export_report.json> <out_dir>
       logits files: <dir>/<net>_s<seed>_{holdout,test}_e{k}.npy ; labels: y_holdout.npy / y_test.npy in <dir>
"""
import sys, json, csv, os, itertools
import numpy as np
from scipy.stats import beta as betad, t as tdist, ks_2samp, norm

net, seed, ld, costp, repp, od = sys.argv[1:7]; os.makedirs(od, exist_ok=True)
D, DR, DPI, DC = 0.10, 0.05, 0.03, 0.02
LAMS = np.linspace(0.99, 0.50, 20); ALPHAS = (0.01, 0.02, 0.05); NS = (500, 2000, 5000); ETAS = (0.05, 0.0, 0.10); NSPLIT = 500


def load(split):
    Z = [np.load(f"{ld}/{net}_s{seed}_{split}_e{k}.npy").astype(np.float64) for k in range(1, 5)]; y = np.load(f"{ld}/y_{split}.npy")
    conf = np.stack([(lambda e: (e / e.sum(1, keepdims=True)).max(1))(np.exp(z - z.max(1, keepdims=True))) for z in Z], 1)
    return conf, np.stack([z.argmax(1) == y for z in Z], 1)


cH, kH = load("holdout"); cT, kT = load("test"); conf = np.concatenate([cH, cT]); corr = np.concatenate([kH, kT]); N = len(conf)
# ---- sec. 6: hold-out vs test two-sample checks (Holm over 8)
tests = []
for k in range(4):
    p1, p2, n1, n2 = kH[:, k].mean(), kT[:, k].mean(), len(kH), len(kT); pp = (kH[:, k].sum() + kT[:, k].sum()) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)); tests.append((f"acc_e{k+1}", float(p1 - p2), float(2 * norm.sf(abs(z)))))
    r = ks_2samp(cH[:, k], cT[:, k]); tests.append((f"ks_conf_e{k+1}", float(r.statistic), float(r.pvalue)))
order = sorted(range(8), key=lambda i: tests[i][2]); rej = [False] * 8
for r_, i in enumerate(order):
    if tests[i][2] <= 0.05 / (8 - r_):
        rej[i] = True
    else:
        break
shift = [dict(test=t, stat=s, p=p, holm_reject=rj) for (t, s, p), rj in zip(tests, rej)]

# ---- policies (identical to proc_A.py)
fl = json.load(open(repp))["flops"]; Fpath = np.array([sum(fl["seg"][:k]) for k in (1, 2, 3, 4)]) / 1e9; Ftr = np.array(fl["trunc"]) / 1e9
TH = []
for r in (1, 2, 3):
    for E in itertools.combinations((0, 1, 2), r):
        for lam in LAMS:
            TH.append(("chain", E, float(lam)))
for j in range(3):
    TH.append(("trunc", j, None))
PATH = np.zeros((len(TH), N), np.int8); CH = {}
for i, (kind, E, lam) in enumerate(TH):
    if kind == "chain":
        ex = np.full(N, 3)
        for j in sorted(E, reverse=True):
            ex = np.where(conf[:, j] >= lam, j, ex)
        PATH[i] = ex; CH.setdefault(("chain", E), []).append(i)
    else:
        PATH[i] = E; CH.setdefault(("trunc", E), []).append(i)
LOSS = (~corr[np.arange(N)[None, :], PATH] & corr[None, :, 3]).astype(np.float64)          # (|Theta|, N)
ONEH = np.stack([(PATH == k) for k in range(4)], 2).astype(np.float64)                       # (|Theta|, N, 4)
C = {}
for r in csv.DictReader(open(costp)):
    if r["prec"] == "fp16" and not r["extra"]:
        C[(r["kind"], int(r["k"]))] = np.array([float(v) for v in r["per_rep_net"].split(";")])
KEYS = [("full", 4)] + [("chain", k) for k in (1, 2, 3, 4)] + [("trunc", k) for k in (1, 2, 3)]; cbar = {k: C[k].mean() for k in KEYS}; cf = cbar[("full", 4)]


def lut(cmap):        # (|Theta|, 4): cost of each path value under each policy
    L = np.zeros((len(TH), 4))
    for i, (kind, E, _) in enumerate(TH):
        L[i] = [cmap[("chain", k)] for k in (1, 2, 3, 4)] if kind == "chain" else cmap[("trunc", E + 1)]
    return L


def bounds(eta):
    nc = min(len(C[k]) for k in KEYS); tq = tdist.ppf(1 - DC / len(KEYS), nc - 1); U = {}
    for k in KEYS:
        U[k] = max(C[k].mean() + tq * C[k].std(ddof=1) / np.sqrt(len(C[k])) + eta * cbar[k], 0.0)
    kf = ("full", 4); Lf = C[kf].mean() - tq * C[kf].std(ddof=1) / np.sqrt(len(C[kf])) - eta * cbar[kf]
    return lut(U), Lf, max(U.values())


LB = lut(cbar); FL = np.array([Fpath if k == "chain" else np.full(4, Ftr[E]) for k, E, _ in TH])
TOTC = ONEH.sum(1); TOTL = LOSS.sum(1)
BND = {eta: bounds(eta) for eta in ETAS}; dq = DPI / len(TH); lvl = DR / len(CH)
rows = []; per = {}
for n in NS:
    for i in range(NSPLIT):
        cal = np.random.default_rng(9301 + i).permutation(N)[:n]; ne = N - n
        cc = ONEH[:, cal, :].sum(1); ls = LOSS[:, cal].sum(1)                       # calib path counts, loss sums
        ce = TOTC - cc; r_eval = (TOTL - ls) / ne; s_eval = cf - (ce * LB).sum(1) / ne
        fl_cal = (cc * FL).sum(1) / n
        for alpha in ALPHAS:
            cert = []
            for ids in CH.values():
                for t in ids:
                    k_ = int(round(ls[t]))
                    if (1.0 if k_ >= n else float(betad.ppf(1 - lvl, k_ + 1, n - k_))) <= alpha:
                        cert.append(t)
                    else:
                        break
            cert = np.array(cert, int)
            for eta in ETAS:
                LU, Lf, R = BND[eta]; o = dict(out=False)
                if len(cert):
                    m = (cc[cert] * LU[cert]).sum(1) / n; m2 = (cc[cert] * LU[cert] ** 2).sum(1) / n; var = np.maximum(m2 - m * m, 0) * n / (n - 1)
                    ub = m + np.sqrt(2 * var * np.log(2 / dq) / n) + 7 * R * np.log(2 / dq) / (3 * (n - 1)); sLB = Lf - ub; b = int(np.argmax(sLB))
                    if sLB[b] > 0:
                        t = cert[b]; o = dict(out=True, th=int(t), sLB=float(sLB[b]), s_eval=float(s_eval[t]), r_eval=float(r_eval[t]),
                                              vr=bool(r_eval[t] > alpha), vs=bool(s_eval[t] < sLB[b]))
                    tf = cert[int(np.argmin(fl_cal[cert]))]; o.update(fth=int(tf), f_s_eval=float(s_eval[tf]), f_r_eval=float(r_eval[tf]))
                o["ncert"] = int(len(cert)); per.setdefault((eta, n, alpha), []).append(o)


def cp_ci(k, m):
    lo = 0.0 if k == 0 else float(betad.ppf(0.025, k, m - k + 1)); hi = 1.0 if k == m else float(betad.ppf(0.975, k + 1, m - k)); return lo, hi


def name(t):
    k, E, lam = TH[t]; return f"{k}:{'+'.join(str(e + 1) for e in E) if k == 'chain' else E + 1}:{lam if lam is None else round(lam, 3)}"


for (eta, n, alpha), L_ in per.items():
    O = [o for o in L_ if o["out"]]; F = [o for o in L_ if "fth" in o]; m = len(L_)
    vr = sum(o.get("vr", False) for o in L_); vs = sum(o.get("vs", False) for o in L_)
    th = [o["th"] for o in O]; fth = [o["fth"] for o in F]
    rows.append(dict(net=net, seed=int(seed), eta=eta, n=n, alpha=alpha, splits=m, out_rate=len(O) / m,
                     modal=name(max(set(th), key=th.count)) if th else "", sLB_mJ=np.mean([o["sLB"] for o in O]) if O else "",
                     sLB_pct=100 * np.mean([o["sLB"] for o in O]) / cf if O else "", s_eval_pct=100 * np.mean([o["s_eval"] for o in O]) / cf if O else "",
                     r_eval_mean=np.mean([o["r_eval"] for o in O]) if O else "",
                     viol_r=vr / m, viol_r_lo=cp_ci(vr, m)[0], viol_r_hi=cp_ci(vr, m)[1], viol_s=vs / m, viol_s_lo=cp_ci(vs, m)[0], viol_s_hi=cp_ci(vs, m)[1],
                     cert_rate=sum(o["ncert"] > 0 for o in L_) / m, flops_modal=name(max(set(fth), key=fth.count)) if fth else "",
                     flops_s_eval_pct=100 * np.mean([o["f_s_eval"] for o in F]) / cf if F else "",
                     flops_loss_rate=sum(o["f_s_eval"] < 0 for o in F) / m, c_full=cf))
with open(f"{od}/procA_{net}_s{seed}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
np.savez_compressed(f"{od}/splits_{net}_s{seed}.npz", **{f"{eta}_{n}_{alpha}": np.array([(o["out"], o.get("vr", False), o.get("vs", False), o.get("f_s_eval", np.nan) < 0 if "f_s_eval" in o else False,
                                                                                            o.get("sLB", np.nan), o.get("s_eval", np.nan), o.get("f_s_eval", np.nan)) for o in L_], float)
                                                       for (eta, n, alpha), L_ in per.items()})
json.dump(dict(net=net, seed=int(seed), N=N, acc_holdout=kH.mean(0).tolist(), acc_test=kT.mean(0).tolist(), cbar={f"{k[0]}_{k[1]}": v for k, v in cbar.items()},
               shift_checks=shift, n_theta=len(TH)), open(f"{od}/procA_{net}_s{seed}.json", "w"), indent=1, default=float)
for r in rows:
    if r["eta"] == 0.05 and r["n"] == 2000:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("alpha", "out_rate", "modal", "sLB_pct", "s_eval_pct", "viol_r", "viol_s", "flops_s_eval_pct", "flops_loss_rate")})
