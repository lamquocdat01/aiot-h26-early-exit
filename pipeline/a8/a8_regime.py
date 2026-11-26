"""A8 step 2 (PC, EXPLORATORY / post-hoc; no new measurement). Regime analysis on S1.5 numbers.
Policy per split: procedure A's output (n=2000, eta=5 %, alpha in {1,2,5} %) re-derived exactly as in s15/proc_A_s15.py;
if A returns "none", the FLOPs selector's policy over the risk-certified set (flagged 'flops'); if nothing is
risk-certified, full (flagged 'full').
Costs (seed 9201 engines, 10 batches/config, 25-11 wave): per batch D = net mJ/img (idle_pre subtracted), G = gross mJ/img
(E_J/n, back-to-back = saturated), T = wall ms/img (sec/n, incl. host sync), p50/p99 ms, idle W.
C1: phi(lam) = lam*D_full/(P_s+lam*D_full), sigma_net = 1 - sum pi_k D_k / D_full, sigma_sys = phi*sigma_net;
    fps in {5,10,30,60, sat = 1/T_full}; P_s = mean idle_pre.
C2: sigma_net (fixed rate; sign = board-level sign at any fixed rate) vs sigma_gross = 1 - sum pi_k G_k / G_full (saturated).
C3: throughput gain = T_full / sum pi_k T_k - 1; streams at 30/10 fps with deadline-miss <= 1 % (deadline = period):
    N periodic streams, uniform random phases, FIFO single server (batch 1), service = path k ~ pi (i.i.d.),
    time | path ~ lognormal(median p50_k, sigma = ln(p99_k/p50_k)/2.326); 10 s horizon.
C4: dynamic policy vs static trunc_k with mean time nearest the policy's; Poisson arrivals, rate = rho / E[S_policy]
    (same stream for both), deadline = 5 * E[S_policy]; p99 response and miss rate; 20,000 arrivals.
Bootstrap/replicates: B = 1000 (C1/C2 CI), R = 200 (C3/C4), seeds 9401+.
Usage: python a8_regime.py <s15_results_dir> <logits_dir> <onnx_dir> <out_dir>
"""
import sys, json, csv, os, itertools
import numpy as np
from scipy.stats import beta as betad, t as tdist

RES, LD, OD, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
NETS = {"n1": LD, "n2": f"{LD}/n2P", "n3": LD, "n4": f"{LD}/n4P", "n5": LD}
SEEDS = range(9201, 9206); ALPHAS = (0.01, 0.02, 0.05); NCAL = 2000; ETA = 0.05; NSPLIT = 500
DR, DPI, DC = 0.05, 0.03, 0.02; LAMS = np.linspace(0.99, 0.50, 20); FPS = (5, 10, 30, 60)


def batches(net):
    R = json.load(open(f"{RES}/cost/raw_{net}_s9201.json")); runs = {}
    for r in R["runs"]:
        k = (r["cfg"], r["rep"])
        if r["tag"] == "remeasure" or k not in runs or (runs[k]["flag"] and not r["flag"]):
            runs[k] = r
    B = {}
    for (cfg, _), r in runs.items():
        _, kind, pk = cfg.split("_")[:3]; ip = r["idle_pre"]["mean_mW"][0] / 1000
        B.setdefault((kind, int(pk[1])), []).append([(r["E_J"][0] - ip * r["sec"]) / r["n"] * 1e3, r["E_J"][0] / r["n"] * 1e3,
                                                    r["sec"] / r["n"] * 1e3, r["lat_p50_ms"], r["lat_p99_ms"], ip])
    return {k: np.array(v) for k, v in B.items()}          # cols: D, G, T, p50, p99, idle


def policies(net, seed):
    """re-derive procedure A (proc_A_s15.py) for n=2000, eta=0.05; return per alpha list of (pi[4], acc_eval, kind)."""
    ld = NETS[net]
    def load(sp):
        Z = [np.load(f"{ld}/{net}_s{seed}_{sp}_e{k}.npy").astype(np.float64) for k in range(1, 5)]; y = np.load(f"{ld}/y_{sp}.npy")
        conf = np.stack([(lambda e: (e / e.sum(1, keepdims=True)).max(1))(np.exp(z - z.max(1, keepdims=True))) for z in Z], 1)
        return conf, np.stack([z.argmax(1) == y for z in Z], 1)
    cH, kH = load("holdout"); cT, kT = load("test"); conf = np.concatenate([cH, cT]); corr = np.concatenate([kH, kT]); N = len(conf)
    fl = json.load(open(f"{OD}/{net}_s{seed}/export_report.json"))["flops"]
    Fpath = np.array([sum(fl["seg"][:k]) for k in (1, 2, 3, 4)]) / 1e9; Ftr = np.array(fl["trunc"]) / 1e9
    TH = [("chain", E, float(l)) for r in (1, 2, 3) for E in itertools.combinations((0, 1, 2), r) for l in LAMS] + [("trunc", j, None) for j in range(3)]
    PATH = np.zeros((len(TH), N), np.int8); CH = {}
    for i, (kind, E, lam) in enumerate(TH):
        if kind == "chain":
            ex = np.full(N, 3)
            for j in sorted(E, reverse=True):
                ex = np.where(conf[:, j] >= lam, j, ex)
            PATH[i] = ex; CH.setdefault(("chain", E), []).append(i)
        else:
            PATH[i] = E; CH.setdefault(("trunc", E), []).append(i)
    LOSS = (~corr[np.arange(N)[None, :], PATH] & corr[None, :, 3]).astype(np.float64)
    ONEH = np.stack([(PATH == k) for k in range(4)], 2).astype(np.float64)
    CORP = corr[np.arange(N)[None, :], PATH].astype(np.float64)
    C = {}
    for r in csv.DictReader(open(f"{RES}/cost/sum_{net}_s9201.csv")):
        if r["prec"] == "fp16" and not r["extra"]:
            C[(r["kind"], int(r["k"]))] = np.array([float(v) for v in r["per_rep_net"].split(";")])
    KEYS = [("full", 4)] + [("chain", k) for k in (1, 2, 3, 4)] + [("trunc", k) for k in (1, 2, 3)]; cbar = {k: C[k].mean() for k in KEYS}
    lut = lambda cm: np.array([[cm[("chain", k)] for k in (1, 2, 3, 4)] if kd == "chain" else [cm[("trunc", E + 1)]] * 4 for kd, E, _ in TH])
    nc = min(len(C[k]) for k in KEYS); tq = tdist.ppf(1 - DC / len(KEYS), nc - 1)
    U = {k: max(C[k].mean() + tq * C[k].std(ddof=1) / np.sqrt(len(C[k])) + ETA * cbar[k], 0.0) for k in KEYS}
    kf = ("full", 4); Lf = C[kf].mean() - tq * C[kf].std(ddof=1) / np.sqrt(len(C[kf])) - ETA * cbar[kf]; LU = lut(U); Rm = max(U.values())
    FL = np.array([Fpath if k == "chain" else np.full(4, Ftr[E]) for k, E, _ in TH]); dq = DPI / len(TH); lvl = DR / len(CH)
    TOTC = ONEH.sum(1); TOTA = CORP.sum(1); out = {a: [] for a in ALPHAS}; n = NCAL
    for i in range(NSPLIT):
        cal = np.random.default_rng(9301 + i).permutation(N)[:n]; ne = N - n
        cc = ONEH[:, cal, :].sum(1); ls = LOSS[:, cal].sum(1); fl_cal = (cc * FL).sum(1) / n; ce = TOTC - cc; ae = (TOTA - CORP[:, cal].sum(1)) / ne
        for a in ALPHAS:
            cert = []
            for ids in CH.values():
                for t in ids:
                    k_ = int(round(ls[t]))
                    if (1.0 if k_ >= n else float(betad.ppf(1 - lvl, k_ + 1, n - k_))) <= a:
                        cert.append(t)
                    else:
                        break
            cert = np.array(cert, int); kind = "full"; t = None
            if len(cert):
                m = (cc[cert] * LU[cert]).sum(1) / n; m2 = (cc[cert] * LU[cert] ** 2).sum(1) / n; var = np.maximum(m2 - m * m, 0) * n / (n - 1)
                ub = m + np.sqrt(2 * var * np.log(2 / dq) / n) + 7 * Rm * np.log(2 / dq) / (3 * (n - 1)); sLB = Lf - ub; b = int(np.argmax(sLB))
                if sLB[b] > 0:
                    t, kind = int(cert[b]), "A"
                else:
                    t, kind = int(cert[int(np.argmin(fl_cal[cert]))]), "flops"
            if t is None:
                out[a].append((np.array([0, 0, 0, 1.0]), float(corr[np.setdiff1d(np.arange(N), cal), 3].mean()), "full", "full"))
            else:
                kd, E, lam = TH[t]
                pi = ce[t] / ne
                out[a].append((pi if kd == "chain" else np.eye(4)[E], float(ae[t]), kind, "trunc" if kd == "trunc" else "chain"))
    acc_exit = corr.mean(0)
    return out, acc_exit


def sim_streams(nstr, tau, pi, p50, p99, rng, horizon=10.0):
    ph = rng.uniform(0, tau, nstr); m = int(horizon / tau)
    A = np.sort((ph[:, None] + tau * np.arange(m)[None, :]).ravel()); K = len(A)
    path = rng.choice(4, K, p=pi / pi.sum()); sg = np.log(p99 / p50) / 2.326
    S = p50[path] * np.exp(sg[path] * rng.standard_normal(K)) / 1e3
    Cs = np.cumsum(S); Cp = np.concatenate([[0], Cs[:-1]]); D = Cs + np.maximum.accumulate(A - Cp)
    return float(((D - A) > tau).mean())


def max_streams(tau, pi, p50, p99, rng):
    n = 0
    while n < 400 and sim_streams(n + 1, tau, pi, p50, p99, rng) <= 0.01:
        n += 1
    return n


def sim_poisson(lam, S_draw, dl, rng, K=20000):
    A = np.cumsum(rng.exponential(1 / lam, K)); S = S_draw(K)
    Cs = np.cumsum(S); Cp = np.concatenate([[0], Cs[:-1]]); D = Cs + np.maximum.accumulate(A - Cp); Rt = D - A
    return float(np.percentile(Rt, 99)), float((Rt > dl).mean())


res = dict(note="EXPLORATORY post-hoc analysis on S1.5 measurements (25-11 wave, seed-9201 costs); not preregistered", nets={})
for net in NETS:
    Bt = batches(net); Ps = float(np.mean([v[:, 5].mean() for v in Bt.values()]))
    P = {a: [] for a in ALPHAS}; acc_ex = []
    for s in SEEDS:
        o, ae = policies(net, s); acc_ex.append(ae)
        for a in ALPHAS:
            P[a] += o[a]
    acc_ex = np.mean(acc_ex, 0); r = dict(P_s_W=Ps, acc_exit=acc_ex.tolist()); rng0 = np.random.default_rng(9401)
    ck = [("chain", k) for k in (1, 2, 3, 4)]; full = ("full", 4)
    mean = lambda key, c: Bt[key][:, c].mean()
    for a in ALPHAS:
        PI = np.array([p[0] for p in P[a]]); kinds = [p[2] for p in P[a]]; ACC = np.array([p[1] for p in P[a]])
        def stat(pi, bd):                                   # bd: dict key -> per-config means (D, G, T)
            Df, Gf, Tf = bd[full]
            Dp = sum(pi[k] * bd[ck[k]][0] for k in range(4)); Gp = sum(pi[k] * bd[ck[k]][1] for k in range(4)); Tp = sum(pi[k] * bd[ck[k]][2] for k in range(4))
            sn = 1 - Dp / Df; sg = 1 - Gp / Gf; lam = {f: f for f in FPS}; lam["sat"] = 1000 / Tf
            phi = {f: l * Df / 1000 / (Ps + l * Df / 1000) for f, l in lam.items()}
            return dict(sigma_net=sn, sigma_gross=sg, thr_gain=Tf / Tp - 1, phi=phi, sigma_sys={f: phi[f] * sn for f in phi})
        # trunc-policy rows (FLOPs selector may pick a trunc engine): cost with trunc engine instead of chain
        def bd_for(i, rb):
            return rb
        rbm = {k: Bt[k][:, :3].mean(0) for k in Bt}
        def stat_split(j, rb):
            pi = PI[j]
            if P[a][j][3] in ("trunc", "full"):            # static trunc/full engine policy: path k via trunc_k (not chain)
                k = int(np.argmax(pi)); rb2 = dict(rb); rb2[ck[k]] = rb[("trunc", k + 1)] if k < 3 else rb[full]; return stat(pi, rb2)
            return stat(pi, rb)
        pts = [stat_split(j, rbm) for j in range(len(PI))]
        point = {k: float(np.mean([p[k] for p in pts])) for k in ("sigma_net", "sigma_gross", "thr_gain")}
        point["phi"] = {str(f): float(np.mean([p["phi"][f] for p in pts])) for f in pts[0]["phi"]}
        point["sigma_sys"] = {str(f): float(np.mean([p["sigma_sys"][f] for p in pts])) for f in pts[0]["sigma_sys"]}
        rng = np.random.default_rng(9402); bs = []
        for b in range(1000):
            j = int(rng.integers(len(PI))); rb = {k: v[rng.integers(0, len(v), len(v)), :3].mean(0) for k, v in Bt.items()}; bs.append(stat_split(j, rb))
        ci = lambda f: [float(np.percentile([f(x) for x in bs], 2.5)), float(np.percentile([f(x) for x in bs], 97.5))]
        q = dict(policy_kind={k: kinds.count(k) / len(kinds) for k in set(kinds)}, pi_mean=PI.mean(0).tolist(), acc_eval=float(ACC.mean()), point=point,
                 ci95=dict(sigma_net=ci(lambda x: x["sigma_net"]), sigma_gross=ci(lambda x: x["sigma_gross"]), thr_gain=ci(lambda x: x["thr_gain"]),
                           phi={str(f): ci(lambda x, f=f: x["phi"][f]) for f in pts[0]["phi"]}, sigma_sys={str(f): ci(lambda x, f=f: x["sigma_sys"][f]) for f in pts[0]["sigma_sys"]}))
        sn, sg = q["ci95"]["sigma_net"], q["ci95"]["sigma_gross"]
        q["reversal"] = bool(np.sign(point["sigma_net"]) != np.sign(point["sigma_gross"])); q["reversal_ci_excl0_both"] = bool(q["reversal"] and (sn[0] > 0 or sn[1] < 0) and (sg[0] > 0 or sg[1] < 0))
        # C3 streams + C4 queue (R = 200 replicates)
        rngq = np.random.default_rng(9403); NS = {30: [], 10: []}; C4 = {rho: [] for rho in (0.5, 0.7, 0.9, 0.95)}
        trunc_T = {k: Bt[("trunc", k)][:, 2].mean() for k in (1, 2, 3)}
        for rep in range(200):
            j = int(rngq.integers(len(PI))); pi = PI[j]; ib = {k: rngq.integers(0, len(v), len(v)) for k, v in Bt.items()}
            keyset = ck if P[a][j][3] == "chain" else [("trunc", k) if k < 4 else full for k in (1, 2, 3, 4)]
            p50 = np.array([Bt[key][ib[key], 3].mean() for key in keyset]); p99 = np.array([Bt[key][ib[key], 4].mean() for key in keyset])
            f50 = np.full(4, Bt[full][ib[full], 3].mean()); f99 = np.full(4, Bt[full][ib[full], 4].mean()); e4 = np.array([0, 0, 0, 1.0])
            for fps in NS:
                NS[fps].append((max_streams(1 / fps, pi, p50, p99, rngq), max_streams(1 / fps, e4, f50, f99, rngq)))
            ES = float((pi * p50 * np.exp((np.log(p99 / p50) / 2.326) ** 2 / 2)).sum()) / 1e3
            kst = min(trunc_T, key=lambda k: abs(trunc_T[k] / 1e3 - ES)); s50 = Bt[("trunc", kst)][ib[("trunc", kst)], 3].mean(); s99 = Bt[("trunc", kst)][ib[("trunc", kst)], 4].mean()
            sgp = np.log(p99 / p50) / 2.326; sgs = np.log(s99 / s50) / 2.326
            dyn = lambda K: p50[(pth := rngq.choice(4, K, p=pi / pi.sum()))] * np.exp(sgp[pth] * rngq.standard_normal(K)) / 1e3
            sta = lambda K: s50 * np.exp(sgs * rngq.standard_normal(K)) / 1e3
            for rho in C4:
                lam = rho / ES; dl = 5 * ES
                C4[rho].append((sim_poisson(lam, dyn, dl, rngq), sim_poisson(lam, sta, dl, rngq), kst))
        q["C3_streams"] = {}
        for fps, v in NS.items():
            v = np.array(v, float); g = v[:, 0] / np.maximum(v[:, 1], 1) - 1
            q["C3_streams"][str(fps)] = dict(policy_median=float(np.median(v[:, 0])), full_median=float(np.median(v[:, 1])), policy_ci=[float(np.percentile(v[:, 0], 2.5)), float(np.percentile(v[:, 0], 97.5))],
                                             full_ci=[float(np.percentile(v[:, 1], 2.5)), float(np.percentile(v[:, 1], 97.5))], gain_median=float(np.median(g)), gain_ci=[float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))])
        q["C4"] = {}
        for rho, v in C4.items():
            dp99 = np.array([x[0][0] for x in v]); dm = np.array([x[0][1] for x in v]); sp99 = np.array([x[1][0] for x in v]); sm = np.array([x[1][1] for x in v])
            ks = [x[2] for x in v]; kmode = max(set(ks), key=ks.count)
            q["C4"][str(rho)] = dict(static_trunc=int(kmode), static_acc=float(acc_ex[kmode - 1]), dyn_acc=float(ACC.mean()),
                                     dyn_p99_ms=float(1e3 * dp99.mean()), static_p99_ms=float(1e3 * sp99.mean()), dyn_miss=float(dm.mean()), static_miss=float(sm.mean()),
                                     miss_diff_ci=[float(np.percentile(dm - sm, 2.5)), float(np.percentile(dm - sm, 97.5))], static_wins_miss=bool(np.percentile(dm - sm, 2.5) > 0))
        r[f"alpha_{a}"] = q
        print(net, a, q["policy_kind"], "sn", round(point["sigma_net"], 4), sn, "sg", round(point["sigma_gross"], 4), sg, "rev", q["reversal"], q["reversal_ci_excl0_both"],
              "thr", round(point["thr_gain"], 4), q["ci95"]["thr_gain"], flush=True)
    res["nets"][net] = r
    json.dump(res, open(f"{OUT}/a8_regime.json", "w"), indent=1, default=float)
print("A8_DONE")
