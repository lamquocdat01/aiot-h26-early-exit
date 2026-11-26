"""H26 S3: generate numbers.tex / numbers.json for the manuscript from result JSON/CSV.

Every number printed in the paper comes from \\nv{key}; this script is the only place
where values are read and rounded.  Usage:
    python make_numbers.py <H26 folder> <out dir>
"""
import csv
import glob
import json
import os
import statistics as st
import sys

ROOT = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
R = os.path.join(ROOT, "results")
NETS = ["n1", "n2", "n3", "n4", "n5"]
NAME_ = {"n1": "N1", "n2": "N2", "n3": "N3", "n4": "N4", "n5": "N5"}
vals = {}
src = {}


def put(key, v, fmt, source):
    if v is None:
        s = "--"
    elif fmt == "int":
        s = str(int(round(v)))
    else:
        s = format(v, fmt)
    vals[key] = s
    src[key] = {"value": v, "source": source}


def J(p):
    with open(os.path.join(R, p), encoding="utf-8") as f:
        return json.load(f)


# ---------- H1 / H2 (cost) ----------
h12 = J("s15/cost/h1h2.json")
sens = J("s15/sens/sens_s15.json")["nets"]
for n in NETS:
    d = h12[n]
    put(f"{n}.R", d["R"], ".3f", "s15/cost/h1h2.json")
    put(f"{n}.Rlo90", d["R_ci90"][0], ".3f", "s15/cost/h1h2.json")
    put(f"{n}.Rhi90", d["R_ci90"][1], ".3f", "s15/cost/h1h2.json")
    put(f"{n}.D", 100 * d["D"], ".2f", "s15/cost/h1h2.json")
    m = sens[n]["main"]
    put(f"{n}.Dlo95", 100 * m["H2_D"]["ci95"][0], ".2f", "s15/sens/sens_s15.json")
    put(f"{n}.Dhi95", 100 * m["H2_D"]["ci95"][1], ".2f", "s15/sens/sens_s15.json")
    put(f"{n}.Dlo90", 100 * d["D_ci90"][0], ".2f", "s15/cost/h1h2.json")
    put(f"{n}.Dhi90", 100 * d["D_ci90"][1], ".2f", "s15/cost/h1h2.json")
    put(f"{n}.cfull", d["cbar"]["full4"], ".2f", "s15/cost/h1h2.json")
    for k in (1, 2, 3):
        put(f"{n}.ctrunc{k}", d["cbar"][f"trunc{k}"], ".2f", "s15/cost/h1h2.json")
    for k in (1, 2, 3, 4):
        put(f"{n}.cchain{k}", d["cbar"][f"chain{k}"], ".2f", "s15/cost/h1h2.json")
        put(f"{n}.cond{k}", d["cbar"][f"cond{k}"], ".2f", "s15/cost/h1h2.json")
        key = f"H3_chain{k}_over_full" if k < 4 else None
        if key and key in m:
            put(f"{n}.ratio{k}", m[key]["mean"], ".3f", "s15/sens/sens_s15.json")
    put(f"{n}.ratio4", d["R"], ".3f", "s15/cost/h1h2.json")
    put(f"{n}.Rs9202", d["R_s9202"], ".3f", "s15/cost/h1h2.json")
    for alt in ("a", "b", "c"):
        put(f"{n}.R.{alt}", sens[n][alt]["H1_R"]["mean"], ".3f", "s15/sens/sens_s15.json")
        put(f"{n}.D.{alt}", 100 * sens[n][alt]["H2_D"]["mean"], ".2f", "s15/sens/sens_s15.json")
for n in ("n2", "n4"):
    u = sens[f"{n}_unpinned"]["main"]
    put(f"{n}u.R", u["H1_R"]["mean"], ".3f", "s15/sens/sens_s15.json")
    put(f"{n}u.Rlo90", u["H1_R"]["ci90"][0], ".3f", "s15/sens/sens_s15.json")
    put(f"{n}u.Rhi90", u["H1_R"]["ci90"][1], ".3f", "s15/sens/sens_s15.json")
    put(f"{n}u.D", 100 * u["H2_D"]["mean"], ".2f", "s15/sens/sens_s15.json")
with open(os.path.join(R, "s15/cost/n2_unpinned/sum_n2_s9201.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["cfg"] == "fp16_full_p4":
            put("n2u.cfull", float(r["mJ_net"]), ".1f", "s15/cost/n2_unpinned/sum_n2_s9201.csv")
# N1 check-2 deviations
dev = h12["n1"]["check2_dev_pct"]
put("n1.chk.full", dev["full4"], "+.1f", "s15/cost/h1h2.json")
put("n1.chk.chain1", dev["chain1"], "+.1f", "s15/cost/h1h2.json")
put("n1.chk.chain4", dev["chain4"], "+.1f", "s15/cost/h1h2.json")
maxdev_other = max(abs(v) for n in ("n2", "n3", "n4", "n5") for v in h12[n]["check2_dev_pct"].values())
put("chk.maxother", maxdev_other, ".1f", "s15/cost/h1h2.json")

# ---------- H3 / H4 (procedure A) ----------
h34 = J("s15/procA_b/h3h4.json")
cells = h34["pooled"]["cells"]
put("H4.cells", len(cells), "int", "s15/procA_b/h3h4.json")
put("H4.ncellpass", sum(1 for c in cells.values() if c["viol_r"] <= 0.1 and c["viol_s"] <= 0.1), "int", "s15/procA_b/h3h4.json")
put("H4.maxviolr", max(c["viol_r"] for c in cells.values()), ".3f", "s15/procA_b/h3h4.json")
put("H4.maxviols", max(c["viol_s"] for c in cells.values()), ".3f", "s15/procA_b/h3h4.json")
put("H4.splits", list(cells.values())[0]["splits"], "int", "s15/procA_b/h3h4.json")
put("H3.seedpass", h34["per_seed_counts"]["H3_pass"], "int", "s15/procA_b/h3h4.json")
put("H4.seedpass", h34["per_seed_counts"]["H4_pass"], "int", "s15/procA_b/h3h4.json")
rows = []
for f in glob.glob(os.path.join(R, "s15/procA_b/procA_n*_s*.csv")):
    with open(f, encoding="utf-8") as fh:
        rows += list(csv.DictReader(fh))
alphas = {"0.01": "a1", "0.02": "a2", "0.05": "a5"}
slb_all = []
for n in NETS:
    for a, tag in alphas.items():
        sel = [r for r in rows if r["net"] == n and r["eta"] == "0.05" and r["n"] == "2000" and r["alpha"] == a]

        def mean(k, scale=1.0):
            v = [float(r[k]) for r in sel if r[k] not in ("", None)]
            return scale * st.mean(v) if v else None
        cell = cells[f"{n}|n2000|a{a}"]
        put(f"{n}.{tag}.out", 100 * cell["out_rate"], ".1f", "s15/procA_b/h3h4.json")
        sl = mean("sLB_pct")
        put(f"{n}.{tag}.slb", sl, ".1f", "s15/procA_b/procA_*.csv (eta 0.05, n 2000, mean of 5 seeds)")
        if sl is not None and n in ("n3", "n5"):
            slb_all.append(sl)
        put(f"{n}.{tag}.seval", mean("s_eval_pct"), ".1f", "s15/procA_b/procA_*.csv")
        put(f"{n}.{tag}.fs", mean("flops_s_eval_pct"), "+.1f", "s15/procA_b/procA_*.csv")
        put(f"{n}.{tag}.floss", 100 * cell["flops_loss_rate"], ".1f", "s15/procA_b/h3h4.json")
        put(f"{n}.{tag}.violr", cell["viol_r"], ".3f", "s15/procA_b/h3h4.json")
        put(f"{n}.{tag}.reval", mean("r_eval_mean", 100), ".2f", "s15/procA_b/procA_*.csv")
put("slb.min", min(slb_all), ".1f", "procA csv")
put("slb.max", max(slb_all), ".1f", "procA csv")

# accuracy (exit 4, test) and FP16 flips
for n in NETS:
    acc = []
    for f in sorted(glob.glob(os.path.join(R, f"s15/procA_b/procA_{n}_s*.json"))):
        with open(f, encoding="utf-8") as fh:
            acc.append(json.load(fh)["acc_test"][3])
    put(f"{n}.acclo", 100 * min(acc), ".1f", "s15/procA_b/procA_*.json acc_test[3] (Orin FP16 logits)")
    put(f"{n}.acchi", 100 * max(acc), ".1f", "s15/procA_b/procA_*.json acc_test[3]")
    fl = []
    for f in glob.glob(os.path.join(R, f"s15/flip/flip_{n}_s*.json")):
        with open(f, encoding="utf-8") as fh:
            fl += json.load(fh)["flip_test"]
    put(f"{n}.flipmax", 100 * max(fl), ".2f", "s15/flip/flip_*.json")

# ---------- mechanism ----------
md = J("s15/mech/mech_decomp.json")
for n in NETS:
    m = J(f"s15/mech/mech_{n}.json")
    ba = m["B_minus_A_ms"]
    put(f"{n}.BA", ba, ".2f", f"s15/mech/mech_{n}.json")
    put(f"{n}.host", 100 * m["B_minus_C_sync_d2h_check_ms"] / ba, ".0f", f"s15/mech/mech_{n}.json")
    put(f"{n}.gpu", 100 * m["C_minus_A_gpu_side_ms"] / ba, ".0f", f"s15/mech/mech_{n}.json")
    put(f"{n}.tfull", m["A_full"]["p50_ms"], ".2f", f"s15/mech/mech_{n}.json")
    put(f"{n}.latratio", m["B_chain4"]["p50_ms"] / m["A_full"]["p50_ms"], ".2f", f"s15/mech/mech_{n}.json")
    if n in md:
        g = md[n]["extra_gpu_ms"]
        put(f"{n}.reform", 100 * md[n]["extra_reformat"] / g, ".0f", "s15/mech/mech_decomp.json")
        put(f"{n}.heads", 100 * md[n]["extra_heads"] / g, ".0f", "s15/mech/mech_decomp.json")
hosts = [100 * J(f"s15/mech/mech_{n}.json")["B_minus_C_sync_d2h_check_ms"] / J(f"s15/mech/mech_{n}.json")["B_minus_A_ms"] for n in NETS]
put("host.min", min(hosts), ".0f", "s15/mech/mech_*.json")
put("host.max", max(hosts), ".0f", "s15/mech/mech_*.json")
bas = [J(f"s15/mech/mech_{n}.json")["B_minus_A_ms"] for n in NETS]
put("BA.min", min(bas), ".1f", "s15/mech/mech_*.json")
put("BA.max", max(bas), ".1f", "s15/mech/mech_*.json")

# ---------- post-hoc regime analysis (A8, exploratory) ----------
a8 = J("a8/a8_regime.json")["nets"]
ps = [a8[n]["P_s_W"] for n in NETS]
gaps = [100 * (a8[n][f"alpha_{a}"]["point"]["sigma_net"] - a8[n][f"alpha_{a}"]["point"]["sigma_gross"])
        for n, a in [("n3", "0.01"), ("n3", "0.02"), ("n3", "0.05"), ("n5", "0.01"), ("n5", "0.02"), ("n5", "0.05"), ("n2", "0.05")]]
put("gap.min", min(gaps), ".0f", "a8/a8_regime.json (net minus gross, certified policies)")
put("gap.max", max(gaps), ".0f", "a8/a8_regime.json")
put("Ps.min", min(ps), ".2f", "a8/a8_regime.json")
put("Ps.max", max(ps), ".2f", "a8/a8_regime.json")
for n in NETS:
    for a, tag in alphas.items():
        c = a8[n][f"alpha_{a}"]
        put(f"{n}.{tag}.snet", 100 * c["point"]["sigma_net"], "+.1f", "a8/a8_regime.json")
        put(f"{n}.{tag}.sgross", 100 * c["point"]["sigma_gross"], "+.1f", "a8/a8_regime.json")
        put(f"{n}.{tag}.thr", 100 * c["point"]["thr_gain"], "+.1f", "a8/a8_regime.json")
        put(f"{n}.{tag}.thrlo", 100 * c["ci95"]["thr_gain"][0], "+.1f", "a8/a8_regime.json")
        put(f"{n}.{tag}.thrhi", 100 * c["ci95"]["thr_gain"][1], "+.1f", "a8/a8_regime.json")
        put(f"{n}.{tag}.sgrosslo", 100 * c["ci95"]["sigma_gross"][0], "+.1f", "a8/a8_regime.json")
        put(f"{n}.{tag}.sgrosshi", 100 * c["ci95"]["sigma_gross"][1], "+.1f", "a8/a8_regime.json")
        put(f"{n}.{tag}.snetlo", 100 * c["ci95"]["sigma_net"][0], "+.1f", "a8/a8_regime.json")
        put(f"{n}.{tag}.snethi", 100 * c["ci95"]["sigma_net"][1], "+.1f", "a8/a8_regime.json")
        s30 = c["C3_streams"]["30"]
        put(f"{n}.{tag}.st30p", s30["policy_median"], "int", "a8/a8_regime.json")
        put(f"{n}.{tag}.st30f", s30["full_median"], "int", "a8/a8_regime.json")

# ---------- conservatism per network (all 9 cells) ----------
for n in NETS:
    cc = [c for k, c in cells.items() if k.startswith(n + "|")]
    put(f"{n}.violrmax", max(c["viol_r"] for c in cc), ".3f", "s15/procA_b/h3h4.json")
    put(f"{n}.violsmax", max(c["viol_s"] for c in cc), ".3f", "s15/procA_b/h3h4.json")

# ---------- certified vs realised saving by n (N3, N5), for Fig. 2 and text ----------
fig2 = {}
for n in ("n3", "n5"):
    for a, tag in alphas.items():
        for nn in ("500", "2000", "5000"):
            sel = [r for r in rows if r["net"] == n and r["eta"] == "0.05" and r["n"] == nn and r["alpha"] == a]
            sl = [float(r["sLB_pct"]) for r in sel if r["sLB_pct"]]
            se = [float(r["s_eval_pct"]) for r in sel if r["s_eval_pct"]]
            fig2[f"{n}|{a}|{nn}"] = (st.mean(sl) if sl else None, st.mean(se) if se else None)
            put(f"{n}.{tag}.n{nn}.slb", fig2[f"{n}|{a}|{nn}"][0], ".1f", "s15/procA_b/procA_*.csv")
            put(f"{n}.{tag}.n{nn}.seval", fig2[f"{n}|{a}|{nn}"][1], ".1f", "s15/procA_b/procA_*.csv")
            if sl and se:
                put(f"{n}.{tag}.n{nn}.gap", st.mean(se) - st.mean(sl), ".1f", "s15/procA_b/procA_*.csv (realised minus certified)")
gaps2 = [fig2[k][1] - fig2[k][0] for k in fig2 if fig2[k][0] is not None]
put("cgap.min", min(gaps2), ".1f", "s15/procA_b/procA_*.csv")
put("cgap.max", max(gaps2), ".1f", "s15/procA_b/procA_*.csv")
with open(os.path.join(OUT, "fig2_data.json"), "w", encoding="utf-8") as f:
    json.dump(fig2, f, indent=1)

# ---------- FP16: pinning N2 / N4 ----------
for n in ("n2", "n4"):
    pin, unp = [], []
    for f in glob.glob(os.path.join(R, f"s15/flip/flip_{n}_s*.json")):
        with open(f, encoding="utf-8") as fh:
            pin.append(json.load(fh)["flip_test"][0])
    for f in glob.glob(os.path.join(R, f"s15/flip/unpinned_{n}/flip_{n}_s9*.json")):
        if f.endswith("_engines.json"):
            continue
        with open(f, encoding="utf-8") as fh:
            unp.append(json.load(fh)["flip_test"][0])
    put(f"{n}.flipe1", 100 * max(pin), ".2f", f"s15/flip/flip_{n}_s*.json flip_test[0], max over seeds")
    put(f"{n}u.flipe1lo", 100 * min(unp), ".2f", f"s15/flip/unpinned_{n}/ flip_test[0], min over seeds")
    put(f"{n}u.flipe1", 100 * max(unp), ".2f", f"s15/flip/unpinned_{n}/ flip_test[0], max over seeds")
    put(f"{n}u.nseedfail", sum(1 for v in unp if v > 0.01), "int", f"s15/flip/unpinned_{n}/")
    with open(os.path.join(R, f"s15/cost/sum_{n}_s9201.csv"), encoding="utf-8") as fh:
        P = {r["cfg"]: r for r in csv.DictReader(fh)}
    with open(os.path.join(R, f"s15/cost/{n}_unpinned/sum_{n}_s9201.csv"), encoding="utf-8") as fh:
        U = {r["cfg"]: r for r in csv.DictReader(fh)}
    for cfg, t in (("fp16_full_p4", "full"), ("fp16_chain_p1", "chain1"), ("fp16_chain_p4", "chain4")):
        put(f"{n}.lat.{t}", float(P[cfg]["lat_p50_ms"]), ".2f", f"s15/cost/sum_{n}_s9201.csv")
        put(f"{n}u.lat.{t}", float(U[cfg]["lat_p50_ms"]), ".2f", f"s15/cost/{n}_unpinned/sum_{n}_s9201.csv")
        put(f"{n}.mj.{t}", float(P[cfg]["mJ_net"]), ".2f", f"s15/cost/sum_{n}_s9201.csv")
        put(f"{n}u.mj.{t}", float(U[cfg]["mJ_net"]), ".2f", f"s15/cost/{n}_unpinned/sum_{n}_s9201.csv")
    put(f"{n}.pinpct.full", 100 * (float(P["fp16_full_p4"]["mJ_net"]) / float(U["fp16_full_p4"]["mJ_net"]) - 1), ".0f",
        "pinned vs unpinned full engine, net energy")
b2 = J("s15/fp16_fix/bisect_n2_s9205_seg1.json")
for k, t in (("none", "none"), ("all", "all"), ("re:body\\.4", "body4"), ("re:/head/", "head")):
    put(f"n2.bis.{t}.flip", 100 * b2[k]["flip"], ".1f", "s15/fp16_fix/bisect_n2_s9205_seg1.json (1,000 images)")
    put(f"n2.bis.{t}.lat", b2[k]["lat_ms"], ".2f", "s15/fp16_fix/bisect_n2_s9205_seg1.json")
    put(f"n2.bis.{t}.np", b2[k]["pinned"], "int", "s15/fp16_fix/bisect_n2_s9205_seg1.json")

# ---------- two cost models, out of sample (predictions hashed before measurement) ----------
s3 = J("s14b_n1/score_n3_fp16.json")
put("m1.med", 100 * s3["cost_median"], ".0f", "s14b_n1/score_n3_fp16.json")
put("m1.max", 100 * s3["cost_max"], ".0f", "s14b_n1/score_n3_fp16.json")
put("m1.rho1.pred", s3["rho_pred"]["chain_1"], ".2f", "s14b_n1/score_n3_fp16.json")
put("m1.rho1.meas", s3["rho_meas"]["chain_1"], ".3f", "s14b_n1/score_n3_fp16.json")
f1 = J("s14b_n1/fit_fp16_n1n2.json")
res1 = [abs(r["rel_resid"]) for r in f1["fit_rows"]]
put("m1.in.med", 100 * st.median(res1), ".1f", "s14b_n1/fit_fp16_n1n2.json")
f2 = J("s14b_n2/cost_v2_fit.json")
res2 = [abs(r["rel"]) for r in f2["rows"]]
put("m2.in.med", 100 * st.median(res2), ".1f", "s14b_n2/cost_v2_fit.json")
for n in ("n4", "n5"):
    s = J(f"s14b_n2/score_{n}.json")
    put(f"m2.{n}.med", 100 * s["median_trunc_chain"], ".0f", f"s14b_n2/score_{n}.json")
    put(f"m2.{n}.max", 100 * s["max_trunc_chain"], ".0f", f"s14b_n2/score_{n}.json")
    for k in (1, 2, 3):
        put(f"m2.{n}.rho{k}.pred", s["rho"][f"chain_{k}"]["pred"], ".2f", f"s14b_n2/score_{n}.json")
        put(f"m2.{n}.rho{k}.meas", s["rho"][f"chain_{k}"]["meas"], ".2f", f"s14b_n2/score_{n}.json")
        put(f"m2.{n}.rho{k}.rel", 100 * s["rho"][f"chain_{k}"]["rel"], ".0f", f"s14b_n2/score_{n}.json")

# ---------- generated tables ----------
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "tab_paths.tex"), "w", encoding="utf-8") as f:
    f.write("% AUTO-GENERATED from results/s15/cost/sum_n?_s9201.csv (seed 9201, 10 batches)\n")
    CFG = [("fp16_full_p4", "full"), ("fp16_chain_p1", "chain 1"), ("fp16_chain_p2", "chain 2"),
           ("fp16_chain_p3", "chain 3"), ("fp16_chain_p4", "chain 4"), ("fp16_trunc_p1", "trunc 1"),
           ("fp16_trunc_p2", "trunc 2"), ("fp16_trunc_p3", "trunc 3"), ("fp16_cond_p1", "If, path 1"),
           ("fp16_cond_p4", "If, path 4")]
    for n in NETS:
        with open(os.path.join(R, f"s15/cost/sum_{n}_s9201.csv"), encoding="utf-8") as fh:
            P = {r["cfg"]: r for r in csv.DictReader(fh)}
        for i, (cfg, lab) in enumerate(CFG):
            r = P[cfg]
            first = f"\\multirow{{{len(CFG)}}}{{*}}{{{NAME_[n]}}}" if i == 0 else ""
            f.write(f"{first} & {lab} & {float(r['mJ_net']):.2f} & [{float(r['mJ_net_lo']):.2f}, {float(r['mJ_net_hi']):.2f}]"
                    f" & {float(r['mJ_tot']):.2f} & {float(r['lat_p50_ms']):.2f} & {float(r['lat_p99_ms']):.2f} \\\\\n")
        if n != NETS[-1]:
            f.write("\\midrule\n")
with open(os.path.join(OUT, "tab_cap.tex"), "w", encoding="utf-8") as f:
    f.write("% AUTO-GENERATED from results/a8/a8_regime.json (exploratory, alpha = 0.05)\n")
    for n in NETS:
        c = a8[n]["alpha_0.05"]
        kind = "A" if c["policy_kind"].get("A", 0) >= 0.5 else "FLOPs"
        s10, s30 = c["C3_streams"]["10"], c["C3_streams"]["30"]
        f.write(f"{NAME_[n]} & {kind} & {100*c['point']['sigma_net']:+.1f} & {100*c['point']['sigma_gross']:+.1f} & "
                f"{100*c['point']['thr_gain']:+.1f} [{100*c['ci95']['thr_gain'][0]:+.1f}, {100*c['ci95']['thr_gain'][1]:+.1f}] & "
                f"{int(s10['full_median'])} / {int(s10['policy_median'])} & {int(s30['full_median'])} / {int(s30['policy_median'])} \\\\\n")
with open(os.path.join(OUT, "tab_seed.tex"), "w", encoding="utf-8") as f:
    f.write("% AUTO-GENERATED from results/s15/procA_b/procA_n?_s????.csv (eta 0.05, n 2000, 500 splits per seed)\n")
    for n in NETS:
        seeds = sorted({r["seed"] for r in rows if r["net"] == n})
        for i, sd in enumerate(seeds):
            parts = []
            for a in ("0.01", "0.02", "0.05"):
                r = [x for x in rows if x["net"] == n and x["seed"] == sd and x["eta"] == "0.05" and x["n"] == "2000" and x["alpha"] == a][0]
                sl = f"{float(r['sLB_pct']):.1f}" if r["sLB_pct"] else "--"
                se = f"{float(r['s_eval_pct']):.1f}" if r["s_eval_pct"] else "--"
                parts.append(f"{100*float(r['out_rate']):.1f} & {sl} & {se}")
            first = f"\\multirow{{{len(seeds)}}}{{*}}{{{NAME_[n]}}}" if i == 0 else ""
            f.write(f"{first} & {sd} & " + " & ".join(parts) + " \\\\\n")
        if n != NETS[-1]:
            f.write("\\midrule\n")
NAME = NAME_
with open(os.path.join(OUT, "tab_h4.tex"), "w", encoding="utf-8") as f:
    f.write("% AUTO-GENERATED from results/s15/procA_b/h3h4.json\n")
    for n in NETS:
        for nn in ("500", "2000", "5000"):
            parts = []
            for a in ("0.01", "0.02", "0.05"):
                c = cells[f"{n}|n{nn}|a{a}"]
                parts.append(f"{100*c['out_rate']:.1f} & {c['viol_r']:.4f} [{c['viol_r_ci'][1]:.4f}] & {c['viol_s']:.4f}")
            f.write(f"{NAME[n]} & {nn} & " + " & ".join(parts) + " \\\\\n")
with open(os.path.join(OUT, "tab_cost.tex"), "w", encoding="utf-8") as f:
    f.write("% AUTO-GENERATED from results/s15/cost/h1h2.json (net mJ per inference, seed 9201, 10 batches)\n")
    for n in NETS:
        cb = h12[n]["cbar"]
        cols = ["full4", "chain1", "chain2", "chain3", "chain4", "trunc1", "trunc2", "trunc3", "cond1", "cond4"]
        f.write(NAME[n] + " & " + " & ".join(f"{cb[c]:.2f}" for c in cols) + " \\\\\n")

# ---------- write ----------
with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as f:
    json.dump({"values": vals, "provenance": src}, f, indent=1, ensure_ascii=False)
with open(os.path.join(OUT, "numbers.tex"), "w", encoding="utf-8") as f:
    f.write("% AUTO-GENERATED by aiot-h26/s3/make_numbers.py -- do not edit by hand\n")
    f.write("\\makeatletter\n\\newcommand{\\nv}[1]{\\ifcsname nv@#1\\endcsname\\csname nv@#1\\endcsname"
            "\\else\\textbf{??#1}\\PackageWarning{numbers}{undefined number #1}\\fi}\n")
    for k, v in sorted(vals.items()):
        v = v.replace("-", "\\textminus{}") if v.startswith("-") else v
        f.write(f"\\expandafter\\def\\csname nv@{k}\\endcsname{{{v}}}\n")
    f.write("\\makeatother\n")
print(len(vals), "numbers written to", OUT)
