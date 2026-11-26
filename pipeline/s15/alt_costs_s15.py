"""S1.5 addendum (secondary, not for scoring): per-batch cost csvs for Procedure A under sensitivity variants
a (sandwich idle), b (gross energy), c (latency p50), in the column format proc_A_s15.py reads (prec, kind, k, extra, per_rep_net).
Usage: python alt_costs_s15.py <measure_raw.json> <out_prefix>   -> <out_prefix>_{a,b,c}.csv
"""
import sys, csv, importlib.util, os
spec = importlib.util.spec_from_file_location("sens", os.path.join(os.path.dirname(__file__), "sens_s15.py"))
src = open(spec.origin).read().split("Dfun = ")[0]            # reuse load() without running the main block
ns = {}; exec(compile(src.replace("root, outp = sys.argv[1:3]; ", ""), "sens", "exec"), ns)
M = ns["load"](sys.argv[1]); outp = sys.argv[2]
for m in ("a", "b", "c"):
    with open(f"{outp}_{m}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prec", "kind", "k", "extra", "per_rep_net"])
        for key, v in M[m].items():
            w.writerow(["fp16", key[:-1], key[-1], "", ";".join(f"{x:.6f}" for x in v)])
print("ALT_DONE", outp)
