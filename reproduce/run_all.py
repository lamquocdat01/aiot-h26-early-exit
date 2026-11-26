"""Regenerate all article numbers/tables/figure from results/ and compare with reference/."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
subprocess.run([sys.executable, os.path.join(HERE, "make_numbers.py"), ROOT, OUT], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "make_fig.py"), ROOT, os.path.join(OUT, "fig_ratios.pdf")], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "make_fig2.py"), os.path.join(OUT, "fig2_data.json"), os.path.join(OUT, "fig_cert.pdf")], check=True)

new = json.load(open(os.path.join(OUT, "numbers.json"), encoding="utf-8"))["values"]
ref = json.load(open(os.path.join(ROOT, "reference", "numbers.json"), encoding="utf-8"))["values"]
bad = [k for k in ref if new.get(k) != ref[k]] + [k for k in new if k not in ref]
for f in ("tab_h4.tex", "tab_cost.tex", "tab_paths.tex", "tab_cap.tex", "tab_seed.tex", "numbers.tex"):
    a = open(os.path.join(OUT, f), encoding="utf-8").read()
    b = open(os.path.join(ROOT, "reference", f), encoding="utf-8").read()
    if a != b:
        bad.append(f)
if bad:
    print("MISMATCH:", bad[:20])
    sys.exit(1)
print(f"REPRODUCED: {len(ref)} numbers, 5 tables, numbers.tex identical; figures written to out/fig_ratios.pdf, out/fig_cert.pdf")
