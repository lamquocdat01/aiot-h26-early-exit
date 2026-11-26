"""H26 S3: Figure 2 (certified vs realised net-energy saving by calibration size, N3 and N5).
Input: out/fig2_data.json written by make_numbers.py.  Usage: python make_fig2.py <fig2_data.json> <out.pdf>
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(sys.argv[1], encoding="utf-8"))
ns = [500, 2000, 5000]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
col = {"0.01": "#1b7837", "0.02": "#762a83", "0.05": "#e08214"}
for ax, (n, lab) in zip(axes, [("n3", "N3 ViT-B/16"), ("n5", "N5 DeiT-S")]):
    for a in ("0.01", "0.02", "0.05"):
        sl = [d[f"{n}|{a}|{k}"][0] for k in ns]
        se = [d[f"{n}|{a}|{k}"][1] for k in ns]
        xs = [i for i, v in enumerate(sl) if v is not None]
        ax.plot(xs, [sl[i] for i in xs], color=col[a], marker="o", ms=4, lw=1.3, label=fr"certified, $\alpha$={float(a):.0%}")
        ax.plot(xs, [se[i] for i in xs], color=col[a], marker="s", ms=4, lw=1.0, ls="--", label=fr"realised, $\alpha$={float(a):.0%}")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"n={k}" for k in ns], fontsize=8)
    ax.set_title(lab, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("net energy saving (% of full)", fontsize=8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(sys.argv[2])
print("wrote", sys.argv[2])
