"""H26 S3: Figure 1 (chain-to-full net-energy ratio per exit path) from results/s15/sens + cost JSON.
Usage: python make_fig.py <H26 folder> <out.pdf>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = os.path.join(sys.argv[1], "results", "s15")
sens = json.load(open(os.path.join(R, "sens", "sens_s15.json"), encoding="utf-8"))["nets"]
h12 = json.load(open(os.path.join(R, "cost", "h1h2.json"), encoding="utf-8"))
nets = [("n1", "N1 ResNet-56"), ("n2", "N2 ResNet-50*"), ("n4", "N4 MobileNetV2*"), ("n3", "N3 ViT-B/16"), ("n5", "N5 DeiT-S")]
style = {"n1": ("#b2182b", "o", "-"), "n2": ("#d6604d", "s", "-"), "n4": ("#f4a582", "^", "-"),
         "n3": ("#2166ac", "D", "--"), "n5": ("#4393c3", "v", "--")}
fig, ax = plt.subplots(figsize=(5.2, 2.9))
for n, lab in nets:
    m = sens[n]["main"]
    ys, lo, hi = [], [], []
    for k in (1, 2, 3):
        d = m[f"H3_chain{k}_over_full"]
        ys.append(d["mean"]); lo.append(d["mean"] - d["ci95"][0]); hi.append(d["ci95"][1] - d["mean"])
    d = m["H1_R"]
    ys.append(d["mean"]); lo.append(d["mean"] - d["ci95"][0]); hi.append(d["ci95"][1] - d["mean"])
    c, mk, ls = style[n]
    ax.errorbar([1, 2, 3, 4], ys, yerr=[lo, hi], color=c, marker=mk, ls=ls, lw=1.3, ms=4, capsize=2, label=lab)
ax.axhline(1.0, color="0.3", lw=0.8)
ax.text(4.08, 1.0, "full engine", va="center", fontsize=7, color="0.3")
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(["exit 1", "exit 2", "exit 3", "exit 4\n(no exit taken)"], fontsize=8)
ax.set_ylabel(r"$\bar c_{\mathrm{chain},k}/\bar c_{\mathrm{full}}$ (net energy)", fontsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.set_xlim(0.8, 4.6)
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(sys.argv[2])
print("wrote", sys.argv[2])
