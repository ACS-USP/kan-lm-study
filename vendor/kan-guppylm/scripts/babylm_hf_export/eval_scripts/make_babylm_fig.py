"""BabyLM standardized-evaluation summary figure.
Three panels sharing the same architecture order (by validation loss, best->worst):
  A: BabyLM validation CE (monotonic worsening L->R by construction)
  B: BLiMP accuracy +/- 95% CI (n=10) -- does NOT track val loss; GR-KAN highest
  C: BLiMP-supplement accuracy +/- 95% CI (n=10) -- ranking reverses; MLP highest
Message: validation loss does not predict the standardized-benchmark ranking.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

# Architectures in validation-loss order (best -> worst)
archs   = ["SwiGLU", "Cheb.\n$d3\\,g8$", "GR-KAN", "MLP"]
colors  = ["#E69F00", "#009E73", "#2F6BFF", "#555555"]  # MLP = neutral gray (baseline)
val_ce  = [3.770, 3.781, 3.800, 3.820]
blimp   = [63.03, 62.77, 63.13, 62.44]
blimp_ci= [0.43, 0.37, 0.28, 0.40]
suppl   = [54.11, 53.95, 54.02, 54.42]
suppl_ci= [0.51, 0.36, 0.39, 0.33]
x = range(len(archs))

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.6, 2.7))

# Panel A: validation CE (bars; lower = better)
axA.bar(x, val_ce, color=colors, width=0.62, edgecolor="black", linewidth=0.4)
axA.set_ylim(3.76, 3.83)
axA.set_ylabel("BabyLM val. CE  ($\\downarrow$)")
axA.set_title("(a) Validation loss", fontsize=9)
axA.annotate("", xy=(3.25, 3.826), xytext=(0.0, 3.826),
             arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
axA.text(1.6, 3.829, "worsening", ha="center", va="bottom", fontsize=7.5, color="0.4")

# Panel B: BLiMP accuracy +/- 95% CI
axB.errorbar(x, blimp, yerr=blimp_ci, fmt="o", ms=6, capsize=3, lw=1.1,
             mfc="white", mew=1.2, ecolor="0.3",
             color="0.3")
for xi, yi, c in zip(x, blimp, colors):
    axB.plot(xi, yi, "o", ms=6, mfc=c, mec="black", mew=0.5, zorder=5)
axB.set_ylim(61.6, 64.0)
axB.set_ylabel("BLiMP acc. \\% ($n{=}10$)")
axB.set_title("(b) BLiMP", fontsize=9)
axB.annotate("best", xy=(2, blimp[2]+blimp_ci[2]), xytext=(2, 63.8),
             ha="center", fontsize=7, color="#2F6BFF",
             arrowprops=dict(arrowstyle="->", color="#2F6BFF", lw=0.8))

# Panel C: BLiMP-supplement accuracy +/- 95% CI
axC.errorbar(x, suppl, yerr=suppl_ci, fmt="o", ms=6, capsize=3, lw=1.1,
             mfc="white", mew=1.2, ecolor="0.3", color="0.3")
for xi, yi, c in zip(x, suppl, colors):
    axC.plot(xi, yi, "o", ms=6, mfc=c, mec="black", mew=0.5, zorder=5)
axC.set_ylim(53.2, 55.0)
axC.set_ylabel("BLiMP-suppl. \\% ($n{=}10$)")
axC.set_title("(c) BLiMP supplement", fontsize=9)
axC.annotate("best", xy=(3, suppl[3]+suppl_ci[3]), xytext=(3, 54.85),
             ha="center", fontsize=7, color="#555555",
             arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

for ax in (axA, axB, axC):
    ax.set_xticks(list(x))
    ax.set_xticklabels(archs, fontsize=7.5)
    ax.tick_params(axis="both", labelsize=7.5, length=3)
    ax.margins(x=0.12)

fig.tight_layout(w_pad=1.4)
out = "/Users/felippealves/Documents/GitHub/projectLM01/docs/figures/babylm_eval_summary.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
