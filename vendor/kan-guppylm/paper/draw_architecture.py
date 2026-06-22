"""
Draw the architecture diagram (Figure 1): three FFN variants side by side.
Saves to paper/figures/architecture.pdf

Usage:
    uv run --with . python paper/draw_architecture.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 300})

C_MLP   = "#7f8c8d"
C_KAN   = "#2980b9"
C_MLP_E = "#27ae60"
C_BG    = "#f8f9fa"
C_NODE  = "#ecf0f1"
C_ARROW = "#2c3e50"

def box(ax, x, y, w, h, text, color, fontsize=8, text_color="white", zorder=3):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.03", linewidth=0.8,
                          edgecolor="#2c3e50", facecolor=color, zorder=zorder)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, zorder=zorder+1, fontweight="bold")

def arrow(ax, x0, y0, x1, y1, color=C_ARROW):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.0, mutation_scale=8),
                zorder=2)

def node(ax, x, y, label, r=0.18, color=C_NODE):
    circ = plt.Circle((x, y), r, linewidth=0.8, edgecolor="#2c3e50",
                      facecolor=color, zorder=3)
    ax.add_patch(circ)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5, zorder=4)


fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.2))

# ────────────────────────────────────────────────────────────────────────────
# Panel A — Standard MLP FFN
# ────────────────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(-0.1, 2.1); ax.set_ylim(-0.3, 5.5)
ax.axis("off")
ax.set_facecolor(C_BG)
ax.set_title("(a) MLP FFN (baseline)", fontsize=9, fontweight="bold", pad=6)

# Input
node(ax, 1.0, 0.5, "$\\mathbf{x}$")
# FC1
box(ax, 1.0, 1.6, 1.5, 0.45, "Linear  384→768", C_MLP)
# ReLU
box(ax, 1.0, 2.6, 1.0, 0.4, "ReLU", "#c0392b", fontsize=8)
# FC2
box(ax, 1.0, 3.6, 1.5, 0.45, "Linear  768→384", C_MLP)
# Output
node(ax, 1.0, 4.7, "$\\mathbf{y}$")

arrow(ax, 1.0, 0.68, 1.0, 1.37)
arrow(ax, 1.0, 1.82, 1.0, 2.40)
arrow(ax, 1.0, 2.80, 1.0, 3.37)
arrow(ax, 1.0, 3.82, 1.0, 4.52)

ax.text(1.0, 5.25, "Standard weight matrices\n+ fixed activation",
        ha="center", va="center", fontsize=7, color="#555", style="italic")

# ────────────────────────────────────────────────────────────────────────────
# Panel B — KANLinear (B-spline)
# ────────────────────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(-0.3, 2.3); ax.set_ylim(-0.3, 5.5)
ax.axis("off")
ax.set_title("(b) KANLinear (B-spline)", fontsize=9, fontweight="bold", pad=6)

# Draw 3 input nodes, 3 output nodes, connecting edge curves
in_ys  = [0.6, 1.2, 1.8]
out_ys = [3.0, 3.6, 4.2]
in_x, out_x = 0.45, 1.55

for iy in in_ys:
    node(ax, in_x, iy, "$x_i$", r=0.15)
for oy in out_ys:
    node(ax, out_x, oy, "$y_j$", r=0.15)

# Draw a few edge arcs with spline symbol
rng = np.random.default_rng(1)
for iy in in_ys:
    for oy in out_ys:
        # Bezier-ish curve via a midpoint jitter
        mx = 1.0 + rng.uniform(-0.18, 0.18)
        my = (iy + oy) / 2 + rng.uniform(-0.1, 0.1)
        xs = np.array([in_x + 0.15, mx, out_x - 0.15])
        ys = np.array([iy, my, oy])
        t = np.linspace(0, 1, 30)
        bx = (1-t)**2 * xs[0] + 2*(1-t)*t * xs[1] + t**2 * xs[2]
        by = (1-t)**2 * ys[0] + 2*(1-t)*t * ys[1] + t**2 * ys[2]
        ax.plot(bx, by, color=C_KAN, linewidth=0.9, alpha=0.6, zorder=1)

# Highlight one edge with label
mx, my = 1.05, 2.1
xs = np.array([in_x + 0.15, mx, out_x - 0.15])
ys = np.array([in_ys[0], my, out_ys[2]])
t  = np.linspace(0, 1, 30)
bx = (1-t)**2*xs[0] + 2*(1-t)*t*xs[1] + t**2*xs[2]
by = (1-t)**2*ys[0] + 2*(1-t)*t*ys[1] + t**2*ys[2]
ax.plot(bx, by, color=C_KAN, linewidth=1.8, zorder=2)
ax.annotate("$f_{i,j}(x)$\nB-spline", xy=(mx, my+0.05), ha="center",
            fontsize=7.5, color=C_KAN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C_KAN, lw=0.7))

ax.text(1.0, 0.05, "Learnable grid knots\nGrid update step", ha="center",
        fontsize=7, color="#555", style="italic")

# Einsum formula
box(ax, 1.0, 4.85, 2.0, 0.48,
    'einsum("bik,oik→bo", B(x), W)', C_KAN, fontsize=6.5)
ax.text(1.0, 5.28, "One spline per edge (n_basis = 5)",
        ha="center", fontsize=7, color="#555", style="italic")

# ────────────────────────────────────────────────────────────────────────────
# Panel C — MLPEdge
# ────────────────────────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(-0.3, 2.3); ax.set_ylim(-0.3, 5.5)
ax.axis("off")
ax.set_title("(c) MLPEdge (proposed)", fontsize=9, fontweight="bold", pad=6)

# Same node layout as panel B
for iy in in_ys:
    node(ax, in_x, iy, "$x_i$", r=0.15)
for oy in out_ys:
    node(ax, out_x, oy, "$y_j$", r=0.15)

rng2 = np.random.default_rng(2)
for iy in in_ys:
    for oy in out_ys:
        mx = 1.0 + rng2.uniform(-0.18, 0.18)
        my = (iy + oy) / 2 + rng2.uniform(-0.1, 0.1)
        xs = np.array([in_x + 0.15, mx, out_x - 0.15])
        ys = np.array([iy, my, oy])
        t = np.linspace(0, 1, 30)
        bx = (1-t)**2*xs[0] + 2*(1-t)*t*xs[1] + t**2*xs[2]
        by = (1-t)**2*ys[0] + 2*(1-t)*t*ys[1] + t**2*ys[2]
        ax.plot(bx, by, color=C_MLP_E, linewidth=0.9, alpha=0.6, zorder=1)

# Highlighted edge with tiny-MLP zoom-in
mx, my = 1.05, 2.1
xs = np.array([in_x + 0.15, mx, out_x - 0.15])
ys = np.array([in_ys[0], my, out_ys[2]])
bx = (1-t)**2*xs[0] + 2*(1-t)*t*xs[1] + t**2*xs[2]
by = (1-t)**2*ys[0] + 2*(1-t)*t*ys[1] + t**2*ys[2]
ax.plot(bx, by, color=C_MLP_E, linewidth=1.8, zorder=2)

# Mini MLP diagram on the highlighted edge
bx_mid, by_mid = 0.82, 2.3
box(ax, bx_mid, by_mid, 0.55, 0.34, "$\\sigma(xW_1+b_1)$", C_MLP_E, fontsize=6)
box(ax, bx_mid, by_mid + 0.52, 0.40, 0.30, "$W_2 h$", C_MLP_E, fontsize=6.5)
ax.annotate("", xy=(bx_mid, by_mid+0.37), xytext=(bx_mid, by_mid+0.17),
            arrowprops=dict(arrowstyle="-|>", color=C_MLP_E, lw=0.8, mutation_scale=7))
ax.text(bx_mid+0.36, by_mid+0.26, "tiny\nMLP", ha="left", fontsize=6.5,
        color=C_MLP_E, fontweight="bold")

ax.text(1.0, 0.05, "No grid · standard matmul\nNo update step needed",
        ha="center", fontsize=7, color="#555", style="italic")

# Einsum formula
box(ax, 1.0, 4.85, 2.0, 0.48,
    'einsum("bih,oih→bo", H, W₂)', C_MLP_E, fontsize=6.5)
ax.text(1.0, 5.28, "One MLP per edge (hidden = 5)",
        ha="center", fontsize=7, color="#555", style="italic")

# ────────────────────────────────────────────────────────────────────────────
# Shared label: "KAN topology preserved"
# ────────────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "KAN topology (additive decomposition over edges) is preserved in all three variants.",
         ha="center", fontsize=8, style="italic", color="#444")

fig.tight_layout(rect=[0, 0.04, 1, 1])
out = FIGURES / "architecture.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
