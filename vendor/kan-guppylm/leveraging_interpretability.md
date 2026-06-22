# Leveraging KAN Interpretability

Concrete ways to turn our interpretability results (87.8% nonlinear, median NLS=0.281, p<10⁻³⁰⁰) into paper contributions.

---

## #1 — Structured Pruning (low effort, high impact)

We know ~0.4% of edges are dead and a fraction are near-linear. Prune edges where:

```
prunable = (activity < τ_act · layer_mean_activity) AND (nls < τ_nls)
```

Then measure val_loss degradation vs. sparsity. If we can prune 20–40% of edges with <0.01 val_loss increase, that's a direct downstream product of the interpretability analysis — the KAN's explicit function structure makes the pruning criterion principled rather than heuristic.

**Paper contribution**: Sparsity table (per-layer prunable fraction) + val_loss vs. sparsity curve. Shows interpretability is not just descriptive but actionable.

---

## #2 — Functional Archetypes via FPCA (medium effort, strong narrative)

Run Functional PCA (via `scikit-fda`) on all reconstructed edge curves per layer. If PC1+PC2 explain >70% of variance, the model converged to a low-dimensional vocabulary of activation shapes despite having 147K free functions.

**Paper contribution**: Plot of top-4 fPCA components per layer (the "shape archetypes"). Report explained variance. If the model is interpretable in the strong sense, these archetypes should be recognizable (monotone, sigmoid-like, quadratic, etc.).

---

## #3 — Domain-Selective Circuits (medium effort, very visual)

Register forward hooks on all KANLinear layers. Run 8 fish-domain prompts vs. 8 generic prompts. Compute per-edge mean activation difference:

```
diff[layer, i, j] = mean_fish_activation[i,j] - mean_generic_activation[i,j]
```

Visualize as signed heatmaps. Sparse, consistent high-magnitude cells = "fish personality circuits" — edges that specifically encode domain knowledge.

**Paper contribution**: Heatmap figure showing domain-selective edges. Analogous to mechanistic interpretability circuit analysis but made tractable by KAN's explicit 1D edge structure.

---

## #4 — Symbolic Regression on High-NLS Edges (high effort, high prestige)

For the top-NLS B-spline edges, fit closed-form expressions using `scipy.curve_fit` or PySR. If even a handful of edges learn recognizable functions (sigmoid, quadratic, periodic), claim partial symbolic extraction — the strongest form of KAN interpretability.

---

## #5 — Targeted Domain Adaptation (high effort, applied angle)

Use the domain-selective circuit map from #3 to do surgical fine-tuning: freeze all edges except the identified domain-selective ones, then adapt the model to a new domain. Compare convergence vs. full fine-tuning. If it converges faster, interpretability directly enables efficient transfer.
