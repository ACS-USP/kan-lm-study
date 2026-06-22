# M1 / M2 Experiment Results (Review #5)

Run on GuppyLM checkpoints, seed 42, Apple M4 Pro / MPS. Audit code: `docs/experiments/audit_grid_sweep.py`, `prune_edges.py`. Raw outputs in `docs/experiments/results/`.

> **Headline:** M2 is a clean *positive* for the paper (the audit is actionable). M1 is a clean *negative/nuance*: the paper's headline interpretability numbers are largely a **basis-ceiling artifact**, and an independent re-run does **not** reproduce the paper's random-init control. Both are publishable; both require manuscript changes.

---

## M2 — Activity-guided pruning (actionable audit)

Checkpoint: `checkpoints/kan_grid2_s42/best.pt` (the paper's grid-2 KAN). Zero-shot, no retraining. Full test split.

| Fraction pruned | Activity-ranked val loss | Random-prune val loss |
|---|---|---|
| 0% (baseline) | 0.2818 | — |
| 10% | 0.2824 | 0.2942 |
| 20% | 0.2847 | 0.3348 |
| 30% | 0.2896 | 0.4798 |
| 40% | 0.3023 | 0.8730 |
| 50% | 0.3550 | 1.8727 |
| 70% | 1.1668 | 3.4290 |
| 90% | 4.1648 | 5.8749 |

**Verdict: positive and clean.** ~20–25% of all 884,736 FFN edges prune with a <0.005-nat increase; random pruning of even 10% hurts more than activity-ranked pruning of 30%. At 50% the gap is 0.355 vs 1.87. The audit's per-edge **activity** score (the same metric used for the paper's "inactive ≤ 0.01" count) directly identifies removable computation — this is the actionable, KAN-specific payoff Reviewer #5 asked for in M2.

Outputs: `results/prune_curve/prune_curve.csv`, `prune_curve.png`.

---

## M1 — Grid-size sweep (is the interpretability structure learned or basis-imposed?)

Trained grid {2,5,10,20} at 8000 steps, seed 42; audited each + a random-init control. `n_basis = grid_size + spline_order(3)`.

### Trained models

| grid | n_basis | val loss | fPCA top-4 (min) | closed-form R²>0.99 | median NLS | %inactive |
|---|---|---|---|---|---|---|
| 2 | 5 | **0.288** | **99.93%** | **97.0%** | 0.29 | 0.46% |
| 5 | 8 | **0.312** | 91.0% | 18.7% | 0.64 | 0.007% |
| 10 | 13 | 1.115 | 72.3% | 3.3% | 0.76 | 0% |
| 20 | 23 | 3.881 | 59.4% | 0.0% | 0.82 | 0% |

### Random-init controls

| grid | n_basis | fPCA top-4 (min) | closed-form R²>0.99 | median NLS |
|---|---|---|---|---|
| 2 | 5 | **99.93%** | 65.0% | 0.27 |
| 5 | 8 | 91.5% | 4.0% | 0.67 |
| 10 | 13 | 64.5% | 0.0% | 0.84 |
| 20 | 23 | 37.2% | 0.0% | 0.92 |

Figure: `results/gridsweep_audit/gridsweep_capacity.png`. Raw: `gridsweep_summary.csv` + per-checkpoint JSON.

### Finding 1 — The headline metrics are largely a basis-ceiling artifact (Reviewer M1 validated)

- **fPCA "99.9% in 4 components" is entirely basis-driven at grid-2.** Trained and random-init both give **99.93%** — identical. So at grid-2 the low-dimensional collapse is a property of the 5-dimensional spline basis, *not* of training.
- **Both metrics collapse as capacity grows**, even at *matched training quality*. The cleanest comparison is grid-2 vs grid-5, which trained to comparable loss (0.288 vs 0.312): closed-form coverage crashes **97% → 18.7%** and fPCA **99.93% → 91.0%** with just 3 more basis functions. This is not a convergence artifact — grid-5 is well-trained.
- The paper's signature numbers (99.9% fPCA, 93–97% closed-form) therefore do **not** generalize beyond the maximally-constrained grid-2 basis.

### Finding 2 — The paper's random-init control does not reproduce (needs reconciliation)

The manuscript (Sec. 4.1) reports the random-init grid-2 control as **top-4 fPCA = 82.3%**, "dominated by high-frequency oscillations," and **34.7%** closed-form coverage, concluding the structure is "training-induced rather than basis-imposed."

This independent re-run of a freshly-initialized grid-2 `KANpreyLM` (`spline_weight ~ N(0,0.1)`, Kaiming `base_weight`) gives:
- top-4 fPCA = **99.93%** (not 82.3%) — indistinguishable from trained;
- median NLS = **0.27** (smooth, *not* high-frequency);
- closed-form coverage = **65.0%** (not 34.7%).

So as computed here, a random-init grid-2 KAN is already smooth and low-dimensional — because the small spline noise leaves the SiLU base dominant. **The fPCA part of the paper's control claim is not reproducible**, and the "training collapses the basis into smooth motifs" conclusion is undercut for fPCA (though closed-form coverage *is* genuinely training-improved: 97% vs 65%).

**Reconciliation (init-variance sweep, `init_variance_sweep.py`).** Sweeping the grid-2 spline-init std over {0.1, 0.3, 0.5, 1.0, 2.0, 5.0} leaves fPCA top-4 pinned at **99.93%** and median NLS at **0.27** at *every* scale — the number does not move at all. The reason is structural: at grid-2, `n_basis + 1 = 6` (five shared B-splines + the SiLU base), so every edge is a linear combination of 6 fixed functions and the per-layer curve matrix has rank ≤ 6. Top-4 fPCA is therefore ≈99.9% *by construction*, independent of weights or training, and a 5-function basis over (−1,1) cannot produce "high-frequency oscillations" (NLS stays ~0.27 even at std 5.0).

This pins down the discrepancy: the paper's random-init numbers (82.3% fPCA, high-frequency oscillations, 34.7% closed-form) are **inconsistent with a grid-2 model** and instead match a **higher-grid** random model — our random-init controls give 91.5% (grid-5) and 64.5% (grid-10) fPCA, bracketing 82.3%. The control almost certainly used a mismatched grid size. The correct grid-2 random-init baseline is ~99.9% fPCA / ~63% closed-form. Consequence: the **fPCA "low-dimensionality" is basis-imposed, not training-induced**; only the **closed-form smoothness is genuinely training-induced** (trained 97% vs random ~63%). Our method reproduces the paper's *trained* numbers exactly (88.2% nonlinear, 99.93% fPCA, val 0.288), so the metric is sound — the issue is the control's architecture.

### Finding 3 — Trainability confound at high grid (honest caveat)

Higher grids did not converge under the fixed 8000-step/LR protocol: val loss 0.288 → 0.312 → 1.115 → **3.881**. So grid-10/20 audits partly reflect undertraining, and the capacity vs convergence effects are entangled there. The grid-2-vs-grid-5 comparison (matched quality) is what carries Finding 1; grid-10/20 only extend the trend. Training itself does still concentrate variance more than random at every grid (fPCA 72% vs 64% at g10; 59% vs 37% at g20), so training is doing *something* — just far less than the grid-2 headline implies.

---

## What this means for the manuscript

1. **M2 → add a positive result.** Pruning curve + random control as a new Sec. 4.1 figure; the audit identifies removable computation (KAN-specific, actionable). Fold in the existing `checkpoints/pruned_finetuned` artifact.
2. **M1 → narrow the interpretability claim.** State explicitly that the fPCA collapse and closed-form coverage are properties of the **low-capacity (grid-2) regime** and degrade with basis size; do not present 99.9%/93.3% as general properties of trained KAN LM edges.
3. **Reconcile or correct the random-init control (Sec. 4.1).** The 82.3% / 34.7% numbers did not reproduce here. Either document the exact init/fPCA procedure that yields them, or revise the control. This is load-bearing — it is the paper's main argument that the structure is "training-induced."

## Confound removal — done (Finding 1 confirmed at matched quality)

Diagnosis from the original training logs: grid-10/20 were **undertrained, not divergent** — both val losses were still descending when the 8000-step cosine schedule ended (grid-20: 3.92→3.89 at step 7800, LR already at 3e-5; grid-10: 1.26→1.115 over its last 1000 steps). A first retraining at **lower LR (1e-4) backfired** (grid-10 stuck at val 4.87 by step 1200) — the issue was the schedule/horizon, not LR magnitude. Retraining at the **original lr=3e-4 with the cosine stretched to 16,000 steps** fixed it:

| grid | n_basis | val loss (8k orig → 16k) | fPCA top-4 (orig → retrained) | closed-form (orig → retrained) |
|---|---|---|---|---|
| 10 | 13 | 1.115 → **0.371** | 72.3% → **75.3%** | 3.3% → **13.3%** |
| 20 | 23 | 3.881 → **3.025** | 59.4% → 62.3% | 0% → 1.0% |

**The decisive result:** grid-10 retrained to **val 0.371** (matched to grid-2's 0.288 / grid-5's 0.312) yet its fPCA top-4 is still **75.3%** and closed-form coverage still **13.3%** — essentially unchanged from the undertrained version. So the interpretability collapse is **not** an undertraining artifact; it persists at matched quality. The confound-free trend across three comparable-loss models is monotonic:

| grid | n_basis | val loss | fPCA top-4 (min) | closed-form R²>0.99 |
|---|---|---|---|---|
| 2 | 5 | 0.288 | 99.9% | 97.0% |
| 5 | 8 | 0.312 | 91.0% | 18.7% |
| 10 | 13 | 0.371 | 75.3% | 13.3% |

**Grid-20 is a separate trainability finding:** it would not converge to comparable loss even at 16,000 steps (val 3.02, still descending), evidence that high-capacity B-spline KAN FFNs are hard to optimize at this scale. Audit JSON: `results/gridsweep_audit_long/`.

## Suggested follow-ups (remaining)

- **Investigate the random-init discrepancy at the source:** audit the *exact* random-init checkpoint the paper used (if saved) to confirm the grid-mismatch hypothesis. The init-variance sweep above already establishes the grid-2 fPCA is basis-pinned regardless of weights.
