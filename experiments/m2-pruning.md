# Experiment M2 — Make the Edge Audit Actionable: Pruning Curve + Causal Control

**Addresses:** Review #5, Major Comment M2 (the interpretability audit is purely observational; show a concrete, KAN-specific payoff) and Minor 2 (reframe the KAN-vs-MLP NLS comparison around *faithfulness*).

**Claim under test:** the per-edge **activity** scores produced by the audit are *actionable* — they identify computation that can be removed with little quality loss. The differentiator vs. MLPs is that this intervention operates on genuine forward-pass paths, which the post-hoc MLP "effective edges" are not.

**Design:** rank all FFN edges by audit activity, zero the lowest-activity *p*% (for *p* = 0…90%), and measure validation loss. Compare against a **random-pruning control** at the same *p*. Optionally finetune the pruned model briefly to show recovery. The gap between activity-ranked and random pruning *is* the evidence that the audit is informative.

There is already a `checkpoints/pruned_finetuned/best.pt` artifact in the repo — document its provenance and fold it in here rather than leaving it unexplained.

---

## 1. Minimum viable result (zero-shot, no retraining)

Run from the `kan-guppylm` repo root after copying the draft script into `scripts/`:

```bash
cp /path/to/projectLM01/docs/experiments/prune_edges.py scripts/

uv run python scripts/prune_edges.py \
  --checkpoint checkpoints/kan_grid2_s42/best.pt \
  --tokenizer tokenizer.json \
  --fractions 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --random-control \
  --out results/prune_curve
```

Output: `results/prune_curve/prune_curve.csv` with columns
`fraction, n_pruned, val_loss_activity, val_loss_random` and a PNG plot.

**Decision metric:** the largest fraction *p\** where `val_loss_activity` stays below `baseline + δ` (use δ = 0.005 nats). Report *p\** and the separation from the random control.

Expected shape (the hypothesis): activity-ranked pruning stays flat well past the ~0.4% the paper calls "inactive," because low-activity edges are near-redundant, while random pruning degrades immediately. If the two curves coincide, the audit's activity ranking carries no extra information — also a clean, reportable result (and a caution against over-selling the audit).

## 2. Stronger result (optional, one short finetune)

```bash
uv run python scripts/prune_edges.py \
  --checkpoint checkpoints/kan_grid2_s42/best.pt --tokenizer tokenizer.json \
  --fractions 0.5 0.7 0.9 --finetune-steps 800 \
  --out results/prune_finetune
```

This zeros the selected edges, then finetunes for `--finetune-steps` while holding the pruned edges at zero (masked optimizer), and re-evaluates. It mirrors whatever produced `checkpoints/pruned_finetuned/best.pt`; cross-check the recovered loss against that artifact and cite it in the paper.

## 3. Faithfulness contrast vs. the MLP probe (ties to Minor 2)

The post-hoc MLP "effective edge" `g_{j,i}(a)` (Sec. 3.4 of the paper) is *not* a separable forward-pass path, so zeroing the bottom-activity MLP effective edges should **not** produce a clean pruning curve. Run the same activity-ranked ablation logic on the MLP effective edges and show the degradation is erratic / immediate. This converts the near-null KAN-vs-MLP NLS comparison into a *faithfulness* argument:

> Pruning low-activity KAN edges degrades quality gracefully and far slower than random; the analogous ablation of low-activity MLP effective-edge paths does not, because those paths are post-hoc probes rather than separable computation. The audit is therefore actionable for KANs in a way it is not for MLPs.

(For the MLP side, reuse `scripts/audit_wikitext_mlp_nls.py` / the MLP effective-edge reconstruction already in the codebase; the ablation is "zero the corresponding `W_in[:,i]`/`W_out[j,:]` contribution" — note clearly that this is a diagnostic, not a structural prune.)

## 4. Suggested manuscript additions

- A new Results paragraph in Sec. 4.1 reporting *p\** and the activity-vs-random separation, with the pruning-curve figure.
- One sentence in the Discussion: "These interventions give the audit a concrete payoff — removable-computation identification — that does not transfer to MLP effective-edge probes."
- Provenance for `pruned_finetuned` in the checkpoint-lineage table (App. A).

## 5. Compute estimate

Zero-shot sweep: 10 evaluations on the 3,000-example test split (~94 batches each) on MPS ≈ a few minutes total (no training). Finetune variant: `--finetune-steps 800` × 3 fractions ≈ 10–20 min on MPS.

## 6. Pitfalls and correctness notes

- **Edge importance ≠ weight magnitude.** Use the reconstructed-curve **activity** (the same metric the paper uses for "inactive ≤ 0.01"), not raw `||spline_weight||`, so the prune is defined in the audit's own units and the link to the paper's claim is exact. The script does this.
- **Global vs per-layer ranking.** The script prunes by a *global* activity quantile across all FFN edges (so layers with more low-activity edges are pruned more). A per-layer variant is one flag change; report whichever you use.
- **Masked finetune must re-zero pruned edges every step**, or gradients revive them. The script re-applies the mask after each `optimizer.step()`.
- **Baseline anchoring.** Report the unpruned val loss (fraction 0) from the same evaluator, not the paper's 0.2883 number, so the curve is internally consistent (eval batch count / device may differ slightly).
- **Do not claim symbolic pruning.** This is magnitude/activity pruning guided by the audit, not symbolic simplification.
