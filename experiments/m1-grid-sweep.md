# Experiment M1 — Grid-Size Sweep for the Interpretability Claims

**Addresses:** Review #5, Major Comment M1 (the 99.9% fPCA / 93.3% closed-form numbers may be artifacts of the grid-2 basis ceiling, and are tested only at grid 2).

**Claim under test:** the low-dimensional, smooth, closed-form-fittable structure of trained KAN edge functions is **training-induced**, not imposed by the small `n_basis = grid_size + spline_order` of the grid-2 basis.

**Design:** train the same GuppyLM KAN at grid sizes {2, 5, 10, 20}, run the identical edge audit on each, and add a random-init control at each grid size. Plot each audit metric vs. grid size.

---

## 1. Why this is the decisive experiment

At grid size 2 with cubic splines, each edge lives in a `2 + 3 = 5`-dimensional spline space plus a SiLU base. A top-4 fPCA explaining 99.9% of variance is then close to structural, and low-degree-polynomial fits on a bounded domain are nearly guaranteed. The only way to show the structure is *learned* is to give training a larger basis and check whether it still collapses to a few smooth motifs. As `grid_size` grows (n_basis = 5, 8, 13, 23), the available function space grows, so a *persisting* collapse is genuine evidence; a *degrading* collapse is itself a publishable scaling-of-interpretability finding.

## 2. Train the checkpoints

Run from the `kan-guppylm` repo root. One seed (42) is enough for the headline sweep; add seeds 43/44 at grid 5 for a stability point. Use the same 8,000-step protocol as the paper.

```bash
for G in 2 5 10 20; do
  uv run python -m kanprey.train --model kan --grid-size $G --steps 8000 \
    --batch-size 32 --seed 42 \
    --checkpoint-dir checkpoints/gridsweep/kan_grid${G}_s42
done

# optional stability point at grid 5
for S in 43 44; do
  uv run python -m kanprey.train --model kan --grid-size 5 --steps 8000 \
    --batch-size 32 --seed $S \
    --checkpoint-dir checkpoints/gridsweep/kan_grid5_s${S}
done
```

Record the **best val loss** each run reports — if it degrades sharply at high grid size, that is a confound (the audit then describes a worse model) and must be reported alongside the interpretability metrics. Grid-2 is already ~0.288 in the paper; expect grid-5/10 to be similar or slightly better.

## 3. Run the audit

The draft script `audit_grid_sweep.py` (in this folder; copy to `kan-guppylm/scripts/`) reconstructs every FFN edge, computes the same metrics as the paper, and writes one JSON per checkpoint plus a combined CSV. It reuses the exact reconstruction formula from `kanprey/interpret.py` and the closed-form library from `paper/symbolic_regression.py`.

```bash
cp /path/to/projectLM01/docs/experiments/audit_grid_sweep.py scripts/

uv run python scripts/audit_grid_sweep.py \
  --checkpoints \
     grid2=checkpoints/gridsweep/kan_grid2_s42/best.pt \
     grid5=checkpoints/gridsweep/kan_grid5_s42/best.pt \
     grid10=checkpoints/gridsweep/kan_grid10_s42/best.pt \
     grid20=checkpoints/gridsweep/kan_grid20_s42/best.pt \
  --random-control \
  --n-points 200 --top-k 50 --r2-thresh 0.99 \
  --out results/gridsweep_audit
```

Each row reports: median NLS, %nonlinear (>0.1), %inactive (≤0.01), min/mean top-4 fPCA variance across the 6 FFN layers, and pooled top-50 closed-form coverage (R²>0.99). The `--random-control` flag audits a freshly initialized model with the same config at each grid size, extending the paper's single grid-2 control across the sweep.

## 4. Expected output and decision rule

Produce this table (trained rows + a random-init row per grid size):

| grid | n_basis | best val loss | median NLS | %>0.1 | %inactive | top-4 fPCA (min) | closed-form R²>0.99 |
|------|---------|---------------|------------|-------|-----------|------------------|---------------------|
| 2    | 5       | …             | …          | …     | …         | …                | …                   |
| 5    | 8       | …             | …          | …     | …         | …                | …                   |
| 10   | 13      | …             | …          | …     | …         | …                | …                   |
| 20   | 23      | …             | …          | …     | …         | …                | …                   |

**Decision rule:**
- If top-4 fPCA variance stays high (say >95%) and closed-form coverage stays high while the random-init control *falls* as the grid grows → **structure is learned.** Add the sweep table and the sentence: *"The fPCA collapse and closed-form coverage are stable across grid sizes 2–20, while the random-init control degrades with grid size, so they reflect training-induced structure rather than the dimensionality of the grid-2 basis."*
- If the trained metrics fall toward the random-init control as grid grows → report the crossover grid size as a finding: *"KAN edge interpretability is strongest in the low-capacity regime and weakens by grid size N."*

Either outcome is reportable; both remove the M1 objection.

## 5. Compute estimate

4 trainings × ~10–20 min each on the M4 Pro/MPS (grid-2 was 1084 s; higher grids are slower per step) ≈ 1–2 h. Audit is seconds per checkpoint. Add ~40 min for the two extra seeds at grid 5.

## 6. Pitfalls

- **Grid update / `update_grid`:** `KANLinear` has an adaptive-grid path (`update_grid` in `kan_layers.py`). The reconstruction reads `module.grid` directly, so it is correct regardless, but confirm the grid actually spans the activation domain at high grid size (the audit reports the per-channel range; flag any near-degenerate ranges).
- **Param count drift:** higher grid sizes increase parameters; this is expected and is *not* a fair-replacement comparison — M1 is purely an interpretability-robustness check, so do not add these rows to the replacement table.
- **fPCA conditioning:** at grid 20 the curve matrix is richer; the script centers per-column before SVD (matching `interpret.py`). Report both min and mean across layers in case one layer behaves differently.
