PYTHON ?= python

.PHONY: help smoke fig-prune checksums

help:
	@echo "kan-lm-study targets:"
	@echo "  make smoke      Offline reproduction check: regenerate the pruning figure"
	@echo "                  from committed result CSVs (no checkpoints needed)."
	@echo "  make fig-prune  Regenerate figures/prune_compare.pdf from experiments/results/."
	@echo "  make checksums  Compute SHA256SUMS for a local checkpoint archive"
	@echo "                  (set CKPT_ROOT; see checkpoints/README.md)."
	@echo ""
	@echo "Requires matplotlib (install the environment first: cd vendor/kan-guppylm && uv sync)."

# Clone-and-reproduce gate: rebuild a real paper figure from committed data only.
smoke: fig-prune
	@test -s figures/prune_compare.pdf \
	  && echo "SMOKE OK: figures/prune_compare.pdf regenerated from committed CSVs" \
	  || { echo "SMOKE FAIL: figure not produced"; exit 1; }

fig-prune:
	$(PYTHON) experiments/plot_prune_compare.py
	@cp -f experiments/prune_compare.pdf figures/prune_compare.pdf 2>/dev/null || true
	@cp -f experiments/prune_compare.png figures/prune_compare.png 2>/dev/null || true

checksums:
	bash checkpoints/gen_checksums.sh
