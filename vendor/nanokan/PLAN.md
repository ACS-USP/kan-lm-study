# Plan: Fix rational_kat_cu + Train & Download grkan

**Budget:** $10  
**Goal:** grkan d12 checkpoints on disk, comparable to the mlp run we already have.

---

## Diagnosis

### Why `rational_kat_cu` never worked

`nanochat/gpt.py` tries:
```python
from rational_kat_cu import rat_cuda as _rat_cuda
```

Three problems in the original `Adamdad/rational_kat_cu` repo:
1. `setup.py` has `ext_modules` **commented out** → CUDA extension never compiles.
2. The Python package installs as `kat_rational`, not `rational_kat_cu` → import fails on name alone.
3. No function named `rat_cuda` is exported anywhere.

### What the fork already has (good news)

The fork contains a **working Triton implementation** in `kat_rational/rational_triton.py`:
- `RationalTriton1DGroup` — full fwd/bwd via Triton kernels, no CUDA compilation needed.
- Triton ships with PyTorch on every RunPod image.

### The fix required

Add a `rational_kat_cu/` Python package to the fork that exports `rat_cuda` — a thin wrapper
adapting the Triton kernel to the calling convention `gpt.py` expects:

| | `gpt.py` call | Triton kernel |
|---|---|---|
| Input | `x_flat: (N, d_in)` 2D | any shape, uses last dim as D |
| Numerator | `a: (m+1,)` shared | `weight_numerator: (g, m+1)` per-group |
| Denominator | `b: (g, n)` per-group | `weight_denominator: (g, n)` per-group ✓ |
| Group count | implicit from `b.shape[0]` | explicit `group: int` |

Adapter: expand `a` → `(g, m+1)`, derive `g` from `b`, call `RationalTriton1DGroup.apply`.

---

## Phase 0 — Create network volume (one-time, ~$0.05/week)

A RunPod Network Volume persists across pod terminations and preemptions. The pod's local disk
is wiped on preemption; the volume is not. Checkpoints written there survive.

```bash
python3 scripts/runpod_launch.py volume create --size 20
```

Note the returned volume ID — it is used in every subsequent `train` command.

**Constraints:**
- Volume is tied to datacenter `EU-RO-1` (hardcoded in the script).
- Pod must also launch in `EU-RO-1` — the script enforces this when `--volume-id` is passed.
- Only one pod can mount the volume at a time.

With the volume, `SAVE_EVERY` can be lowered to `500` — checkpoints on the volume don't
need to be downloaded, so size doesn't matter. More frequent saves = less work lost on
any preemption that slips through.

With the volume, **community cloud is safe to use** ($0.79/hr vs $1.14/hr for secure).
If the pod is preempted, we resume from the last checkpoint instead of restarting from zero.

---

## Phase 1 — Fix the fork + add auto-resume (local, $0)

**All work done locally. No pods, no credits spent.**

### 1.1 Add `rational_kat_cu/` package to the fork

Create `rational_kat_cu/__init__.py`:
```python
from kat_rational.rational_triton import RationalTriton1DGroup

def rat_cuda(x, a, b):
    """
    Adapter matching the nanochat/gpt.py calling convention.
    x: (N, d_in), a: (m+1,) shared numerator, b: (g, n) per-group denominator.
    """
    g = b.shape[0]
    a_grouped = a.unsqueeze(0).expand(g, -1).contiguous()
    return RationalTriton1DGroup.apply(x, a_grouped, b, g)
```

### 1.2 Update `setup.py` in the fork

```python
from setuptools import setup, find_packages
setup(
    name='kat_rational',
    version='0.4',
    packages=['kat_rational', 'rational_kat_cu'],
    # ext_modules stays commented out — Triton replaces the CUDA kernel
    ...
)
```

### 1.3 Local import test (CPU, no GPU needed)

```bash
cd ../rational_kat_cu
pip install -e . --quiet
python -c "from rational_kat_cu import rat_cuda; print('OK:', rat_cuda)"
```

Expected: `OK: <function rat_cuda at 0x...>`  
**If this fails: stop. Do not proceed to Phase 2.**

### 1.4 Add auto-resume to `runpod_launch.py` startup script

When a pod restarts after preemption, it should detect the last saved checkpoint on the
volume and pass `--resume-from-step N` automatically instead of starting from zero.

Add this bash snippet to the startup script (runs before training):

```bash
# Detect last checkpoint on volume and set resume flag
CKPT_DIR="<nanochat_base>/base_checkpoints/<run_name>"
LAST_STEP=$(ls "$CKPT_DIR"/model_*.pt 2>/dev/null \
    | sed 's/.*model_0*//' | sed 's/\.pt//' \
    | sort -n | tail -1)
if [ -n "$LAST_STEP" ] && [ "$LAST_STEP" -gt 0 ]; then
    RESUME_FLAG="--resume-from-step $LAST_STEP"
    echo "Resuming from step $LAST_STEP"
else
    RESUME_FLAG=""
    echo "Starting from scratch"
fi
```

Then append `$RESUME_FLAG` to the `torchrun` command.

### 1.5 Update `runpod_launch.py` pip install line

Change the grkan install to use our fork:
```python
pip install git+https://github.com/<fork-url>/rational_kat_cu.git
```

Also restore `device_batch_size=32` for grkan (OOM was caused by Horner loop intermediates,
not the rational function itself — Triton kernel is memory-efficient; validate in smoke test).

And set `SAVE_EVERY = 500`.

### 1.6 Push fork + push nanokan

```bash
# Push fork
cd ../rational_kat_cu
git add -A
git commit -m "fix: add rational_kat_cu package with Triton-backed rat_cuda adapter"
git push

# Push nanokan
cd ../nanokan
git add scripts/runpod_launch.py
git commit -m "fix: auto-resume, install rational_kat_cu from fork, SAVE_EVERY=500"
git push
```

**Verify both pushes** with `curl raw.githubusercontent.com/...` before any pod launch.

---

## Phase 2 — Smoke test (~$0.20 estimated)

**Goal:** Confirm the full pipeline — install, import, train, checkpoint to volume, download — before committing to a 7-hour run.

### Checklist

1. `from rational_kat_cu import rat_cuda` imports successfully.
2. `_RAT_CUDA_AVAILABLE = True` appears in the training log.
3. 10 training steps complete without OOM or crash.
4. A checkpoint appears on the volume (`model_000010.pt` or similar).
5. `cmd_download` rsyncs that checkpoint to local disk successfully.

### How to run

```bash
python3 scripts/runpod_launch.py train d12 grkan \
    --volume-id <vol_id> \
    --smoke   # sets max_steps=10, save_every=10
```

(`--smoke` flag to be added to `runpod_launch.py` in Phase 1.)

### Go / no-go

| Check | Pass | Fail → action |
|---|---|---|
| Import | `_RAT_CUDA_AVAILABLE = True` in log | Fix Phase 1 locally, push, retry smoke |
| No OOM at batch 32 | First step completes | Lower `device_batch_size` to 16, update, retry smoke |
| Checkpoint on volume | `model_000010.pt` exists on volume | Fix startup path, retry smoke |
| Download succeeds | `.pt` file on local disk, nonzero | Fix rsync config, retry smoke |

**Hard rule: if smoke fails twice for the same reason, stop and diagnose. Do not retry blind.**

---

## Phase 3 — Full grkan training run (~$5.50)

**Only start after Phase 2 passes all four checks.**

### Pod configuration

- **Cloud type: community** — safe now that checkpoints are on the volume. Saves ~$2.45
  over secure cloud on a 7-hour run.
- GPU: L40S (same as mlp run, ensures fair comparison).
- `SAVE_EVERY=500` — checkpoint every ~8 min on L40S; at most 8 min lost on preemption.
- `device_batch_size=32` if smoke test validated; `16` if it OOMed.
- `--volume-id <vol_id>`

### Launch

```bash
python3 scripts/runpod_launch.py train d12 grkan --volume-id <vol_id>
```

### If preempted

1. `runpod_launch.py ls` — confirm pod is gone.
2. Relaunch with the same `--volume-id`. Auto-resume picks up from last checkpoint.
3. No manual `--resume-from-step` needed (that's what Phase 1.4 adds).

### Monitor

Keep tmux watcher running. If SSH check fails 3 times:
- Check remaining credits before doing anything.
- If < $2 remaining: **do not relaunch**, download whatever is on the volume, stop.
- If credits OK and auto-resume logic is in place: relaunch.

### Download

Watcher handles automatically. Rsync uses `--partial`.  
After download: verify `model_002520.pt` is present and ~756 MB.

---

## Cost budget

| Item | Estimated cost |
|---|---|
| Phase 0 — volume (20 GB, 1 week) | ~$0.05 |
| Phase 1 — local | $0 |
| Phase 2 — smoke test (~15 min, any GPU) | ~$0.20 |
| Phase 3 — full run, L40S community ~7h | ~$5.53 |
| **Total** | **~$5.78** |
| Buffer remaining | ~$4.22 |

**Hard stop rule:** If remaining credits drop below $2 unexpectedly at any point,
terminate all pods immediately and assess before continuing.

---

## Success definition

Both files present on local disk:
```
checkpoints/nanochat/d12-mlp/model_002520.pt    # already have this ✓
checkpoints/nanochat/d12-grkan/model_002520.pt  # target
```

Then: load both, evaluate val_bpb at matched steps, compare.
