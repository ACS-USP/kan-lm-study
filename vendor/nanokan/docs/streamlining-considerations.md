# Streamlining the Orchestrator Workflow

**Date:** 2026-06-10  
**Context:** g4 smoke→pilot→full campaign completed (~4.5h wall clock). G16 first pod failed (SSH unreachable). User asked how to reduce total time.

## Where the time goes (g4 campaign, ~4.5h)

| Component | Time | Notes |
|---|---|---|
| GPU training (all 3 phases) | ~82 min | Unavoidable computation |
| Full-phase download (6 ckpts) | ~40 min | **Dominant overhead** — slow MFS↔local rsync (~2-3 MB/s) |
| Pod provisioning × 3 | ~60-90 min total | Smoke pod: full cold boot. Pilot & full: fast (venv+dataset on volume) |
| Inter-phase overhead | ~30-40 min | Terminate+download+verify+relaunch × 2 transitions |

## Option A: Chain training phases on one pod

**Saves:** ~30-40 min (10-15% of total) by eliminating 2 intermediate pod launches and 2 intermediate downloads.

**Costs:**
- No intermediate artifact verification — smoke/pilot checkpoints live only on the volume. A crash at full step 2400 loses all three phases' work.
- Crash blast radius: entire campaign, not one phase.
- The orchestrator's safety model (download → verify local → only then terminate) is designed per-phase. Chaining would need re-architecting intra-pod multi-phase sequencing including the setup-deadline guard and per-phase gating.

**Verdict:** Not recommended. The speedup is modest; the safety regression is significant.

## Option B: Skip intermediate downloads (verify on-volume)

**Saves:** ~20-25 min (eliminates smoke and pilot downloads).

**How:** Instead of downloading smoke/pilot checkpoints locally before advancing, verify them in-place via SSH: check file exists on volume, parse meta for finite loss, check log for guard events. Only download after the full phase or do one bulk download at the end.

**Costs:** Delays the "prove we can retrieve artifacts" proof to the end of the campaign rather than per-phase. User explicitly rejected this: "testing if we will be able to retrieve the artifacts is crucial."

**Verdict:** Rejected by user. The retrieval proof per phase is the orchestrator's core value.

## Option C: Download from pod NVMe instead of volume

**The real bottleneck is MFS read bandwidth** (~2-10 MB/s). The pod's local NVMe can read at 50-200× that speed. Currently, volume-backed runs write checkpoints directly to `/runpod-volume/nanochat/jobs/{job_id}/` (MFS), so every download reads over the network filesystem.

**Required changes:**
1. Write training checkpoints to NVMe (`~/.cache/nanochat/jobs/{job_id}/`) instead of the volume.
2. After each checkpoint save, copy NVMe → volume (for persistence) and rsync NVMe → local (for retrieval proof). Both run concurrently with training.
3. By the time training finishes at step 2520, 5 of 6 checkpoints are already local and on the volume. Only the final checkpoint downloads post-DONE (~1-2 seconds at NVMe read speed).

**Pod-side impact:** The startup script is the highest-stakes code in the repository. Changing `NANOCHAT_BASE_DIR` from the volume path to an NVMe path touches the startup script, `base_train.py`, `checkpoint_manager.py`, and the dataset/tokenizer layout. A bug here silently breaks every training pod.

**Verdict:** Highest potential speedup (~35-40 min saved), but highest implementation risk due to pod-side changes. Recommended only as a dedicated follow-up with thorough pod testing.

## Option D: Parallel rsync from volume (supervisor-side only, zero pod changes)

**Saves:** ~20-25 min of post-training wait by saturating the MFS link with concurrent transfers.

**How:** The supervisor already streams the training log line-by-line and already runs incremental rsync. When a checkpoint-save log line appears (e.g. `Saved model parameters to: .../model_000500.pt`), launch targeted per-file rsyncs in parallel (`subprocess.Popen`) instead of one blocking bulk rsync. Track pending transfers; DONE waits for all to finish; transfer failure stops the pod (integrity guard).

**Code impact:** ~30-50 lines in `cmd_watch` only. No pod-side changes. No startup script changes. No changes to base_train, checkpoint_manager, or gpt.py. Tests: mock Popen, verify transfer-fires-on-save-line, verify DONE blocks on incomplete transfers, verify failure stops pod.

**Verdict:** Recommended first step. Zero blast radius (pod-side unchanged), significant speedup, naturally testable. Implementation deferred to a follow-up session.

## Option E: runpodctl direct transfer

RunPod's `runpodctl` uses RunPod infrastructure rather than SSH tunneling; may have better bandwidth than SSH+rsync. The binary is already installed locally. Requires investigation of `runpodctl`'s transfer commands and bandwidth characteristics before committing.

**Verdict:** Worth evaluating alongside Option D. If `runpodctl` delivers 2-3× the throughput of SSH+rsync, the combined effect with parallel transfers approaches NVMe-level speed without any pod-side changes.

## Summary

| Option | Time saved | Pod-side risk | Code changes |
|---|---|---|---|
| A. Chain phases on one pod | 30-40 min | High (crash blast radius) | Orchestrator + supervisor |
| B. Skip intermediate downloads | 20-25 min | None | Orchestrator verify logic |
| C. NVMe checkpoint writes | 35-40 min | **High** (startup script) | Startup + base_train + checkpoint_manager |
| D. Parallel rsync from volume | 20-25 min | **None** | `cmd_watch` only (~30-50 lines) |
| E. runpodctl transfer | Unknown | None | `cmd_watch` download path |

**Recommended sequence:** D (parallel rsync, low-risk/high-reward) → evaluate speedup → E (runpodctl if bandwidth is better) → C (NVMe if still bottlenecked).
