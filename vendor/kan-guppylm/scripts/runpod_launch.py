"""
RunPod pipeline for KanpreyLM training.

Safe route: test → volume create → train → download → terminate

Subcommands
-----------
  test     [--gpu GPU] [--dry-run]
      Spin up a pod, run the GPU smoke test, auto-terminate.
      Use this before every real training run (~2 min, ~$0.02).

  train    --model MODEL [--gpu GPU] [--volume-id ID] [--steps N] [--dry-run]
      Launch a training pod. Pod stops itself when done (disk preserved).
      If --volume-id is given, checkpoints are also written to the
      persistent network volume (survives community-cloud interruptions).

  download <pod_id> [--dest DIR]
      rsync results from a stopped pod to local disk, then offer to terminate.
      Requires your SSH public key to be registered in RunPod settings.

  ls
      List running/stopped pods with cost per hour and uptime.

  stop  <pod_id>
      Stop a pod (preserves disk) without terminating it.

  terminate  <pod_id>
      Permanently terminate a pod and delete its disk.

  volume  create [--size GB]  |  ls  |  delete <volume_id>
      Manage persistent NVMe network volumes.
      Cost: ~$0.07 / GB / month  (5 GB ≈ $0.35/month).
      Volumes survive pod terminations and community interruptions.

  gpus
      List available GPU types and community prices.

Prerequisites
-------------
    export RUNPOD_API_KEY=<your-key>
    export GITHUB_TOKEN=<your-pat>   # only needed for train/test

Examples
--------
    # 1. Verify GPU environment (do this first, every time)
    python scripts/runpod_launch.py test

    # 2. Create a persistent volume once (reuse across runs)
    python scripts/runpod_launch.py volume create --size 5

    # 3. Launch training with checkpoint persistence
    python scripts/runpod_launch.py train --model mlp --volume-id <id>

    # 4. Monitor
    python scripts/runpod_launch.py ls

    # 5. Monitor and auto-download (incrementally, auto-terminates on completion)
    python scripts/runpod_launch.py watch <pod_id>
"""

import argparse
import base64

import os
import subprocess
import sys
from pathlib import Path

try:
    import runpod
except ImportError:
    print("ERROR: runpod SDK not installed.  Run: pip install runpod")
    sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────

REPO_URL = "https://github.com/HCAI-USP/kanprey-lm"
BRANCH = "main"

# Preference order for community GPU selection — cheapest 24 GB+ first.
# The script walks this list and picks the first available one.
GPU_PREFERENCE = [
    "NVIDIA GeForce RTX 4090",       # 24 GB ~$0.34/hr  — fastest/cheapest when available
    "NVIDIA RTX A6000",              # 48 GB ~$0.33/hr
    "NVIDIA RTX 5000 Ada Generation",# 32 GB ~$0.49/hr
    "NVIDIA L40S",                   # 48 GB ~$0.79/hr
    "NVIDIA L40",                    # 48 GB ~$0.69/hr
    "NVIDIA A40",                    # 48 GB ~$0.49/hr
    "NVIDIA RTX 6000 Ada Generation",# 48 GB ~$0.79/hr
    "NVIDIA GeForce RTX 3090",       # 24 GB ~$0.24/hr
    "NVIDIA GeForce RTX 3090 Ti",    # 24 GB ~$0.29/hr
]

DOCKER_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"

TRAIN_STEPS = 20_000
BATCH_SIZE = 16
GRAD_ACCUM = 8        # effective batch = 128
LOCAL_DEST = "checkpoints/scale"

# Persistent volume mount path inside the container
VOLUME_MOUNT = "/runpod-volume"

# Community machines known to have networking issues (SSH always refused,
# container doesn't start, or other persistent problems).
# Add machine IDs here; _launch_pod will terminate and retry automatically.
MACHINE_BLACKLIST = {
    "3z47kcltj1d0",   # RTX 5000 Ada — container never starts, SSH refused
}


# ── SSH helpers ────────────────────────────────────────────────────────────────

SSH_BASE_OPTS = [
    "ssh",
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=/tmp/ssh_mux_%r@%h:%p",
    "-o", "ControlPersist=300",
]

def _ssh_cmd(ssh_ip: str, ssh_port: int, *extra: str) -> list[str]:
    """Build an SSH command list with multiplexing options."""
    return SSH_BASE_OPTS + ["-p", str(ssh_port), f"root@{ssh_ip}"] + list(extra)

def _ssh_rsync_opt(ssh_port: int) -> str:
    """SSH option string for rsync -e, with multiplexing."""
    opts = " ".join(SSH_BASE_OPTS[1:])  # skip "ssh"
    return f"ssh {opts} -p {ssh_port}"



# ── Helpers ───────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        print("ERROR: RUNPOD_API_KEY not set.  Run: export RUNPOD_API_KEY=<your-key>")
        sys.exit(1)
    runpod.api_key = key
    return key


def get_github_token() -> str:
    tok = os.environ.get("GITHUB_TOKEN", "")
    if not tok:
        print("ERROR: GITHUB_TOKEN not set.  Run: export GITHUB_TOKEN=<your-pat>")
        sys.exit(1)
    return tok


def _try_launch_pod(
    name: str,
    gpu_type_id: str,
    startup: list[str],
    volume_id: str | None,
    disk_gb: int,
) -> dict | None:
    """Attempt to launch a pod. Returns pod dict on success, None on unavailable.
    Raises RuntimeError if the machine is blacklisted (so caller can skip to next GPU)."""
    import time

    tok = get_github_token()
    cmd = " && ".join(startup)
    kwargs = dict(
        name=name,
        image_name=DOCKER_IMAGE,
        gpu_type_id=gpu_type_id,
        cloud_type="COMMUNITY",
        container_disk_in_gb=disk_gb,
        start_ssh=True,
        support_public_ip=True,
        ports="22/tcp",
        env={"PYTHONUNBUFFERED": "1", "GITHUB_TOKEN": tok},
        docker_args=f"bash -c '{cmd}'",
    )
    if volume_id:
        kwargs["network_volume_id"] = volume_id
        kwargs["volume_mount_path"] = VOLUME_MOUNT

    pod = runpod.create_pod(**kwargs)
    pod_id = pod["id"]

    # Poll for machine ID — populated within ~30 s of launch
    machine_id = ""
    for _ in range(8):
        time.sleep(5)
        info = runpod.get_pod(pod_id)
        machine_id = info.get("machineId") or ""
        if machine_id:
            break

    if machine_id in MACHINE_BLACKLIST:
        print(f"  WARNING: Landed on blacklisted machine {machine_id} — terminating …")
        runpod.terminate_pod(pod_id)
        time.sleep(10)
        raise RuntimeError(f"blacklisted:{machine_id}")

    return pod


def find_and_launch_pod(
    name: str,
    startup: list[str],
    preferred_gpu: str | None = None,
    volume_id: str | None = None,
    disk_gb: int = 50,
) -> tuple[str, dict]:
    """Walk GPU_PREFERENCE, skip unavailable or blacklisted machines. Returns (gpu_type_id, pod)."""
    candidates = (
        [preferred_gpu] + [g for g in GPU_PREFERENCE if g != preferred_gpu]
        if preferred_gpu else GPU_PREFERENCE
    )
    print("Finding available GPU …")
    for gpu in candidates:
        try:
            pod = _try_launch_pod(name, gpu, startup, volume_id, disk_gb)
            print(f"  Selected: {gpu}")
            return gpu, pod
        except RuntimeError as e:
            if str(e).startswith("blacklisted:"):
                print(f"  {gpu}: all available instances are on a blacklisted machine — skipping")
            else:
                print(f"  {gpu}: ERROR — {e}")
        except Exception as e:
            msg = str(e)
            if "no longer any instances" in msg or "does not have the resources" in msg:
                print(f"  {gpu}: unavailable")
            else:
                print(f"  {gpu}: ERROR — {e}")
    print("ERROR: No GPU from preference list is currently available on a healthy machine.")
    sys.exit(1)


def _read_ssh_pubkey() -> str:
    """Read the local SSH public key to embed literally in the startup script."""
    for candidate in ["id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"]:
        p = Path.home() / ".ssh" / candidate
        if p.exists():
            return p.read_text().strip()
    print("WARNING: No SSH public key found in ~/.ssh/. SSH access to pods will not work.")
    print("Generate one with: ssh-keygen -t ed25519")
    return ""


def _base_startup() -> list[str]:
    """Common setup steps shared by test and train pods."""
    tok = get_github_token()
    ssh_pubkey = _read_ssh_pubkey()
    return [
        "apt-get update -q",
        "apt-get install -y git rsync openssh-server -q",
        # Set up SSH manually — docker_args overrides the container entrypoint,
        # bypassing RunPod's normal SSH injection. The PyTorch image has no sshd.
        # Key is embedded literally (no shell variable) to avoid GraphQL escaping issues.
        "ssh-keygen -A",
        "mkdir -p /root/.ssh && chmod 700 /root/.ssh",
        # base64-encode the key so it contains only [A-Za-z0-9+/=] —
        # safe for both the GraphQL string and bash single-quote context.
        f'echo {base64.b64encode(ssh_pubkey.encode()).decode()} | base64 -d > /root/.ssh/authorized_keys',
        "chmod 600 /root/.ssh/authorized_keys",
        # Write a drop-in sshd config processed BEFORE the main sshd_config.
        # Ubuntu 22.04 sshd_config has "Include /etc/ssh/sshd_config.d/*.conf" at the top,
        # so these directives are the FIRST occurrence and win over later defaults.
        # Use keyword=value (no spaces in values) to avoid quoting/GraphQL issues.
        "mkdir -p /etc/ssh/sshd_config.d",
        "echo UsePAM=no > /etc/ssh/sshd_config.d/10-docker.conf",
        "echo PermitRootLogin=yes >> /etc/ssh/sshd_config.d/10-docker.conf",
        "echo StrictModes=no >> /etc/ssh/sshd_config.d/10-docker.conf",
        "echo PubkeyAuthentication=yes >> /etc/ssh/sshd_config.d/10-docker.conf",
        # sshd needs /run/sshd for privilege separation
        "mkdir -p /run/sshd",
        # Run in foreground mode (-D) backgrounded by bash (&).
        # Plain `/usr/sbin/sshd` double-forks to daemonize, which is unreliable in Docker.
        "( /usr/sbin/sshd -D & ) && sleep 2",
        "rm -rf ~/kanprey-lm",
        f"git clone --branch {BRANCH} https://{tok}@github.com/HCAI-USP/kanprey-lm ~/kanprey-lm",
        "cd ~/kanprey-lm",
        "/opt/conda/bin/pip install tiktoken datasets -q",
    ]




# ── Subcommand: gpus ──────────────────────────────────────────────────────────

def cmd_gpus(_args):
    get_api_key()
    gpus = runpod.get_gpus()
    print(f"{'GPU':<42} {'VRAM':>6}  {'Community $/hr':>14}")
    print("-" * 70)
    for g in sorted(gpus, key=lambda x: x.get("communityPrice") or 999):
        mem = g.get("memoryInGb", "?")
        price = g.get("communityPrice", "?")
        price_str = f"${price}" if price != "?" else "n/a"
        print(f"{g['id']:<42} {str(mem)+'GB':>6}  {price_str:>14}")


# ── Subcommand: ls ────────────────────────────────────────────────────────────

def cmd_ls(_args):
    get_api_key()
    pods = runpod.get_pods()
    if not pods:
        print("No pods.")
        return
    print(f"{'Name':<30} {'ID':<20} {'GPU':<22} {'$/hr':>5}  {'Status'}")
    print("-" * 90)
    for p in pods:
        gpu = p.get("machine", {}).get("gpuDisplayName", "?")
        cost = p.get("costPerHr", "?")
        status = p.get("desiredStatus", "?")
        print(f"{p['name']:<30} {p['id']:<20} {gpu:<22} ${cost:>4}  {status}")


# ── Subcommand: stop ──────────────────────────────────────────────────────────

def cmd_stop(args):
    get_api_key()
    runpod.stop_pod(args.pod_id)
    print(f"Pod {args.pod_id} stopped (disk preserved).")


# ── Subcommand: terminate ─────────────────────────────────────────────────────

def cmd_terminate(args):
    get_api_key()
    runpod.terminate_pod(args.pod_id)
    print(f"Pod {args.pod_id} terminated.")


# ── Subcommand: volume ────────────────────────────────────────────────────────

def cmd_volume(args):
    get_api_key()

    if args.volume_action == "create":
        size = args.size
        # RunPod network volumes must be in a specific datacenter.
        # We default to EU-RO-1 which has good community GPU availability.
        # Change DATACENTER_ID if your preferred GPUs are in a different region.
        DATACENTER_ID = "EU-RO-1"
        vol = runpod.create_network_volume(
            name="kanprey-checkpoints",
            size=size,
            data_center_id=DATACENTER_ID,
        )
        vol_id = vol["id"]
        monthly_cost = size * 0.07
        print(f"Created volume: {vol_id}")
        print(f"  Size        : {size} GB")
        print(f"  Datacenter  : {DATACENTER_ID}")
        print(f"  Monthly cost: ~${monthly_cost:.2f}  (${0.07}/GB/month)")
        print(f"\nUse with:  python scripts/runpod_launch.py train --volume-id {vol_id}")
        print("NOTE: Pod must be launched in the same datacenter as the volume.")

    elif args.volume_action == "ls":
        vols = runpod.get_network_volumes()
        if not vols:
            print("No network volumes.")
            return
        print(f"{'ID':<25} {'Name':<25} {'Size':>6}  {'Datacenter'}")
        print("-" * 70)
        for v in vols:
            print(f"{v['id']:<25} {v.get('name','?'):<25} {str(v.get('size','?'))+'GB':>6}  {v.get('dataCenterId','?')}")

    elif args.volume_action == "delete":
        runpod.delete_network_volume(args.volume_id)
        print(f"Volume {args.volume_id} deleted.")


# ── Subcommand: test ──────────────────────────────────────────────────────────

def cmd_test(args):
    get_api_key()

    if args.dry_run:
        startup = _base_startup() + [
            "/opt/conda/bin/python scripts/runpod_gpu_test.py",
            "runpodctl stop pod $RUNPOD_POD_ID",
        ]
        print("Startup script (dry run):")
        print("\n".join(f"  {s}" for s in startup))
        return

    startup = _base_startup() + [
        "/opt/conda/bin/python scripts/runpod_gpu_test.py",
        # Stop regardless of test result — don't leave a broken pod running
        "runpodctl stop pod $RUNPOD_POD_ID || true",
    ]

    gpu, pod = find_and_launch_pod(
        name="kanprey-gpu-test",
        startup=startup,
        preferred_gpu=args.gpu,
        disk_gb=20,
    )
    print(f"\nLaunching test pod on {gpu} …")
    pod_id = pod["id"]
    print(f"  Pod ID  : {pod_id}")
    print(f"  Console : https://www.runpod.io/console/pods/{pod_id}")
    print()
    print("The pod will run the smoke test and stop itself (~2–3 min).")
    print("Check the pod logs in the RunPod console for PASS/FAIL output.")
    print("Once stopped, terminate with:")
    print(f"  python scripts/runpod_launch.py terminate {pod_id}")


# ── Subcommand: train ─────────────────────────────────────────────────────────

def cmd_train(args):
    get_api_key()

    if args.dry_run:
        volume_dir = f"{VOLUME_MOUNT}/{args.model}_gpt2" if args.volume_id else ""
        startup = _make_train_startup(args.model, args.steps, volume_dir)
        print("Startup script (dry run):")
        print("\n".join(f"  {s}" for s in startup))
        if args.volume_id:
            print(f"\nVolume {args.volume_id} would be mounted at {VOLUME_MOUNT}")
        return

    volume_dir = f"{VOLUME_MOUNT}/{args.model}_gpt2" if args.volume_id else ""
    startup = _make_train_startup(args.model, args.steps, volume_dir)

    step_time_s = 2.5   # conservative estimate for 32 GB GPU
    total_hours = (args.steps * step_time_s) / 3600

    print(f"\nLaunching training pod:")
    print(f"  Model   : {args.model}")
    print(f"  Steps   : {args.steps}  (effective batch = {BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM})")
    if args.volume_id:
        print(f"  Volume  : {args.volume_id} → {VOLUME_MOUNT}")
    print(f"  Est. time: ~{total_hours:.0f} h")

    gpu, pod = find_and_launch_pod(
        name=f"kanprey-lm-{args.model}",
        startup=startup,
        preferred_gpu=args.gpu,
        volume_id=args.volume_id,
    )
    print(f"  GPU     : {gpu}")
    pod_id = pod["id"]
    cost = pod.get("costPerHr", "?")
    est_cost = total_hours * float(cost) if isinstance(cost, (int, float)) else "?"

    print(f"\n  Pod ID  : {pod_id}")
    print(f"  Console : https://www.runpod.io/console/pods/{pod_id}")
    print(f"  Rate    : ${cost}/hr")
    if est_cost != "?":
        print(f"  Est. cost: ~${est_cost:.2f}")
    print()
    print("The pod will stop itself when training finishes (disk preserved).")
    print("Monitor progress in the RunPod console logs.")
    print(f"When status shows 'Exited', download results with:")
    print(f"  python scripts/runpod_launch.py download {pod_id}")
DOWNLOAD_WINDOW_HOURS = 0.25  # 15 min safety net; supervisor-driven termination is primary

def _make_train_startup(model: str, steps: int, volume_dir: str) -> list[str]:
    cmds = _base_startup() + [
        (
            f"/opt/conda/bin/python scripts/train_scale.py"
            f" --model {model}"
            f" --steps {steps}"
            f" --batch-size {BATCH_SIZE}"
            f" --grad-accum {GRAD_ACCUM}"
            f" --eval-interval 1000"
            f" --num-workers 0"
            f" --grad-checkpoint"
            f" --output-dir ~/results/{model}_gpt2"
            + (f" --volume-dir {volume_dir}" if volume_dir else "")
        ),
        # Write artifact manifest before DONE sentinel so the supervisor can verify.
        f"ls ~/results/{model}_gpt2/ > ~/results/ARTIFACTS.txt",
        "touch ~/results/DONE",
        f"echo Training complete. Pod will auto-terminate in {DOWNLOAD_WINDOW_HOURS}h.",
        f"echo Download now: python scripts/runpod_launch.py download $RUNPOD_POD_ID",
        f"sleep {int(DOWNLOAD_WINDOW_HOURS * 3600)}",
        "runpodctl terminate pod $RUNPOD_POD_ID",
    ]
    return cmds


def _make_benchmark_startup(script_b64: str) -> list[str]:
    """Startup for a short microbenchmark: embed a local script (base64), run it,
    write results + DONE sentinel, then SELF-TERMINATE.

    The self-terminate is the key safety net: even if the local watcher dies or
    is never started, the pod kills itself shortly after the benchmark finishes.
    """
    return _base_startup() + [
        "mkdir -p ~/results",
        f"echo {script_b64} | base64 -d > ~/kanprey-lm/_bench.py",
        "cd ~/kanprey-lm",
        # run; never let a failure prevent DONE+terminate (we want teardown regardless)
        "PYTHONPATH=~/kanprey-lm /opt/conda/bin/python _bench.py 2>&1 | tee ~/results/bench.log || true",
        "cp ~/kanprey-lm/m3a_cuda_latency_result.json ~/results/ 2>/dev/null || true",
        "ls ~/results/ > ~/results/ARTIFACTS.txt",
        "touch ~/results/DONE",
        "echo Benchmark complete. Pod self-terminates shortly as a safety net.",
        # Short window: the benchmark is seconds; watch normally terminates first.
        "sleep 600",
        "runpodctl terminate pod $RUNPOD_POD_ID",
    ]


def cmd_benchmark(args):
    """Launch a short microbenchmark pod that runs a local script and self-terminates."""
    get_api_key()

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}")
        sys.exit(1)
    script_b64 = base64.b64encode(script_path.read_bytes()).decode()

    startup = _make_benchmark_startup(script_b64)

    if args.dry_run:
        print("Startup script (dry run):")
        print("\n".join(f"  {s}" for s in startup))
        print(f"\n[dry-run] would launch a benchmark pod running {script_path.name}")
        print("[dry-run] pod self-terminates ~10 min after the script finishes")
        return

    gpu, pod = find_and_launch_pod(
        name="kanprey-m3a-benchmark",
        startup=startup,
        preferred_gpu=args.gpu,
        disk_gb=20,            # tiny job; image + repo only
    )
    pod_id = pod["id"]
    cost = pod.get("costPerHr", "?")
    print(f"\nLaunched benchmark pod on {gpu}")
    print(f"  Pod ID  : {pod_id}")
    print(f"  Rate    : ${cost}/hr")
    print(f"  Console : https://www.runpod.io/console/pods/{pod_id}")
    print("\nThe pod self-terminates ~10 min after the benchmark finishes (safety net).")
    print("Download results + force teardown now with:")
    print(f"  python scripts/runpod_launch.py watch {pod_id} --interval 1 --dest <dir>")



def _incremental_sync(ssh_ip: str, ssh_port: int, dest: Path, quiet: bool = False):
    """Rsync new/changed training artifacts from the pod.

    Uses two rsync passes:
    1. Everything except step_*.pt — rsync delta handles appended/overwritten files.
    2. Step checkpoints with --ignore-existing — only transfers newly-created files.

    Also syncs loose log files (~/train_*.log).
    """
    ssh_opt = _ssh_rsync_opt(ssh_port)
    remote = f"root@{ssh_ip}"

    flags = ["-az", "--progress"] if not quiet else ["-azq"]

    # Pass 1: everything except immutable step checkpoints
    # (train_log.csv is appended-to, best.pt is overwritten — rsync delta handles both)
    rc1 = subprocess.run(
        ["rsync"] + flags + ["-e", ssh_opt,
         "--exclude=*/step_*.pt",
         f"{remote}:~/results/", str(dest) + "/"]
    )
    if rc1.returncode != 0:
        print(f"WARNING: rsync pass 1 (non-step files) exited with code {rc1.returncode}")

    # Pass 2: immutable step checkpoints — only transfer new ones
    rc2 = subprocess.run(
        ["rsync"] + flags + ["--ignore-existing", "-e", ssh_opt,
         "--include=*/", "--include=*/step_*.pt", "--exclude=*",
         f"{remote}:~/results/", str(dest) + "/"]
    )
    if rc2.returncode not in (0, 23):
        print(f"WARNING: rsync pass 2 (step checkpoints) exited with code {rc2.returncode}")

    # Loose log files (non-fatal)
    rc3 = subprocess.run(
        ["rsync"] + flags + ["-e", ssh_opt,
         f"{remote}:~/train_*.log", str(dest) + "/"]
    )
    if rc3.returncode not in (0, 23):
        print(f"WARNING: log rsync exited with code {rc3.returncode} (non-fatal)")


def _verify_artifacts(ssh_ip: str, ssh_port: int, dest: Path) -> bool:
    """Check that expected training artifacts are present locally.

    Reads the ARTIFACTS.txt manifest from the pod (if available) and verifies
    each listed file exists in the local destination.
    Falls back to checking for best.pt + train_log.csv if no manifest.
    """
    remote = f"root@{ssh_ip}"
    # Try to fetch the manifest
    r = subprocess.run(
        _ssh_cmd(ssh_ip, ssh_port, "cat ~/results/ARTIFACTS.txt"),
        capture_output=True, text=True,
    )
    if r.returncode == 0 and r.stdout.strip():
        expected = [line.strip() for line in r.stdout.strip().split("\n") if line.strip()]
        missing = []
        for fname in expected:
            # Find the file in the destination directory tree
            found = list(dest.rglob(fname))
            if not found:
                missing.append(fname)
        if missing:
            print(f"WARNING: {len(missing)} artifact(s) listed in manifest not found locally:")
            for m in missing[:10]:
                print(f"  - {m}")
            return False
        print(f"All {len(expected)} artifacts verified.")
        return True
    else:
        # No manifest — basic check
        best = list(dest.rglob("best.pt"))
        log = list(dest.rglob("train_log.csv"))
        print(f"Artifact manifest not available. Found best.pt: {len(best)}, train_log.csv: {len(log)}")
        return len(best) > 0 and len(log) > 0


# ── Subcommand: download ──────────────────────────────────────────────────────

def _get_ssh_details(pod_id: str) -> tuple[str, int]:
    """Return (host, port) for SSH. Pod must be RUNNING."""
    import time, json

    pod = runpod.get_pod(pod_id)
    if not pod:
        print(f"ERROR: Pod {pod_id} not found.")
        sys.exit(1)

    desired = pod.get("desiredStatus", "")
    if desired != "RUNNING":
        print(f"ERROR: Pod is '{desired}', not RUNNING.")
        print("Training pods stay alive for 15 min after finishing — check the console.")
        sys.exit(1)

    # Poll until SSH port is assigned (can take up to 30 s after start)
    print("Getting SSH connection details ", end="", flush=True)
    ssh_port, ssh_ip = None, None
    for _ in range(12):
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []
        for p in ports:
            if p.get("privatePort") == 22:
                ssh_port = p.get("publicPort")
                ssh_ip = p.get("ip")
                break
        if ssh_port and ssh_ip:
            break
        time.sleep(5)
        pod = runpod.get_pod(pod_id)
        print(".", end="", flush=True)
    print()

    if not ssh_port or not ssh_ip:
        print(f"ERROR: Could not find SSH port for pod {pod_id}.")
        print("Pod runtime info:")
        runtime = (pod.get("runtime") or {})
        print(json.dumps(runtime, indent=2))
        sys.exit(1)

    return ssh_ip, ssh_port


def cmd_download(args):
    get_api_key()

    ssh_ip, ssh_port = _get_ssh_details(args.pod_id)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    import time

    print(f"SSH: root@{ssh_ip} (port {ssh_port})")

    # Wait for SSH daemon to be ready
    print("Waiting for SSH daemon ", end="", flush=True)
    for attempt in range(12):
        result = subprocess.run(
            _ssh_cmd(ssh_ip, ssh_port, "echo ok"),
            capture_output=True,
        )
        if result.returncode == 0:
            print(" ready.")
            break
        print(".", end="", flush=True)
        time.sleep(10)
    else:
        print("\nERROR: SSH daemon did not become ready. Pod NOT terminated.")
        print(f"Try manually: ssh -p {ssh_port} root@{ssh_ip}")
        sys.exit(1)

    # Download artifacts incrementally (handles both partial and complete results)
    print("\nDownloading artifacts …")
    _incremental_sync(ssh_ip, ssh_port, dest)

    print(f"\nResults saved to: {dest}/")

    try:
        answer = input("Terminate pod (delete disk permanently)? [y/N] ").strip().lower()
    except EOFError:
        # No stdin (e.g. called from watch in background) — auto-terminate.
        answer = "y"
        print("y  (auto-terminated — no stdin)")

    if answer == "y":
        runpod.terminate_pod(args.pod_id)
        print(f"Pod {args.pod_id} terminated.")
    else:
        print(f"Pod kept. Terminate later with:")
        print(f"  python scripts/runpod_launch.py terminate {args.pod_id}")



# ── Subcommand: watch ────────────────────────────────────────────────────────

def cmd_watch(args):
    """Poll the pod; incrementally download artifacts; auto-terminate on DONE.

    At each poll interval:
    1. rsync new/changed artifacts (train_log.csv, best.pt, new step checkpoints)
    2. Check for DONE sentinel

    When DONE is detected, performs a final incremental sync, verifies artifacts,
    and (by default) auto-terminates the pod. Use --no-auto-terminate for
    interactive prompt.
    """
    import time

    get_api_key()
    pod_id = args.pod_id
    poll_minutes = args.interval
    dest = Path(args.dest)
    auto_terminate = not getattr(args, "no_auto_terminate", False)

    print(f"Watching pod {pod_id}.")
    print(f"Incremental sync every {poll_minutes} min. Auto-terminate: {'yes' if auto_terminate else 'no'}.")
    print("Keep this terminal open. Ctrl+C to cancel.\n")

    # Wait for SSH to become available (pod may still be setting up)
    print("Waiting for SSH port ", end="", flush=True)
    ssh_ip, ssh_port = None, None
    for _ in range(60):  # up to 10 min
        try:
            pod = runpod.get_pod(pod_id)
        except Exception:
            time.sleep(10)
            continue
        if not pod or pod.get("desiredStatus") != "RUNNING":
            print(f"\nPod is no longer RUNNING (status: {pod.get('desiredStatus') if pod else 'gone'}).")
            sys.exit(1)
        ports = (pod.get("runtime") or {}).get("ports") or []
        p22 = next((p for p in ports if p.get("privatePort") == 22), None)
        if p22:
            ssh_ip, ssh_port = p22["ip"], p22["publicPort"]
            break
        print(".", end="", flush=True)
        time.sleep(10)

    if not ssh_ip:
        print("\nERROR: SSH port never appeared. Check the pod in the RunPod console.")
        sys.exit(1)

    print(f"\nSSH endpoint: {ssh_ip}:{ssh_port}")

    # Wait for sshd to accept connections
    print("Waiting for sshd ", end="", flush=True)
    for _ in range(30):
        r = subprocess.run(
            _ssh_cmd(ssh_ip, ssh_port, "echo ok"),
            capture_output=True,
        )
        if r.returncode == 0:
            print(" ready.\n")
            break
        print(".", end="", flush=True)
        time.sleep(10)
    else:
        print("\nERROR: sshd never became ready.")
        sys.exit(1)

    # Poll loop: incremental sync → check DONE → repeat
    check_cmd = "test -f ~/results/DONE && echo done || echo running"
    ssh_failures = 0

    while True:
        # Incremental sync of training artifacts before checking DONE.
        # This way, by the time DONE appears, 90%+ of bytes are already local.
        print(f"[{time.strftime('%H:%M')}] Incremental sync …")
        _incremental_sync(ssh_ip, ssh_port, dest, quiet=True)

        # Check DONE sentinel
        r = subprocess.run(
            _ssh_cmd(ssh_ip, ssh_port, check_cmd),
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            ssh_failures += 1
            if ssh_failures >= 3:
                print(f"[{time.strftime('%H:%M')}] SSH failed 3× in a row — pod may have terminated.")
                print("Partial artifacts are in:", dest)
                sys.exit(1)
            print(f"[{time.strftime('%H:%M')}] SSH check failed (attempt {ssh_failures}/3), retrying in 1 min …")
            time.sleep(60)
            continue

        ssh_failures = 0
        status = r.stdout.strip()
        if status == "done":
            print(f"\n[{time.strftime('%H:%M')}] Training finished! Final sync …")
            _incremental_sync(ssh_ip, ssh_port, dest)
            _verify_artifacts(ssh_ip, ssh_port, dest)

            if auto_terminate:
                runpod.terminate_pod(pod_id)
                print(f"Pod {pod_id} auto-terminated.")
                print(f"Artifacts: {dest}/")
            else:
                print(f"\nResults saved to: {dest}/")
                try:
                    answer = input("Terminate pod (delete disk permanently)? [y/N] ").strip().lower()
                except EOFError:
                    answer = "y"
                    print("y  (auto-terminated — no stdin)")
                if answer == "y":
                    runpod.terminate_pod(pod_id)
                    print(f"Pod {pod_id} terminated.")
                else:
                    print(f"Pod kept. Terminate later with:")
                    print(f"  python scripts/runpod_launch.py terminate {pod_id}")
            return

        print(f"[{time.strftime('%H:%M')}] Still training … next check in {poll_minutes} min.\n")
        time.sleep(poll_minutes * 60)



# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RunPod pipeline for KanpreyLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # gpus
    sub.add_parser("gpus", help="List available GPU types and prices")

    # ls
    sub.add_parser("ls", help="List running/stopped pods")

    # stop
    p_stop = sub.add_parser("stop", help="Stop a pod (preserves disk)")
    p_stop.add_argument("pod_id")

    # terminate
    p_term = sub.add_parser("terminate", help="Terminate a pod (deletes disk)")
    p_term.add_argument("pod_id")

    # volume
    p_vol = sub.add_parser("volume", help="Manage persistent network volumes")
    vol_sub = p_vol.add_subparsers(dest="volume_action", required=True)
    p_vc = vol_sub.add_parser("create", help="Create a new volume")
    p_vc.add_argument("--size", type=int, default=5, help="Size in GB (default: 5, cost: ~$0.07/GB/month)")
    p_vl = vol_sub.add_parser("ls", help="List volumes")
    p_vd = vol_sub.add_parser("delete", help="Delete a volume")
    p_vd.add_argument("volume_id")

    # test
    p_test = sub.add_parser("test", help="Run GPU smoke test pod (~2 min, ~$0.02)")
    p_test.add_argument("--gpu", default=None, help="Force a specific GPU type")
    p_test.add_argument("--dry-run", action="store_true", help="Print startup script, don't launch")

    # train
    p_train = sub.add_parser("train", help="Launch a training pod")
    p_train.add_argument("--model", choices=["mlp", "mlpedge", "mlpedge_matched"],
                         default="mlp")
    p_train.add_argument("--gpu", default=None, help="Force a specific GPU type")
    p_train.add_argument("--volume-id", default=None,
                         help="Persistent volume ID for checkpoint backup")
    p_train.add_argument("--steps", type=int, default=TRAIN_STEPS)
    p_train.add_argument("--dry-run", action="store_true",
                         help="Print startup script, don't launch")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run a short local script on a GPU pod, then self-terminate")
    p_bench.add_argument("--script", required=True, help="Path to local python script to run on the pod")
    p_bench.add_argument("--gpu", default=None, help="Force a specific GPU type")
    p_bench.add_argument("--dry-run", action="store_true", help="Print startup script, don't launch")

    # download
    p_dl = sub.add_parser("download", help="rsync results from a running pod")
    p_dl.add_argument("pod_id")
    p_dl.add_argument("--dest", default=LOCAL_DEST,
                      help=f"Local destination directory (default: {LOCAL_DEST})")

    # watch
    p_watch = sub.add_parser("watch", help="Incrementally sync artifacts and auto-terminate on completion")
    p_watch.add_argument("pod_id")
    p_watch.add_argument("--dest", default=LOCAL_DEST,
                         help=f"Local destination directory (default: {LOCAL_DEST})")
    p_watch.add_argument("--interval", type=int, default=5,
                         help="Polling interval in minutes (default: 5)")
    p_watch.add_argument("--no-auto-terminate", action="store_true",
                         help="Prompt before terminating pod (default: auto-terminate after verified download)")


    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "gpus": cmd_gpus,
        "ls": cmd_ls,
        "stop": cmd_stop,
        "terminate": cmd_terminate,
        "volume": cmd_volume,
        "test": cmd_test,
        "train": cmd_train,
        "benchmark": cmd_benchmark,
        "download": cmd_download,
        "watch": cmd_watch,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
