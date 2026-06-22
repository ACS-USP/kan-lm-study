"""
Channel-level lesion experiment.

Hypothesis: Input channel 147 in layer 2 is a domain-selective hub
(appears in 3 of the top 5 most selective edges). Zeroing ALL 384
outgoing edges from this channel should cause asymmetric Δloss
(larger on one domain). A control channel (low mean |diff|) should
show symmetric, near-zero Δloss.

Experiment:
  1. Load KANpreyLM
  2. Identify hub channel (147) and control channel (200)
  3. Apply full channel lesion: zero spline_weight[:, in_ch, :] and base_weight[:, in_ch]
  4. Measure Δloss on 8 fish + 8 generic prompts
  5. Compare asymmetry = Δloss_fish − Δloss_generic

Saves:
  paper/figures/channel_lesion.pdf
  paper/channel_lesion_results.json

Usage:
  uv run --with . python paper/channel_lesion.py
"""

import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

N_PROMPTS = 100   # prompts per domain sampled from the test split
FISH_KEYWORDS = {
    "fish", "guppy", "tank", "aquarium", "water", "swim", "fin",
    "gill", "bubble", "scale", "pond", "float", "hungry", "feed",
    "eat", "tail", "spawn", "fry", "reef", "coral",
}

def load_domain_prompts(n=N_PROMPTS, seed=42):
    """Sample n fish-domain and n generic prompts from the test split."""
    from datasets import load_dataset
    import random
    rng = random.Random(seed)
    raw = load_dataset("arman-bd/guppylm-60k-generic", split="test")
    fish, generic = [], []
    for item in raw:
        text = item["input"].lower()
        if any(kw in text for kw in FISH_KEYWORDS):
            fish.append(item["input"])
        else:
            generic.append(item["input"])
    rng.shuffle(fish)
    rng.shuffle(generic)
    fish    = fish[:n]
    generic = generic[:n]
    print(f"  Loaded {len(fish)} fish prompts, {len(generic)} generic prompts from test split")
    return fish, generic

LAYER_IDX = 2
# Hub channel: rank 384/384 most selective (mean|diff|=0.01578 across all outputs)
HUB_CHANNEL = 147
# Second-most-selective hub for comparison
HUB2_CHANNEL = 85   # rank 383/384, mean|diff|=0.01467
# True control: rank 1/384 (mean|diff|=0.000005 — near-zero cross-domain selectivity)
CONTROL_CHANNEL = 192


def load_kan_model():
    from kanprey.config import ModelConfig
    from kanprey.model import KANpreyLM
    ckpt = torch.load(ROOT / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_cfg", ModelConfig())
    model = KANpreyLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def get_ffn_kan(model, layer_idx):
    return dict(model.named_modules())[f"blocks.{layer_idx}.ffn.kan"]


@torch.no_grad()
def prompt_loss(model, tokenizer, prompts, device):
    fmt = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    losses = []
    model.eval()
    for p in prompts:
        ids = torch.tensor(
            tokenizer.encode(fmt.format(p)).ids, dtype=torch.long, device=device
        )
        if len(ids) < 2:
            continue
        logits = model(ids.unsqueeze(0)).squeeze(0)
        loss = F.cross_entropy(logits[:-1], ids[1:])
        losses.append(loss.item())
    return float(np.mean(losses))


def apply_channel_lesion(model, layer_idx, in_ch):
    """Zero all outgoing edges from input channel in_ch."""
    m = copy.deepcopy(model)
    kan = get_ffn_kan(m, layer_idx)
    with torch.no_grad():
        kan.spline_weight[:, in_ch, :] = 0.0
        kan.base_weight[:, in_ch]      = 0.0
    return m


def channel_mean_selectivity(model, layer_idx, in_ch):
    """Mean |diff| over all output channels for a given input channel."""
    from paper.pruning_and_circuits import reconstruct_layer, compute_metrics_2d
    kan = get_ffn_kan(model, layer_idx)
    curves, x_grids = reconstruct_layer(kan)
    nls, act, _ = compute_metrics_2d(curves, x_grids)
    return float(act[:, in_ch].mean()), float(nls[:, in_ch].mean())


def run_lesion(model, tokenizer, device, in_ch, label, fish_prompts, generic_prompts):
    print(f"\n  Channel {in_ch}  ({label})")
    base_fish    = prompt_loss(model, tokenizer, fish_prompts, device)
    base_generic = prompt_loss(model, tokenizer, generic_prompts, device)
    print(f"    baseline  fish={base_fish:.4f}  generic={base_generic:.4f}")

    m_lesion = apply_channel_lesion(model, LAYER_IDX, in_ch).to(device)
    loss_fish    = prompt_loss(m_lesion, tokenizer, fish_prompts, device)
    loss_generic = prompt_loss(m_lesion, tokenizer, generic_prompts, device)
    delta_fish    = loss_fish    - base_fish
    delta_generic = loss_generic - base_generic
    asymmetry     = delta_fish - delta_generic

    print(f"    lesion    fish={loss_fish:.4f}  generic={loss_generic:.4f}")
    print(f"    Δfish={delta_fish:+.4f}  Δgen={delta_generic:+.4f}  asymmetry={asymmetry:+.4f}")

    return {
        "in_ch": in_ch,
        "label": label,
        "baseline_fish": base_fish,
        "baseline_generic": base_generic,
        "loss_fish": loss_fish,
        "loss_generic": loss_generic,
        "delta_fish": delta_fish,
        "delta_generic": delta_generic,
        "asymmetry": asymmetry,
    }


def plot_results(results):
    labels = [r["label"] for r in results]
    delta_fish    = [r["delta_fish"]    for r in results]
    delta_generic = [r["delta_generic"] for r in results]
    asymmetry     = [r["asymmetry"]     for r in results]

    x = np.arange(len(results))
    w = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Panel 1: Δloss per domain
    bars1 = ax1.bar(x - w/2, delta_fish,    w, label="Δloss fish",    color="#2980b9", alpha=0.85)
    bars2 = ax1.bar(x + w/2, delta_generic, w, label="Δloss generic", color="#e67e22", alpha=0.85)
    ax1.axhline(0, color="gray", lw=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Δ val loss (lesion − baseline)")
    ax1.set_title("Channel lesion: Δloss by domain")
    ax1.legend()

    # Panel 2: asymmetry
    colors = ["#c0392b" if "rank 384" in r["label"] else
              "#e74c3c" if "rank 383" in r["label"] else "#7f8c8d" for r in results]
    ax2.bar(x, asymmetry, color=colors, alpha=0.85)
    ax2.axhline(0, color="gray", lw=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Asymmetry (Δfish − Δgen)")
    ax2.set_title("Asymmetry confirms domain selectivity")

    fig.suptitle(
        f"Layer {LAYER_IDX} KAN FFN — Channel-level lesion\n"
        "Hub channel 147 (top circuit edges) vs. control channel",
        fontsize=10)
    fig.tight_layout()
    out = FIGURES / "channel_lesion.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")


def main():
    from kanprey.dataset import load_tokenizer
    from kanprey.train import detect_device

    device = detect_device()
    print(f"Device: {device}")

    print("Loading KANpreyLM...")
    model = load_kan_model().to(device)
    tokenizer = load_tokenizer(str(ROOT / "tokenizer.json"))

    # Print channel stats for context
    kan = get_ffn_kan(model, LAYER_IDX)
    print(f"\nLayer {LAYER_IDX} KAN shape: {kan.spline_weight.shape}")

    from paper.pruning_and_circuits import reconstruct_layer, compute_metrics_2d
    curves, x_grids = reconstruct_layer(kan)
    nls, act, _ = compute_metrics_2d(curves, x_grids)

    for ch, name in [(HUB_CHANNEL, "hub"), (CONTROL_CHANNEL, "control")]:
        mean_act = float(act[:, ch].mean())
        mean_nls = float(nls[:, ch].mean())
        print(f"  ch={ch} ({name}): mean_act={mean_act:.4f}  mean_nls={mean_nls:.4f}")

    channels = [
        (HUB_CHANNEL,  "hub ch=147 (rank 384)"),
        (HUB2_CHANNEL, "hub ch=85 (rank 383)"),
        (CONTROL_CHANNEL, "control ch=192 (rank 1)"),
    ]

    print(f"\nLoading domain prompts from test split...")
    fish_prompts, generic_prompts = load_domain_prompts()

    print(f"\nRunning channel lesions ({N_PROMPTS} prompts/domain)...")
    results = []
    for in_ch, label in channels:
        r = run_lesion(model, tokenizer, device, in_ch, label, fish_prompts, generic_prompts)
        results.append(r)

    plot_results(results)

    out_json = Path(__file__).parent / "channel_lesion_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_json}")

    print("\n── Summary ──────────────────────────────────────────────")
    print(f"{'Channel':<22} {'Δfish':>8} {'Δgen':>8} {'asymmetry':>10}")
    print("─" * 52)
    for r in results:
        print(f"{r['label']:<22} {r['delta_fish']:>+8.4f} {r['delta_generic']:>+8.4f} "
              f"{r['asymmetry']:>+10.4f}")


if __name__ == "__main__":
    main()
