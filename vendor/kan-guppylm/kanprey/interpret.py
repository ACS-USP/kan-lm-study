"""
KAN Interpretability Analysis Notebook
=======================================
Tests the interpretability claims of KAN-based LLMs against the MLP baseline.

Run with:
    uv run --with . marimo edit kanpy/interpret.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # KAN Interpretability Study

    Testing the claim: *"KANs are more interpretable than MLPs because every activation
    function is a learnable, visualizable 1D curve on an edge."*

    **Models under analysis:**
    - `KANpreyLM` — standard attention + KAN-FFN (6 × KANLinear(384→384), grid=2)
    - `KATpreyLM` — KAT attention + KAN-FFN (18 KANLinear total; adds Q/K maps at head_dim=64)
    - `Original GuppyLM` — MLP baseline (Linear→ReLU→Linear, 384→768→384)

    **Scale:** ~147K spline functions per FFN model, ~197K for KAT-Full.
    Each function is a learned 1D curve parameterized by 5–6 B-spline basis coefficients.
    """)
    return


@app.cell
def _():
    import sys, os, importlib.util, json
    import torch
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from kanprey.config import ModelConfig
    from kanprey.model import KANpreyLM, KATpreyLM
    from kanprey.kan_layers import KANLinear
    from kanprey.dataset import load_tokenizer

    DEVICE = torch.device("cpu")
    return (
        DEVICE,
        F,
        KANpreyLM,
        KANLinear,
        KATpreyLM,
        ModelConfig,
        Path,
        importlib,
        json,
        load_tokenizer,
        np,
        os,
        pd,
        plt,
        sys,
        torch,
    )


@app.cell
def _(
    DEVICE,
    KANpreyLM,
    KATpreyLM,
    ModelConfig,
    Path,
    importlib,
    json,
    os,
    sys,
    torch,
):
    def _load_kan(path, device=DEVICE):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("model_cfg", ModelConfig())
        cls = KATpreyLM if ckpt.get("model_type", "kan") == "kat" else KANpreyLM
        m = cls(cfg)
        m.load_state_dict(ckpt["model"])
        m.eval()
        return m

    def _load_orig(orig_dir, device=DEVICE):
        sys.path.insert(0, orig_dir)
        spec = importlib.util.spec_from_file_location("_cfg_orig", os.path.join(orig_dir, "config.py"))
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        spec2 = importlib.util.spec_from_file_location("_model_orig", os.path.join(orig_dir, "model.py"))
        mod_mod = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(mod_mod)
        with open(os.path.join(orig_dir, "config.json")) as f:
            cd = json.load(f)
        cfg = cfg_mod.GuppyConfig(
            vocab_size=cd.get("vocab_size", 4096),
            max_seq_len=cd.get("max_position_embeddings", 128),
            d_model=cd.get("hidden_size", 384),
            n_layers=cd.get("num_hidden_layers", 6),
            n_heads=cd.get("num_attention_heads", 6),
            ffn_hidden=cd.get("intermediate_size", 768),
            dropout=0.0,
        )
        state = torch.load(os.path.join(orig_dir, "pytorch_model.bin"), map_location=device, weights_only=False)
        m = mod_mod.KanpreyLM(cfg).to(device)
        m.load_state_dict(state, strict=False)
        m.eval()
        return m

    base_dir = Path(__file__).parent.parent
    kan_model = _load_kan(base_dir / "checkpoints/best.pt")
    kat_model = _load_kan(base_dir / "checkpoints/kat/best.pt")
    orig_model = _load_orig(str(base_dir.parent / "guppylm-original"))

    print(f"KANpreyLM:  {sum(p.numel() for p in kan_model.parameters()):,} params")
    print(f"KATpreyLM:  {sum(p.numel() for p in kat_model.parameters()):,} params")
    print(f"OrigGuppyLM: {sum(p.numel() for p in orig_model.parameters()):,} params")
    return base_dir, kan_model, kat_model, orig_model


@app.cell
def _(KANLinear, kan_model, kat_model):
    def _inventory(model, model_name):
        layers = {}
        for name, module in model.named_modules():
            if isinstance(module, KANLinear):
                layers[f"{model_name}::{name}"] = {
                    "model": model_name,
                    "path": name,
                    "module": module,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "n_basis": module.n_basis,
                    "spline_order": module.spline_order,
                    "n_functions": module.in_features * module.out_features,
                }
        return layers

    kan_inventory = _inventory(kan_model, "kan")
    kat_inventory = _inventory(kat_model, "kat")
    all_inventory = {**kan_inventory, **kat_inventory}

    print("KANpreyLM layers:")
    for _v in kan_inventory.values():
        print(f"  {_v['path']:35s}  {_v['in_features']}→{_v['out_features']}  "
              f"n_basis={_v['n_basis']}  functions={_v['n_functions']:,}")
    print(f"\nKATpreyLM layers:")
    for _v in kat_inventory.values():
        print(f"  {_v['path']:35s}  {_v['in_features']}→{_v['out_features']}  "
              f"n_basis={_v['n_basis']}  functions={_v['n_functions']:,}")
    total_kan = sum(v["n_functions"] for v in kan_inventory.values())
    total_kat = sum(v["n_functions"] for v in kat_inventory.values())
    print(f"\nTotal spline functions — KAN: {total_kan:,}  |  KAT: {total_kat:,}")
    return (all_inventory,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 1 — Function Reconstruction Engine
    """)
    return


@app.cell
def _(F, torch):
    @torch.no_grad()
    def reconstruct_layer_functions(module, n_points=200):
        """
        Returns curves (out, in, n_points), x_grids (in, n_points), x_norm (in, n_points).
        curves[j,i,:] = f_{i,j}(x) evaluated on x_grids[i,:].
        x_norm rescales each channel's domain to [0,1] for cross-channel comparison.
        """
        in_f = module.in_features
        order = module.spline_order

        x_grids = torch.zeros(in_f, n_points)
        for i in range(in_f):
            g = module.grid[i]
            x_min = g[order].item()
            x_max = g[-(order + 1)].item()
            margin = 0.05 * max(x_max - x_min, 1e-6)
            x_grids[i] = torch.linspace(x_min - margin, x_max + margin, n_points)

        bases = module.b_splines(x_grids.T)  # (n_points, in, n_basis)
        # bases: (p,i,n)  spline_weight: (o,i,n)  → target (o,i,p)
        spline_curves = torch.einsum("pin,oin->oip", bases, module.spline_weight)
        silu_x = F.silu(x_grids)  # (in, n_points)
        base_curves = module.base_weight.unsqueeze(-1) * silu_x.unsqueeze(0)  # (out, in, n_points)
        curves = (spline_curves + base_curves).numpy()

        x_min_v = x_grids.min(dim=1, keepdim=True).values
        x_max_v = x_grids.max(dim=1, keepdim=True).values
        x_norm = ((x_grids - x_min_v) / (x_max_v - x_min_v + 1e-8)).numpy()

        return curves, x_grids.numpy(), x_norm

    return (reconstruct_layer_functions,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 2 — Per-Function Statistics (Master DataFrame)
    """)
    return


@app.cell
def _(np):
    def compute_metrics_batch(curves_flat, x_norm_flat):
        """
        curves_flat: (N, P)   x_norm_flat: (N, P) — normalized x in [0,1]
        Returns dict of scalar arrays of shape (N,).
        """
        N, P = curves_flat.shape

        # Activity
        activity = np.linalg.norm(curves_flat, axis=1) / np.sqrt(P)

        # NLS: deviation from best-fit affine (uses first row's x as shared grid)
        _x = x_norm_flat[0]
        A = np.column_stack([_x, np.ones(P)])
        coeffs, _, _, _ = np.linalg.lstsq(A, curves_flat.T, rcond=None)  # (2, N)
        f_lin = (A @ coeffs).T  # (N, P)
        nls = np.linalg.norm(curves_flat - f_lin, axis=1) / (np.linalg.norm(curves_flat, axis=1) + 1e-8)

        # Roughness: mean |f''|
        dx = _x[1] - _x[0]
        df2 = np.diff(curves_flat, n=2, axis=1)
        roughness = np.mean(np.abs(df2), axis=1) / (dx**2 + 1e-12)

        # Monotonicity
        monotonicity = np.abs(np.sum(np.sign(np.diff(curves_flat, axis=1)), axis=1)) / (P - 1)

        # Symmetry (even/odd around center of [0,1])
        f_flip = curves_flat[:, ::-1].copy()
        denom = np.sqrt(np.sum(curves_flat**2, axis=1) * np.sum(f_flip**2, axis=1) + 1e-8)
        even_score = np.sum(curves_flat * f_flip, axis=1) / denom
        odd_score  = np.sum(curves_flat * -f_flip, axis=1) / denom

        return dict(activity=activity, nls=nls, roughness=roughness,
                    monotonicity=monotonicity, even_score=even_score, odd_score=odd_score)

    return (compute_metrics_batch,)


@app.cell
def _(
    all_inventory,
    compute_metrics_batch,
    mo,
    np,
    pd,
    reconstruct_layer_functions,
):
    _rows = []
    with mo.status.spinner(title="Computing per-function statistics…"):
        for _key, _info in all_inventory.items():
            _mod = _info["module"]
            _in_f, _out_f = _info["in_features"], _info["out_features"]
            _curves, _xg, _xn = reconstruct_layer_functions(_mod, n_points=200)
            _cf = _curves.reshape(_out_f * _in_f, 200)
            _xf = np.tile(_xn, (_out_f, 1, 1)).reshape(_out_f * _in_f, 200)
            _m = compute_metrics_batch(_cf, _xf)
            _oi, _ii = np.meshgrid(np.arange(_out_f), np.arange(_in_f), indexing="ij")
            for _n in range(_out_f * _in_f):
                _rows.append({
                    "model": _info["model"], "layer": _info["path"],
                    "in_ch": int(_ii.ravel()[_n]), "out_ch": int(_oi.ravel()[_n]),
                    **{k: float(v[_n]) for k, v in _m.items()},
                })

    stats_df = pd.DataFrame(_rows)
    print(f"Master DataFrame: {len(stats_df):,} rows × {len(stats_df.columns)} columns")
    print(stats_df.groupby(["model", "layer"])[["nls", "activity", "roughness"]].median().round(4).to_string())
    return (stats_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 3 — Statistical Summarization
    """)
    return


@app.cell
def _(mo, np, plt, stats_df):
    """3a. Distribution of all metrics per model."""
    _metrics = ["nls", "activity", "roughness", "monotonicity", "even_score", "odd_score"]
    _colors = {"kan": "#2196F3", "kat": "#FF5722"}
    _fig, _axes = plt.subplots(2, 3, figsize=(9, 6))
    _fig.suptitle("Per-Function Metric Distributions (all KAN layers)", fontsize=13)
    for _ax, _met in zip(_axes.flat, _metrics):
        for _mn, _col in _colors.items():
            _vals = stats_df[stats_df.model == _mn][_met].values
            _vals = _vals[np.isfinite(_vals)]
            _ax.hist(_vals, bins=80, alpha=0.6, color=_col, label=_mn, density=True)
            _ax.axvline(np.median(_vals), color=_col, linestyle="--", linewidth=1.5,
                        label=f"{_mn} med={np.median(_vals):.3f}")
        _ax.set_title(_met); _ax.set_xlabel(_met); _ax.set_ylabel("density")
        _ax.legend(fontsize=7)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo, plt, stats_df):
    """3e. NLS heatmap per FFN layer (KAN model)."""
    _ffn_layers = sorted(l for l in stats_df[stats_df.model == "kan"].layer.unique() if "ffn" in l)
    _fig, _axes = plt.subplots(2, 3, figsize=(9, 6))
    _fig.suptitle("NLS Heatmap per FFN Layer (KAN-GuppyLM)\nRows=out_ch, Cols=in_ch", fontsize=12)
    for _ax, _layer in zip(_axes.flat, _ffn_layers):
        _sub = stats_df[(stats_df.model == "kan") & (stats_df.layer == _layer)]
        _mat = _sub.pivot_table(index="out_ch", columns="in_ch", values="nls", aggfunc="first").values
        _im = _ax.imshow(_mat, cmap="hot_r", aspect="auto", vmin=0, vmax=0.5)
        _ax.set_title(".".join(_layer.split(".")[-3:-1]), fontsize=9)
        _ax.set_xlabel("in_ch"); _ax.set_ylabel("out_ch")
        plt.colorbar(_im, ax=_ax, fraction=0.03)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(all_inventory, mo, np, plt, reconstruct_layer_functions):
    """3b/3c. Coefficient-space PCA + Functional PCA on blocks.0.ffn.kan."""
    from sklearn.decomposition import PCA
    try:
        from skfda import FDataGrid
        from skfda.preprocessing.dim_reduction import FPCA as _FPCA
        _has_skfda = True
    except ImportError:
        _has_skfda = False

    _key = [k for k in all_inventory if "kan::blocks.0.ffn" in k][0]
    _info = all_inventory[_key]
    _mod = _info["module"]
    _in_f, _out_f = _info["in_features"], _info["out_features"]

    _curves, _, _ = reconstruct_layer_functions(_mod, n_points=200)
    _cf = _curves.reshape(_out_f * _in_f, 200)

    # Coefficient PCA
    _sw = _mod.spline_weight.detach().reshape(-1, _mod.n_basis).numpy()
    _bw = _mod.base_weight.detach().reshape(-1, 1).numpy()
    _X = np.concatenate([_sw, _bw], axis=1)
    _X = (_X - _X.mean(0)) / (_X.std(0) + 1e-8)
    _pca = PCA(n_components=min(6, _X.shape[1])).fit(_X)
    _cs = _pca.transform(_X)

    # Functional PCA
    _pts = np.linspace(0, 1, 200)
    if _has_skfda:
        _fd = FDataGrid(_cf, sample_points=_pts)
        _fpca = _FPCA(n_components=4).fit(_fd)
        fpca_scores = _fpca.transform(_fd)
        _fpc_curves = [c.data_matrix[0, :, 0] for c in _fpca.components_]
        _evr = getattr(_fpca, "explained_variance_ratio_", None)
    else:
        _C = _cf - _cf.mean(0)
        _U, _S, _Vt = np.linalg.svd(_C, full_matrices=False)
        fpca_scores = _U[:, :4] * _S[:4]
        _fpc_curves = [_Vt[i] for i in range(4)]
        _evr = (_S[:4]**2) / (_S**2).sum()

    _fig, _axes = plt.subplots(2, 2, figsize=(9, 6))
    _fig.suptitle("blocks.0.ffn.kan — PCA & Functional PCA", fontsize=12)

    _axes[0, 0].bar(range(1, len(_pca.explained_variance_ratio_) + 1),
                    _pca.explained_variance_ratio_, color="#2196F3")
    _axes[0, 0].set(xlabel="PC", ylabel="Explained variance ratio", title="Coefficient-space PCA (scree)")

    _axes[0, 1].scatter(_cs[:, 0], _cs[:, 1], alpha=0.01, s=1, color="#2196F3")
    _axes[0, 1].set(xlabel="Coeff PC1", ylabel="Coeff PC2", title="Coefficient PCA: PC1 vs PC2")

    for _i, _fc in enumerate(_fpc_curves):
        _lbl = f"fPC{_i+1}" + (f" ({_evr[_i]*100:.1f}%)" if _evr is not None else "")
        _axes[1, 0].plot(_pts, _fc, label=_lbl)
    _axes[1, 0].axhline(0, color="k", linewidth=0.5, linestyle="--")
    _axes[1, 0].set(xlabel="x (normalized)", ylabel="f(x)", title="Functional Principal Components")
    _axes[1, 0].legend()

    _axes[1, 1].scatter(fpca_scores[:, 0], fpca_scores[:, 1], alpha=0.01, s=1, color="#FF5722")
    _axes[1, 1].set(xlabel="fPC1 score", ylabel="fPC2 score",
                    title="fPCA score scatter (one point = one spline)")

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return (fpca_scores,)


@app.cell
def _(fpca_scores, mo, np, plt):
    """3d. K-means clustering on fPCA scores."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    _sub_idx = np.random.choice(len(fpca_scores), size=min(5000, len(fpca_scores)), replace=False)
    _sub = fpca_scores[_sub_idx, :4]
    _sils = []
    for _k in range(2, 9):
        _lbl = MiniBatchKMeans(n_clusters=_k, random_state=42, n_init=3).fit_predict(_sub)
        _sils.append(silhouette_score(_sub, _lbl))
    _best_k = 2 + int(np.argmax(_sils))
    print(f"Best k by silhouette: {_best_k}  scores: {[round(s,3) for s in _sils]}")

    _labels = MiniBatchKMeans(n_clusters=_best_k, random_state=42, n_init=5).fit_predict(
        fpca_scores[:, :4]
    )
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.scatter(fpca_scores[:, 0], fpca_scores[:, 1], c=_labels, cmap="tab10", alpha=0.05, s=1)
    _ax.set(xlabel="fPC1", ylabel="fPC2",
            title=f"Functional clusters (k={_best_k}) — blocks.0.ffn.kan")
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 4 — Pruning / Sparsity Analysis
    """)
    return


@app.cell
def _(mo):
    tau_act_ui = mo.ui.slider(0.01, 0.5, value=0.1, step=0.01,
                              label="τ_activity (fraction of layer mean)")
    tau_nls_ui = mo.ui.slider(0.01, 0.3, value=0.05, step=0.01,
                              label="τ_nls (linearity threshold)")
    mo.hstack([tau_act_ui, tau_nls_ui])
    return tau_act_ui, tau_nls_ui


@app.cell
def _(mo, np, pd, plt, stats_df, tau_act_ui, tau_nls_ui):
    _rows_sp = []
    for _mn in ["kan", "kat"]:
        for _layer in stats_df[stats_df.model == _mn].layer.unique():
            _sub = stats_df[(stats_df.model == _mn) & (stats_df.layer == _layer)]
            _mean_act = _sub.activity.mean()
            _dead = _sub.activity < tau_act_ui.value * _mean_act
            _linear = _sub.nls < tau_nls_ui.value
            _prunable = _dead | (_linear & (_sub.activity < 0.5 * _mean_act))
            _mat = _sub.pivot_table(index="out_ch", columns="in_ch", values="activity", aggfunc="first").values
            if _mat.size > 1:
                _, _s, _ = np.linalg.svd(_mat, full_matrices=False)
                _sp = _s / (_s.sum() + 1e-12)
                _eff_rank = float(np.exp(-np.sum(_sp * np.log(_sp + 1e-12))))
            else:
                _eff_rank = 1.0
            _rows_sp.append({
                "model": _mn, "layer": _layer,
                "dead_frac": float(_dead.mean()),
                "linear_frac": float(_linear.mean()),
                "prunable_frac": float(_prunable.mean()),
                "eff_rank": _eff_rank,
            })

    sparsity_df = pd.DataFrame(_rows_sp)
    print(sparsity_df[["model", "layer", "dead_frac", "linear_frac", "prunable_frac", "eff_rank"]]
          .to_string(index=False))

    _kan_rows = sparsity_df[sparsity_df.model == "kan"].sort_values("layer")
    _x = range(len(_kan_rows))
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _ax.bar(_x, _kan_rows.dead_frac, label="dead", color="#EF5350", alpha=0.8)
    _ax.bar(_x, _kan_rows.linear_frac, label="linear", color="#FFA726", alpha=0.8,
            bottom=_kan_rows.dead_frac)
    _ax.set_xticks(list(_x))
    _ax.set_xticklabels(
        [r.split("blocks.")[-1].split(".")[0] + "." + r.split(".")[-1] for r in _kan_rows.layer],
        rotation=30, ha="right", fontsize=8)
    _ax.set(ylabel="Fraction of edges",
            title="KAN-GuppyLM FFN: Dead and Linear Edge Fractions per Layer",
            ylim=(0, 1))
    _ax.legend()
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 5 — Token-Conditioned Activation Study
    """)
    return


@app.cell
def _(KANLinear, base_dir, kan_model, load_tokenizer, np, torch):
    _tokenizer = load_tokenizer(str(base_dir / "tokenizer.json"))

    _PROBES = {
        "fish": ["are you hungry", "what do you eat", "do you like your tank",
                 "what do you do all day", "what color are you", "are you lonely",
                 "do you have friends", "how are you"],
        "generic": ["what is money", "what is the internet", "goodbye", "can you talk",
                    "what is your name", "are you scared", "what is the temperature", "hello"],
    }

    def _collect(model, prompts):
        _store = {}
        def _make_hook(name):
            def _h(m, inp, _out):
                _store.setdefault(name, []).append(
                    inp[0].detach().reshape(-1, m.in_features).mean(0).numpy()
                )
            return _h
        _handles = [m.register_forward_hook(_make_hook(n))
                    for n, m in model.named_modules() if isinstance(m, KANLinear)]
        model.eval()
        with torch.no_grad():
            for _p in prompts:
                _text = f"<|im_start|>user\n{_p}<|im_end|>\n<|im_start|>assistant\n"
                _ids = _tokenizer.encode(_text).ids[:128]
                model(torch.tensor([_ids], dtype=torch.long))
        for _h in _handles:
            _h.remove()
        return {k: np.stack(v).mean(0) for k, v in _store.items()}

    fish_acts = _collect(kan_model, _PROBES["fish"])
    generic_acts = _collect(kan_model, _PROBES["generic"])
    print("Collected activations for layers:", list(fish_acts.keys()))
    return fish_acts, generic_acts


@app.cell
def _(KANLinear, fish_acts, generic_acts, kan_model, mo, np, plt):
    _ffn_names = sorted(n for n, m in kan_model.named_modules()
                        if isinstance(m, KANLinear) and "ffn" in n)
    _fig, _axes = plt.subplots(2, 3, figsize=(9, 6))
    _fig.suptitle("Domain-Selective Activations: Fish vs Generic\n"
                  "(|fish_mean − generic_mean| per input channel, sorted)", fontsize=11)
    for _ax, _ln in zip(_axes.flat, _ffn_names):
        if _ln not in fish_acts:
            _ax.set_visible(False)
            continue
        _diff = np.abs(fish_acts[_ln] - generic_acts[_ln])
        _ax.bar(range(len(_diff)), _diff[np.argsort(_diff)[::-1]], color="#4CAF50", alpha=0.8)
        _ax.set_title(_ln.split("blocks.")[-1][:20], fontsize=9)
        _ax.set_xlabel("input channel (sorted)"); _ax.set_ylabel("|Δ mean activation|")
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 6 — MLP Baseline Comparison
    """)
    return


@app.cell
def _(compute_metrics_batch, mo, np, orig_model, plt, stats_df, torch):
    # Extract W1 (768,384) and W2 (384,768) from original GuppyLM FFN
    _w1 = _w2 = None
    for _n, _m in orig_model.named_modules():
        if isinstance(_m, torch.nn.Linear):
            if _m.weight.shape == (768, 384) and _w1 is None:
                _w1 = _m.weight.detach()
            elif _m.weight.shape == (384, 768) and _w2 is None:
                _w2 = _m.weight.detach()
        if _w1 is not None and _w2 is not None:
            break

    if _w1 is not None and _w2 is not None:
        _n_pts = 200
        _in_f, _out_f = _w1.shape[1], _w2.shape[0]
        _x_range = torch.linspace(-2, 2, _n_pts)
        _mlp = torch.zeros(_out_f, _in_f, _n_pts)
        for _p, _xv in enumerate(_x_range):
            _h = torch.relu(_w1 * _xv.item())   # (hidden, in)
            _mlp[:, :, _p] = _w2 @ _h           # (out, in)

        _mlp_flat = _mlp.numpy().reshape(_out_f * _in_f, _n_pts)
        _xf = np.tile(np.linspace(0, 1, _n_pts), (_out_f * _in_f, 1))
        _mlp_met = compute_metrics_batch(_mlp_flat, _xf)

        _kan_nls = stats_df[stats_df.model == "kan"]["nls"].values
        _mlp_nls = _mlp_met["nls"]

        from scipy.stats import mannwhitneyu
        _stat, _pval = mannwhitneyu(_kan_nls, _mlp_nls, alternative="greater")
        print(f"Mann-Whitney U (KAN NLS > MLP NLS): p={_pval:.4f}")
        print(f"KAN NLS median={np.median(_kan_nls):.4f}  |  MLP NLS median={np.median(_mlp_nls):.4f}")

        _fig, _axes = plt.subplots(1, 3, figsize=(9, 5))
        _fig.suptitle(f"KAN vs MLP Edge Functions — Mann-Whitney p={_pval:.4f}", fontsize=11)
        for _ax, _met, _lbl in zip(_axes, ["nls", "activity", "roughness"],
                                   ["Nonlinearity Score", "Activity (L2)", "Roughness"]):
            _kv = stats_df[stats_df.model == "kan"][_met].values
            _mv = _mlp_met[_met]
            _vp = _ax.violinplot([_kv[np.isfinite(_kv)], _mv[np.isfinite(_mv)]],
                                 positions=[1, 2], showmedians=True)
            _vp["bodies"][0].set_facecolor("#2196F3")
            _vp["bodies"][1].set_facecolor("#FF5722")
            _ax.set_xticks([1, 2]); _ax.set_xticklabels(["KAN", "MLP"])
            _ax.set(ylabel=_lbl, title=_lbl)
        plt.tight_layout()
        _out = mo.mpl.interactive(_fig)
    else:
        _out = mo.md("*Could not extract MLP weights from original model.*")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 7 — KAT Attention Kernel Visualization
    """)
    return


@app.cell
def _(KANLinear, kat_model, mo, np, plt, torch):
    _kan_q = _kan_k = None
    for _nm, _mod in kat_model.named_modules():
        if isinstance(_mod, KANLinear) and "kan_q" in _nm and "blocks.0" in _nm:
            _kan_q = _mod
        if isinstance(_mod, KANLinear) and "kan_k" in _nm and "blocks.0" in _nm:
            _kan_k = _mod
        if _kan_q and _kan_k:
            break

    if _kan_q is None:
        _out = mo.md("*KAT Q/K maps not found.*")
    else:
        _g = np.linspace(-2, 2, 50)
        _G1, _G2 = np.meshgrid(_g, _g)
        _vecs = np.zeros((2500, _kan_q.in_features), dtype=np.float32)
        _vecs[:, 0] = _G1.ravel(); _vecs[:, 1] = _G2.ravel()

        with torch.no_grad():
            _phi_q = _kan_q(torch.tensor(_vecs))
            _phi_k = _kan_k(torch.tensor(_vecs))

        _k_kan = (_phi_q * _phi_k).sum(1).numpy().reshape(50, 50)
        _k_lin = _G1**2 + _G2**2

        # PSD check
        _x_rand = torch.randn(500, _kan_q.in_features) * 0.5
        with torch.no_grad():
            _pq = _kan_q(_x_rand); _pk = _kan_k(_x_rand)
        _K = (_pq @ _pk.T).numpy()
        _neg_frac = float((np.linalg.eigvalsh(_K) < 0).mean())
        print(f"KAT kernel PSD check — negative eigenvalue fraction: {_neg_frac:.3f}")

        _fig, _axes = plt.subplots(1, 3, figsize=(9, 5))
        _fig.suptitle(f"KAT Attention Kernel (block 0, dims 0 & 1)\n"
                      f"PSD: {_neg_frac*100:.1f}% negative eigenvalues", fontsize=11)
        _im0 = _axes[0].contourf(_G1, _G2, _k_lin, levels=20, cmap="RdBu_r")
        _axes[0].set_title("Linear kernel K=q·k")
        plt.colorbar(_im0, ax=_axes[0])
        _im1 = _axes[1].contourf(_G1, _G2, _k_kan, levels=20, cmap="RdBu_r")
        _axes[1].set_title("Learned KAN kernel K=φ(q)·φ(k)")
        plt.colorbar(_im1, ax=_axes[1])
        _diff = _k_kan - _k_lin / (np.abs(_k_lin).max() + 1e-8) * np.abs(_k_kan).max()
        _im2 = _axes[2].contourf(_G1, _G2, _diff, levels=20, cmap="PiYG")
        _axes[2].set_title("Difference (KAN − scaled linear)")
        plt.colorbar(_im2, ax=_axes[2])
        for _ax in _axes:
            _ax.set_xlabel("dim 0"); _ax.set_ylabel("dim 1")
        plt.tight_layout()
        _out = mo.mpl.interactive(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 8 — Interactive Spline Explorer

    Select any layer and (input, output) channel pair to see the exact learned 1D function.
    """)
    return


@app.cell
def _(all_inventory, mo):
    layer_keys = sorted(all_inventory.keys())
    layer_select = mo.ui.dropdown(
        options={k: k for k in layer_keys},
        value=layer_keys[0],
        label="Layer",
    )
    mo.hstack([layer_select])
    return (layer_select,)


@app.cell
def _(all_inventory, layer_select, mo):
    _info_sel = all_inventory[layer_select.value]
    in_ch_slider = mo.ui.slider(0, _info_sel["in_features"] - 1, value=0, label="Input channel i")
    out_ch_slider = mo.ui.slider(0, _info_sel["out_features"] - 1, value=0, label="Output channel j")
    mo.hstack([in_ch_slider, out_ch_slider])
    return in_ch_slider, out_ch_slider


@app.cell
def _(
    all_inventory,
    in_ch_slider,
    layer_select,
    mo,
    np,
    out_ch_slider,
    plt,
    reconstruct_layer_functions,
    stats_df,
):
    _key = layer_select.value
    _info = all_inventory[_key]
    _i = in_ch_slider.value
    _j = out_ch_slider.value

    _curves, _xg, _ = reconstruct_layer_functions(_info["module"], n_points=300)
    _f = _curves[_j, _i, :]
    _x = _xg[_i, :]

    _row = stats_df[
        (stats_df.model == _info["model"]) & (stats_df.layer == _info["path"]) &
        (stats_df.in_ch == _i) & (stats_df.out_ch == _j)
    ]

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.plot(_x, _f, color="#1565C0", linewidth=2, label=f"f_{{{_i},{_j}}}(x)")
    _ax.axhline(0, color="k", linewidth=0.5, linestyle="--")

    _A = np.column_stack([_x, np.ones_like(_x)])
    _c, _, _, _ = np.linalg.lstsq(_A, _f, rcond=None)
    _ax.plot(_x, _A @ _c, color="#EF9A9A", linewidth=1.5, linestyle="--", label="linear fit")

    if not _row.empty:
        _r = _row.iloc[0]
        _ax.set_title(
            f"{_info['path']}  →  in={_i}, out={_j}\n"
            f"NLS={_r.nls:.4f}  Activity={_r.activity:.4f}  "
            f"Roughness={_r.roughness:.4f}  Monotone={_r.monotonicity:.3f}",
            fontsize=9
        )
    _ax.set_xlabel("x (activation value)"); _ax.set_ylabel("f(x)")
    _ax.legend()
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
