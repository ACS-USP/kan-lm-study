from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 4096  # overridden at runtime to match actual tokenizer vocab
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    max_seq_len: int = 128
    dropout: float = 0.1
    # KAN-FFN (B-spline basis, replaces transformer FFN)
    kan_grid_size: int = 5      # 5 = expressive; 2 = parameter-matched (~9.5M)
    kan_spline_order: int = 3

    # KAT attention (learnable kernel via KAN feature maps on Q and K)
    # Only used by KATpreyLM. Operates on head_dim (64), not d_model (384),
    # so grid can be higher without blowing up parameter count.
    kat_grid_size: int = 3
    kat_spline_order: int = 3

    # MLPEdge variant — each edge f_{i,j} is a tiny MLP (R→R) instead of a spline.
    # hidden=5 matches n_basis=5 for grid=2,order=3 splines (comparable param count).
    mlp_edge_hidden: int = 5

    # GR-KAN FFN (KAT, Yang & Wang ICLR 2025) — rational function basis with
    # group-wise parameter sharing. Replaces B-splines with a Safe Padé rational:
    #   F(x) = (a₀ + … + aₘxᵐ) / (1 + |b₁x + … + bₙxⁿ|)
    # numerator coefficients aₘ are shared across all groups;
    # denominator coefficients bₙ are per-group.
    grkan_m: int = 5          # numerator polynomial degree
    grkan_n: int = 4          # denominator polynomial degree
    grkan_groups: int = 8     # number of groups for parameter sharing
    grkan_expand: int = 4     # FFN expansion factor (hidden = d_model * expand)
    grkan_denominator: str = "abs"  # abs, softplus, square


    # Grouped function-basis KAN FFN. These variants keep the GR-KAN placement
    # (basis activation -> dense linear -> basis activation -> dense linear) but
    # swap the univariate basis family for local GuppyLM screening before any
    # GPT-2-scale Triton work.
    basis_family: str = "chebyshev"      # chebyshev, legendre, gaussian, inverse_quadratic, wendland, triangular_hat, quadratic_hat, relu_power, soft_tree
    basis_degree: int = 5                # polynomial degree for Chebyshev/Legendre
    basis_groups: int = 8                # group-shared coefficients
    basis_centers: int = 8               # centers for RBF/hat/ReLU-power families
    basis_width_scale: float = 1.5       # h = width_scale / (centers - 1)
    basis_input_norm: str = "tanh"       # none, tanh, clamp
    basis_relu_power: int = 2            # exponent for ReLU-power family
    basis_expand: int = 4                # FFN expansion factor
    # Soft regression tree basis (oblivious / NODE-style differentiable tree).
    # Each edge function is a partition-of-unity mixture over 2**depth leaves,
    # gated by learnable split thresholds with steepness `basis_tree_steepness`.
    # n_basis = 2**basis_tree_depth (depth=3 -> 8 leaves, matched to centers=8).
    basis_tree_depth: int = 3
    basis_tree_steepness: float = 1.0  # GuppyLM β-sweep optimum; β>=4 saturates gates
    # Sparse MoE (MoEGRKANpreyLM) — replaces each dense GR-KAN FFN with a sparse
    # mixture of n_moe_experts independent GR-KAN expert FFNs. Only top_k experts
    # activate per token; load_balance_coeff weights the auxiliary balancing loss.
    n_moe_experts: int = 8
    moe_top_k: int = 2
    load_balance_coeff: float = 0.01

    # Sequential unit composition (ModuleChainLM) — number of transformer blocks
    # per independently-trained unit. All units share d_model (residual stream).
    unit_n_layers: int = 3    # layers per unit; full model = n_layers (stacked units)

    # LoopGRKANpreyLM — Ouro-style looped LM (same unit applied T times) with
    # optional EqR attractor-shaping interventions (RI + NI).
    # Reference: Ouro (arXiv 2510.25741) + EqR (arXiv 2605.21488).
    loop_t_max: int = 4          # max recurrent steps; Ouro finds 4 balances stability/depth
    loop_beta: float = 0.1       # entropy regularization coefficient β (Stage I)
    loop_exit_threshold: float = 0.8  # CDF threshold q for early exit at inference
    loop_init_std: float = 0.0   # RI: std of z₀ perturbation (0 = disabled)
    loop_noise_std: float = 0.0  # NI: std of per-step additive noise (0 = disabled)
    loop_damping: float = 0.05   # NI: damping λ — weight on residual update vs carry


@dataclass
class TrainConfig:
    # Data
    dataset_name: str = "arman-bd/guppylm-60k-generic"
    tokenizer_path: str = "tokenizer.json"
    # BabyLM override (used when --dataset babylm)
    babylm_dataset_path: str = "BabyLM-community/BabyLM-2026-Strict-Small"
    babylm_tokenizer_path: str = "tokenizer_babylm.json"
    # Optimiser
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple = field(default_factory=lambda: (0.9, 0.95))
    grad_clip: float = 1.0

    # Schedule
    max_steps: int = 10_000
    warmup_steps: int = 200
    grid_update_step: int = 500   # when to call update_grid_all (after model stabilises)

    # Batching
    batch_size: int = 32

    # Logging / checkpointing
    eval_interval: int = 200
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
