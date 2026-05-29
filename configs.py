# ==============================================================
#  configs.py — single source of truth for all hyperparameters
# ==============================================================

# XJTU domain definitions — 4 clients, one per (smartphone, lighting) pair
XJTU_VARIATIONS = [
    ("iPhone", "Flash"),
    ("iPhone", "Nature"),
    ("huawei", "Flash"),
    ("huawei", "Nature"),
]

CONFIG = {
    # ── Dataset selection ──────────────────────────────────────
    # "casiams" : 6 clients, one per spectral band
    # "xjtu"    : 4 clients, one per (smartphone, lighting) domain
    "dataset"          : "casiams",

    # ── Model selection ────────────────────────────────────────
    "model"            : "compnet",  # "compnet" | "ccnet" | "dinov2"

    # ── Paths ──────────────────────────────────────────────────
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",   # CASIA-MS
    "xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",        # XJTU

    "base_results_dir" : "./rst_fedavg_casiams",

    # ── Fair comparison: shared splits and init weights ────────
    "splits_path"      : "./rst_fedavg_{dataset}/splits.pkl",
    "init_weights_path": "./rst_fedavg_{dataset}/init_weights_{model}.pth",

    # ── Dataset ────────────────────────────────────────────────
    "n_ids"            : 200,
    "k_test"           : 0.20,
    "gallery_ratio"    : 0.20,

    # ── Augmentation ───────────────────────────────────────────
    "use_fft_aug"         : True,
    "use_mixed_aug"       : False,
    "mixed_aug_round"     : 15,
    "fft_beta"            : 0.1,
    "M"                   : 2,

    # FFT augmentation method:
    #   "amplitude" — swap raw 2D amplitude in Gaussian-masked region.
    #   "radial"    — match donor's radial log-amplitude profile ring-by-ring.
    "fft_aug_method"      : "amplitude",  # "amplitude" | "radial"

    # FFT donor mode:
    #   False — cross-domain: donor templates from OTHER clients.
    #   True  — local-only: donor templates from OWN client pool only.
    "fft_local_only"      : True,

    # ── FL hyperparameters ─────────────────────────────────────
    "n_rounds"         : 30,
    "local_epochs"     : 1,

    # ── MoE ───────────────────────────────────────────────────
    #
    # Architecture
    # ────────────
    # FC layer: shared LoRA base + per-domain LoRA experts.
    #   output = B(ReLU(A(x))) + B_d(ReLU(A_d(x)))
    #   Both base and experts use _LoRABlock (A→ReLU→B, B init≈0).
    #
    # moe_position controls WHERE domain-specific parameters are applied:
    #   "fc"   — LoRA experts at the FC bottleneck only.
    #   "norm" — per-domain LayerNorm affine only.
    #   "both" — LoRA experts at FC AND per-domain LayerNorm.
    #
    # Deferred activation  (moe_warmup_round > 0)
    # ────────────────────────────────────────────
    # Phase 1 — rounds 1 … moe_warmup_round:
    #   MoE is disabled. The FC is a single shared _LoRABlock (base only)
    #   and the norm is a standard LayerNorm. All clients train together
    #   via FedAvg — the shared base learns a domain-agnostic representation.
    #
    # Phase 2 — round moe_warmup_round + 1 … n_rounds:
    #   The server calls model.activate_moe() immediately after aggregation
    #   at round moe_warmup_round. This:
    #     1. Replaces the shared _LoRABlock with MoEFC, copying trained base
    #        weights into the shared base slot AND into every expert (warm start).
    #     2. Replaces LayerNorm with MoELayerNorm (γ=1, β=0 — identity).
    #     3. Sets use_moe=True on the global model.
    #   The upgraded global model is then distributed to all clients as usual.
    #   Experts start from the trained base — not random init — so they
    #   immediately have a meaningful starting point and diverge only by the
    #   domain-specific residual they learn in subsequent rounds.
    #
    # Set moe_warmup_round = 0 to disable deferred activation (MoE from round 1).
    "use_moe"             : True,
    "moe_position"        : "norm",   # "fc" | "norm" | "both"
    "n_experts"           : 6,
    "lora_rank"           : 64,
    "moe_warmup_round"    : 5,        # 0 = MoE active from round 1
                                      # N = MoE activates after round N

    # ── GRL ───────────────────────────────────────────────────
    "use_grl"          : False,
    "lambda_grl"       : 0.05,
    "n_domains"        : 6,

    # ── Whitening ─────────────────────────────────────────────
    "use_whitening"    : False,

    # ── Losses ────────────────────────────────────────────────
    # Style Consistency Loss:
    "lambda_style"     : 1,

    # Supervised Contrastive Loss:
    "use_supcon"       : True,
    "lambda_supcon"    : 0.5,
    "temperature"      : 0.07,

    # CCNet-specific:
    "ce_weight"        : 0.8,
    "con_weight"       : 0.2,

    # Center Loss:
    "use_center_loss"    : False,
    "center_loss_weight" : 0.5,
    "center_loss_lr"     : 0.5,

    # ── CompNet hyperparameters ────────────────────────────────
    "img_side"         : 128,
    "embedding_dim"    : 512,
    "dropout"          : 0.25,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,

    # ── CCNet-specific hyperparameters ─────────────────────────
    "comp_weight"      : 0.8,

    # ── DINOv2-specific hyperparameters ───────────────────────
    "dino_img_side"    : 224,
    "dino_weight_decay": 1e-4,
    "dino_margin"      : 0.3,
    "dino_scale"       : 16,

    # ── Training ───────────────────────────────────────────────
    "batch_size"       : 32,
    "lr"               : 0.001,

    # ── Misc ───────────────────────────────────────────────────
    "random_seed"      : 42,
    "num_workers"      : 4,
    "avg_last_rounds"  : 5,
}
