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
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",
    "base_results_dir" : "./rst_fedavg_casiams",
    "splits_path"      : "./rst_fedavg_{dataset}/splits.pkl",
    "init_weights_path": "./rst_fedavg_{dataset}/init_weights_{model}.pth",

    # ── Dataset ────────────────────────────────────────────────
    "n_ids"            : 200,
    "k_test"           : 0.20,
    "gallery_ratio"    : 0.20,

    # ── Augmentation ───────────────────────────────────────────
    "use_fft_aug"      : True,
    "use_mixed_aug"    : False,
    "mixed_aug_round"  : 15,
    "fft_beta"         : 0.1,
    "M"                : 2,
    "fft_aug_method"   : "amplitude",  # "amplitude" | "radial"
    "fft_local_only"   : True,

    # ── FL hyperparameters ─────────────────────────────────────
    "n_rounds"         : 30,
    "local_epochs"     : 1,

    # ── MoE — MultiExpertCompNet ───────────────────────────────
    #
    # Architecture
    # ────────────
    # use_moe=True  → MultiExpertCompNet:
    #   experts[0]       — shared base CompNet (always active)
    #   experts[1..N]    — one domain-specific CompNet per domain
    #   N = n_domains    (6 for CASIA-MS, 4 for XJTU)
    #
    # Forward (train & eval):
    #   logits = 0.5 × base_expert(x) + 0.5 × domain_expert[d](x)
    #
    # Embedding (gallery/probe):
    #   emb = L2_norm(0.5 × base_emb + 0.5 × domain_emb)
    #   domain label of the test sample used to select domain_expert[d]
    #
    # use_moe=False → plain single CompNet (baseline)
    #
    # Deferred activation  (moe_warmup_round > 0)
    # ────────────────────────────────────────────
    # Phase 1 — rounds 1 … moe_warmup_round:
    #   Only experts[0] (base) is constructed and trained.
    #   All clients share a single CompNet via FedAvg.
    #   The base learns a domain-agnostic representation.
    #
    # Phase 2 — round moe_warmup_round + 1 … n_rounds:
    #   Server calls activate_moe() immediately after aggregation at
    #   round moe_warmup_round.  This:
    #     1. Copies trained base weights into every domain expert
    #        (warm start — not random init).
    #     2. Sets use_moe=True on global model and all clients.
    #   All (1 + n_domains) CompNets are then trained and aggregated
    #   via FedAvg in every subsequent round.
    #
    # Set moe_warmup_round = 0 to skip warmup (MoE active from round 1).
    #
    # FedAvg scope
    # ────────────
    # All CompNet backbone weights shared (base + all domain experts).
    # arc.* heads of every expert stay local (never sent to server).
    "use_moe"             : True,
    "n_domains"           : 6,        # must match dataset: 6 CASIA-MS, 4 XJTU
    "moe_warmup_round"    : 10,       # 0 = full MoE from round 1

    # ── GRL ───────────────────────────────────────────────────
    "use_grl"          : False,
    "lambda_grl"       : 0.05,

    # ── Whitening ─────────────────────────────────────────────
    "use_whitening"    : False,

    # ── Losses ────────────────────────────────────────────────
    "lambda_style"     : 1.0,

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

    # ── CCNet-specific ─────────────────────────────────────────
    "comp_weight"      : 0.8,

    # ── DINOv2-specific ────────────────────────────────────────
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
