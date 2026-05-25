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
    # Both runs (use_fft_aug=True/False) load from the same files so
    # that data splits and starting weights are identical.
    # {dataset} and {model} are resolved at runtime.
    "splits_path"      : "./rst_fedavg_{dataset}/splits.pkl",
    "init_weights_path": "./rst_fedavg_{dataset}/init_weights_{model}.pth",

    # ── Dataset ────────────────────────────────────────────────
    "n_ids"            : 200,    # number of identities to sample
    "k_test"           : 0.20,   # fraction of IDs allocated to test set
    "gallery_ratio"    : 0.20,   # fraction of test-ID samples → gallery

    # ── Augmentation mode ──────────────────────────────────────
    # Exactly one of the three modes should be active:
    #
    #   use_fft_aug=False, use_mixed_aug=False → spatial aug only (baseline)
    #   use_fft_aug=True,  use_mixed_aug=False → FFT + spatial, all rounds
    #   use_fft_aug=False, use_mixed_aug=True  → spatial first, FFT second
    #
    # use_mixed_aug takes priority over use_fft_aug when both are True.
    "use_fft_aug"      : True,  # True → FFT aug for all rounds
    "use_mixed_aug"    : False,  # True → spatial → FFT switch mid-training
    "mixed_aug_round"  : 15,     # round at which to switch (mixed mode only)

    # FFT augmentation parameters
    "fft_beta"         : 0.15,   # Gaussian mask sigma as fraction of image size
    "M"                : 2,      # samples per original (1 original + M-1 FFT copies)

    # ── FL hyperparameters ─────────────────────────────────────
    "n_rounds"         : 50,    # R: total communication rounds
    "local_epochs"     : 1,      # E: local training epochs per round

    # ── Mixture of Experts — CompNet only ─────────────────────
    # When use_moe=True, the single FC(9708→512) bottleneck in CompNet
    # is replaced by MoEFC: base_FC(x) + expert[domain_id](x).
    # base_FC learns the domain-invariant projection (shared via FedAvg).
    # Each expert[d] is a low-rank residual (9708→rank→512) that learns
    # the domain-d-specific correction on top of the shared base.
    # Domain routing uses explicit domain_id labels during training;
    # at inference domain_ids=None so only base_FC is used.
    # Requires use_fft_aug=True to generate cross-domain training signal.
    "use_moe"          : False,  # True → MoEFC for CompNet
    "n_experts"        : 6,      # number of domain experts (= n_clients)
    "lora_rank"        : 64,     # expert bottleneck rank

    # CrossEntropy + ArcFace is always active for all models.
    #
    # Style Consistency Loss — CompNet and DINOv2:
    #   Enforces that FFT-augmented (or spatially-augmented) embeddings
    #   stay close to their original counterparts via cosine similarity.
    #   Loss = 1 - cosine_sim(emb_orig, emb_aug).  Range [0, 2]; 0 = perfect.
    #   No NaN risk — always has exactly one positive pair per sample.
    "lambda_style"     : 5,    # StyleConsistencyLoss weight (0.0 = disabled)

    # SupConLoss — CCNet only (between paired same-identity views):
    "ce_weight"        : 0.8,    # CE loss weight (CCNet only; CompNet/DINOv2 use 1.0)
    "con_weight"       : 0.2,    # SupConLoss weight (CCNet only)
    "temperature"      : 0.07,   # SupConLoss temperature (CCNet only)

    # CenterLoss — all models (optional):
    #   Minimises distance between embeddings and per-client class centres.
    #   Centres are kept local and carried over across rounds (never shared).
    "use_center_loss"    : False,  # True → add CenterLoss to training
    "center_loss_weight" : 0.003,  # λ_c — keep small so ArcFace dominates
    "center_loss_lr"     : 0.5,    # SGD lr for centre updates (paper default)

    # ── CompNet hyperparameters ────────────────────────────────
    "img_side"         : 128,
    "embedding_dim"    : 512,
    "dropout"          : 0.25,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,

    # ── CCNet-specific hyperparameters ─────────────────────────
    "comp_weight"      : 0.8,    # channel vs spatial competition weight

    # ── DINOv2-specific hyperparameters ───────────────────────
    # RGB 224×224 + ImageNet normalisation.
    # ViT-S/14 blocks 10 and 11 are unfrozen; all others are frozen.
    "dino_img_side"    : 224,
    "dino_weight_decay": 1e-4,   # AdamW weight decay
    "dino_margin"      : 0.3,    # ArcFace angular margin
    "dino_scale"       : 16,     # ArcFace scale

    # ── Training ───────────────────────────────────────────────
    "batch_size"       : 32,
    "lr"               : 0.001,  # constant lr across all rounds (no scheduler)

    # ── Misc ───────────────────────────────────────────────────
    "random_seed"      : 42,
    "num_workers"      : 4,
    "avg_last_rounds"  : 5,     # number of final rounds to average for reporting
}
