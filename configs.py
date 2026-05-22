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
    # CASIA-MS
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    # XJTU  (only used when dataset="xjtu")
    "xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",

    "base_results_dir" : "./rst_fedavg_{dataset}_{model}",

    # ── Fair comparison: shared splits and init weights ────────
    # Both runs (use_fft_aug=True and False) must load from the
    # same files so that data splits and starting weights are
    # identical. Generated automatically on the first run.
    # {dataset} and {model} are filled in at runtime so each
    # combination gets its own file.
    "splits_path"      : "./rst_fedavg_{dataset}/splits.pkl",
    "init_weights_path": "./rst_fedavg_{dataset}/init_weights_{model}.pth",

    # ── Dataset ────────────────────────────────────────────────
    "n_ids"            : 200,    # number of identities to sample
    "k_test"           : 0.20,   # fraction of IDs allocated to test set
    "gallery_ratio"    : 0.20,   # fraction of test-ID samples → gallery

    # ── FFT style augmentation ─────────────────────────────────
    "fft_beta"         : 0.1,   # Gaussian mask sigma as fraction of image size
    "M"                : 2,      # augmented copies per sample (1 original + M-1 synthetic)

    # Augmentation mode — exactly one should be True (or both False for spatial-only):
    #   use_fft_aug=False, use_mixed_aug=False → spatial augmentation only
    #   use_fft_aug=True,  use_mixed_aug=False → FFT + spatial all rounds
    #   use_fft_aug=False, use_mixed_aug=True  → spatial first, FFT second
    #   (use_mixed_aug takes priority over use_fft_aug when both are True)
    "use_fft_aug"      : True,  # True → FFT aug for all rounds
    "use_mixed_aug"    : False,  # True → spatial first, then FFT (overrides use_fft_aug)
    "mixed_aug_round"  : 15,     # round at which to switch spatial → FFT (mixed mode only)

    # ── Domain distance-aware mixing ───────────────────────────
    # Only used when use_fft_aug=True or use_mixed_aug=True.
    "domain_aware_mixing" : False,  # True → distance-based donor selection
    "prefer_distant_domain": False,  # True → most different domain (max L2)
                                     # False → most similar domain (min L2)
    "use_mean_template"   : True,  # True → use donor's mean template
                                     # False → random sample from donor's bank

    # ── Mixture of Experts (CompNet and DINOv2) ───────────────
    # CompNet:  MoEFC replaces the single FC(9708→512) bottleneck.
    #   Each expert is a low-rank 2-layer projection (9708→rank→512).
    #   The gate reads the 9708-d multi-scale Gabor feature vector and
    #   routes each sample to its top-k experts via soft top-k gating.
    #   Best position: the FC bottleneck where all three Gabor scales merge.
    #
    # DINOv2:   LoRA MoE adapters inside transformer blocks 10-11.
    #   Each expert is a low-rank LoRA adapter injected into fc1.
    #
    # When use_moe=True, FFT augmentation is changed for both models:
    #   M is auto-set to n_clients so each sample gets one synthetic copy
    #   per other domain (deterministic), ensuring systematic cross-domain
    #   coverage in every epoch. Requires use_fft_aug or use_mixed_aug.
    #
    # share_moe toggle:
    #   True  (recommended) → MoE gate + experts shared via FedAvg.
    #     Gate learns universal domain routing from all clients.
    #     Experts specialise per domain through cross-client gradients.
    #   False → MoE kept local per client (not shared).
    #     Gate only sees one domain → cannot learn universal routing.
    #     Degenerates to a domain-specific projection — defeats the purpose.
    "use_moe"            : True,  # True → MoE FC for compnet | LoRA MoE for dinov2
    "n_experts"          : 6,      # number of experts (= number of FL domains)
    "lora_rank"          : 64,     # expert bottleneck rank (64 for compnet, 16 for dinov2)
    "moe_top_k"          : 2,      # top-k active experts per sample
    "share_moe"          : False,   # True → FedAvg MoE | False → keep local
    "lambda_load_balance": 0.1,    # load balancing loss weight
    # CrossEntropy + ArcFace is always active for all models.

    
    # ── FL hyperparameters ─────────────────────────────────────
    "n_rounds"         : 50,    # R: total communication rounds
    "local_epochs"     : 1,      # E: local training epochs per round

    # ── CompNet hyperparameters ────────────────────────────────
    "img_side"         : 128,
    "embedding_dim"    : 512,    # CompNet FC output dim
    "dropout"          : 0.25,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,

    # ── CCNet-specific hyperparameters ─────────────────────────
    # (only used when model="ccnet")
    "comp_weight"      : 0.8,    # channel vs spatial competition weight
    "ce_weight"        : 0.8,    # CrossEntropy loss weight
    "con_weight"       : 0.2,    # SupConLoss weight  (CCNet, CompNet, DINOv2)
    "temperature"      : 0.07,   # SupConLoss temperature (all models)

    # ── Center Loss ────────────────────────────────────────────
    # Minimises distance between embeddings and their class centres,
    # enforcing intra-class compactness. Centres are kept local per
    # client and carried over across rounds.
    "use_center_loss"    : False,  # True → add CenterLoss to training
    "center_loss_weight" : 0.003,  # λ — small so CE/ArcFace still dominates
    "center_loss_lr"     : 0.5,    # SGD lr for centre updates (paper default)

    
    # ── DINOv2-specific hyperparameters ───────────────────────
    # (only used when model="dinov2")
    # DINOv2 uses RGB 224×224 with ImageNet normalisation.
    # Blocks 10 and 11 of ViT-S/14 are unfrozen; the rest are frozen.
    "dino_img_side"    : 224,    # DINOv2 input resolution
    "dino_lamb"        : 0.2,    # SupConLoss weight (arc + lamb*supcon)
    "dino_weight_decay": 1e-4,   # AdamW weight decay
    "dino_margin"      : 0.3,    # ArcFace angular margin
    "dino_scale"       : 16,     # ArcFace scale (lower than CompNet — RGB embeds)

    # ── Training ───────────────────────────────────────────────
    "batch_size"       : 64,
    "lr"               : 0.001,  # constant lr across all rounds (no scheduler)

    # ── Misc ───────────────────────────────────────────────────
    "random_seed"      : 42,
    "num_workers"      : 4,
    "avg_last_rounds"  : 5,     # number of final rounds to average for reporting
}
