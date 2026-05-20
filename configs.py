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
    "model"            : "dinov2",  # "compnet" | "ccnet" | "dinov2"

    # ── Paths ──────────────────────────────────────────────────
    # CASIA-MS
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    # XJTU  (only used when dataset="xjtu")
    "xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",

    "base_results_dir" : "./rst_fedavg_casiams",

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
    "use_fft_aug"      : False,  # True → FFT style augmentation | False → normal augmentation 
    "use_mixed_aug"    : False,   # True → spatial first, FFT second (overrides use_fft_aug)
    "mixed_aug_round"  : 15,     # round at which to switch from spatial → FFT
    
    # ── FL hyperparameters ─────────────────────────────────────
    "n_rounds"         : 50,    # R: total communication rounds
    "local_epochs"     : 1,      # E: local training epochs per round

    # ── Center Loss ────────────────────────────────────────────
    # Minimises distance between embeddings and their class centres,
    # enforcing intra-class compactness. Centres are kept local per
    # client and carried over across rounds.
    "use_center_loss"    : False,  # True → add CenterLoss to training
    "center_loss_weight" : 1,  # λ — small so CE/ArcFace still dominates
    "center_loss_lr"     : 0.5,    # SGD lr for centre updates (paper default)

    
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
    "con_weight"       : 0.2,    # SupConLoss weight
    "temperature"      : 0.07,   # SupConLoss temperature

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
    "batch_size"       : 32,
    "lr"               : 0.001,  # constant lr across all rounds (no scheduler)

    # ── Misc ───────────────────────────────────────────────────
    "random_seed"      : 42,
    "num_workers"      : 4,
}
