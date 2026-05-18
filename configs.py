# ==============================================================
#  configs.py — single source of truth for all hyperparameters
# ==============================================================

CONFIG = {
    # ── Model selection ────────────────────────────────────────
    "model"            : "compnet",  # "compnet" | "ccnet"

    # ── Paths ──────────────────────────────────────────────────
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "base_results_dir" : "./rst_fedavg_casiams",

    # ── Fair comparison: shared splits and init weights ────────
    # Both runs (use_fft_aug=True and False) must load from the
    # same files so that data splits and starting weights are
    # identical. Generated automatically on the first run.
    "splits_path"      : "./rst_fedavg_casiams/splits.pkl",
    "init_weights_path": "./rst_fedavg_casiams/init_weights.pth",

    # ── Dataset ────────────────────────────────────────────────
    "n_ids"            : 200,    # number of identities to sample
    "k_test"           : 0.20,   # fraction of IDs allocated to test set
    "gallery_ratio"    : 0.20,   # fraction of test-ID samples → gallery

    # ── FFT style augmentation ─────────────────────────────────
    "fft_beta"         : 0.15,   # Gaussian mask sigma as fraction of image size
    "M"                : 2,      # augmented copies per sample (1 original + M-1 synthetic)
    "use_fft_aug"      : False,  # True → FFT style augmentation | False → standard training

    # ── FL hyperparameters ─────────────────────────────────────
    "n_rounds"         : 30,    # R: total communication rounds
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
    "con_weight"       : 0.2,    # SupConLoss weight
    "temperature"      : 0.07,   # SupConLoss temperature

    # ── Training ───────────────────────────────────────────────
    "batch_size"       : 32,
    "lr"               : 0.001,  # constant lr across all rounds (no scheduler)

    # ── Misc ───────────────────────────────────────────────────
    "random_seed"      : 42,
    "num_workers"      : 4,
}
