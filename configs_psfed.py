# ==============================================================
#  configs_psfed.py — hyperparameters for PSFed-Palm
#  Yang et al., "Physics-Driven Spectrum-Consistent FL for
#  Palmprint Verification", IJCV 2024
#  Adapted for CASIA-MS / XJTU open-set cross-domain evaluation
# ==============================================================

# XJTU domain definitions — 4 clients, one per (smartphone, lighting) pair
XJTU_VARIATIONS = [
    ("iPhone", "Flash"),
    ("iPhone", "Nature"),
    ("huawei", "Flash"),
    ("huawei", "Nature"),
]

# ── CASIA-MS spectrum grouping ─────────────────────────────────────────────
# PSFed-Palm partitions clients into short- and long-spectrum groups.
# The anchor model for the SHORT group constrains LONG clients, and vice-versa,
# enforcing cross-spectrum feature consistency.
#
# CASIA-MS spectra by wavelength:
#   460nm  — blue, visible                           → SHORT
#   630nm  — red, visible                            → SHORT
#   700nm  — deep red, edge of visible spectrum      → SHORT
#   850nm  — near-infrared (NIR)                     → LONG
#   940nm  — near-infrared (NIR)                     → LONG
#   WHT    — broadband white light (full visible)    → SHORT
#
# Boundary: visible light ends at ~700nm; NIR begins at ~750nm.
# 700nm is kept in SHORT as it falls within the visible range.
CASIAMS_SHORT_SPECTRA = ["460", "630", "700", "WHT"]  # visible / short-wave
CASIAMS_LONG_SPECTRA  = ["850", "940"]                 # NIR / long-wave

# ── XJTU spectrum grouping ────────────────────────────────────────────────
# XJTU uses smartphone cameras (iPhone, Huawei) under Flash/Nature lighting.
# Smartphone sensors capture visible+NIR; grouping by lighting condition:
#   Flash   → controlled illumination → SHORT group
#   Nature  → ambient/natural light   → LONG group
XJTU_SHORT_SPECTRA = ["iPhone/Flash", "huawei/Flash"]
XJTU_LONG_SPECTRA  = ["iPhone/Nature", "huawei/Nature"]


CONFIG_PSFED = {
    # ── Dataset selection ──────────────────────────────────────
    "dataset"          : "casiams",

    # ── Paths ──────────────────────────────────────────────────
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",
    "base_results_dir" : "./rst_psfed_{dataset}",

    # Shared splits — same identity partitioning as main framework
    "splits_path"      : "./rst_fedavg_{dataset}/splits.pkl",

    # ── Dataset ────────────────────────────────────────────────
    "n_ids"            : 200,
    "k_test"           : 0.20,
    "gallery_ratio"    : 0.20,

    # ── Model ──────────────────────────────────────────────────
    # Uses compnet_fedpalm (same backbone as FedPalm baseline).
    "img_side"         : 128,
    "embedding_dim"    : 512,
    "dropout"          : 0.25,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,

    # ── PSFed-Palm architecture ────────────────────────────────
    # Each client maintains:
    #   local model  (θ_i) — private, never aggregated
    #   anchor model (φ_i) — aggregated within its spectrum group
    #
    # Two global anchor models:
    #   visib_net — FedAvg of SHORT-spectrum clients' models
    #   invis_net — FedAvg of LONG-spectrum  clients' models
    #
    # Cross-spectrum alignment:
    #   SHORT client → aligns with invis_net via MSE (enforces NIR consistency)
    #   LONG  client → aligns with visib_net via MSE (enforces visible consistency)
    #
    # Global server model = FedAvg of ALL local models.

    # ── Loss functions ─────────────────────────────────────────
    # PSFed-Palm total loss per client per batch:
    #   L = w1 × CE(local, y)
    #     + w2 × SupCon([fe1, fe2], y)        ← paired augmented views
    #     + w3 × MSE(fe1, anchor_fe.detach()) ← cross-spectrum alignment
    #     + mu/2 × ||θ_i - Φ||               ← FedProx to global model
    #     + mu/2 × ||θ_i - anchor||           ← FedProx to anchor model
    "w1"               : 0.7,    # CE loss weight
    "w2"               : 0.15,   # SupCon loss weight
    "w3"               : 100.0,  # MSE cross-spectrum alignment weight
    "mu"               : 0.01,   # FedProx weight (to server + to anchor)
    "temperature"      : 0.07,   # SupCon temperature

    # ── FL hyperparameters ─────────────────────────────────────
    "n_rounds"         : 100,
    "local_epochs"     : 1,
    "M"                : 2,      # augmentation multiplier (matches proposed method)

    # ── Training ───────────────────────────────────────────────
    "batch_size"       : 64,
    "lr"               : 0.001,
    "lr_step"          : 30,
    "lr_gamma"         : 0.8,

    # ── Evaluation ─────────────────────────────────────────────
    "eval_global"      : True,   # server model (FedAvg of all locals)
    "eval_local_avg"   : True,   # per-client local model, averaged metrics
    "avg_last_rounds"  : 5,

    # ── Misc ───────────────────────────────────────────────────
    "random_seed"      : 42,
    "num_workers"      : 4,
}
