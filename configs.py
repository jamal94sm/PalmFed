# ==============================================================
#  configs.py — Unified config for all methods.
#  Shared evaluation framework: open-set / closed-set (holdout / cross-spectrum)
# ==============================================================

XJTU_VARIATIONS = [
    ("iPhone", "Flash"), ("iPhone", "Nature"),
    ("huawei", "Flash"), ("huawei", "Nature"),
]
CASIA_SPECTRUMS = ["460", "630", "700", "850", "940", "WHT"]

# PSFed spectrum grouping
CASIAMS_SHORT_SPECTRA = ["460", "630", "700", "WHT"]
CASIAMS_LONG_SPECTRA  = ["850", "940"]
XJTU_SHORT_SPECTRA    = ["iPhone/Flash", "huawei/Flash"]
XJTU_LONG_SPECTRA     = ["iPhone/Nature", "huawei/Nature"]

# ══════════════════════════════════════════════════════════════
#  SHARED BASE — same for ALL methods
# ══════════════════════════════════════════════════════════════
_BASE = {
    # ── Dataset ──────────────────────────────────────────────
    "dataset"          : "casiams",
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",
    "splits_path"      : None,
    "n_ids"            : 200,
    "k_test"           : 0.20,
    "gallery_ratio"    : 0.20,
    "img_side"         : 128,

    # ── Evaluation Protocol (shared across all methods) ──────
    "eval_protocol"    : "open_set",       # open_set | closed_set
    "closed_set_mode"  : "cross_spectrum", # holdout | cross_spectrum
    "closed_set_sample_ratio": 0.20,

    # ── Model ────────────────────────────────────────────────
    "model"            : "compnet",
    "embedding_dim"    : 512,
    "dropout"          : 0.25,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,

    # ── FL ───────────────────────────────────────────────────
    "n_rounds"         : 100,
    "local_epochs"     : 1,
    "batch_size"       : 64,
    "lr"               : 0.001,
    "lr_step"          : 30,
    "lr_gamma"         : 0.8,
    "M"                : 2,
    "num_workers"      : 4,

    # ── Evaluation ───────────────────────────────────────────
    "eval_every"       : 5,

    # ── Misc ─────────────────────────────────────────────────
    "random_seed"      : 42,
}


# ══════════════════════════════════════════════════════════════
#  PROPOSED METHOD (Global / Local / MoE with Domain Predictor)
# ══════════════════════════════════════════════════════════════
CONFIG = {
    **_BASE,
    "base_results_dir" : "./rst_palmfed_{dataset}",
    "use_moe"          : False,

    # FFT augmentation
    "beta"             : 0.15,
    "local_M"          : 2,
    "local_beta"       : 0.15,

    # Domain predictor
    "dp_arch"          : "mlp",
    "dp_input"         : "style",
    "dp_pool_size"     : 16,
    "dp_hidden"        : 128,
    "dp_epochs"        : 100,
    "dp_lr"            : 1e-3,
    "dp_batch_size"    : 64,
    "dp_mode"          : "ideal",    # ideal | predicted
}


# ══════════════════════════════════════════════════════════════
#  FEDPALM BASELINE (Yang et al., TIFS 2026)
# ══════════════════════════════════════════════════════════════
CONFIG_FEDPALM = {
    **_BASE,
    "base_results_dir" : "./rst_fedpalm_{dataset}",
    "splits_path"      : "./rst_palmfed_{dataset}/splits.pkl",

    # TEIM routing
    "teim_blend_anchor" : 0.8,
    "teim_blend_side"   : 0.2,
    "teim_self_weight"  : 0.8,
    "teim_top1_weight"  : 0.1,
    "teim_top2_weight"  : 0.1,

    # Loss weights
    "w1"               : 0.8,
    "w2"               : 0.2,
    "temperature"      : 0.07,

    # Aggregation
    "fedavg_weights"   : "uniform",

    # Evaluation modes
    "eval_global"      : True,
    "eval_local_avg"   : True,
    "eval_full"        : True,
    "avg_last_rounds"  : 5,
}


# ══════════════════════════════════════════════════════════════
#  PSFED BASELINE (Yang et al., IJCV 2024)
# ══════════════════════════════════════════════════════════════
CONFIG_PSFED = {
    **_BASE,
    "base_results_dir" : "./rst_psfed_{dataset}",
    "splits_path"      : "./rst_palmfed_{dataset}/splits.pkl",
    "n_rounds"         : 100,

    # Loss weights
    "w1"               : 0.7,
    "w2"               : 0.15,
    "w3"               : 100.0,
    "mu"               : 0.01,
    "temperature"      : 0.07,

    # Evaluation modes
    "eval_global"      : True,
    "eval_local_avg"   : True,
    "avg_last_rounds"  : 5,
}


# ══════════════════════════════════════════════════════════════
#  METHOD SELECTOR
# ══════════════════════════════════════════════════════════════

def get_config(method="proposed"):
    """
    Get config dict by method name.
    All methods share the same eval_protocol / closed_set_mode.

    method: "proposed" | "fedpalm" | "psfed"
    """
    if method == "proposed":
        return CONFIG.copy()
    elif method == "fedpalm":
        return CONFIG_FEDPALM.copy()
    elif method == "psfed":
        return CONFIG_PSFED.copy()
    else:
        raise ValueError(f"Unknown method: '{method}'. "
                         f"Choose: proposed, fedpalm, psfed")
