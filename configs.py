# ==============================================================
#  configs.py — Unified config for all methods.
# ==============================================================

XJTU_VARIATIONS = [
    ("iPhone", "Flash"), ("iPhone", "Nature"),
    ("huawei", "Flash"), ("huawei", "Nature"),
]
CASIA_SPECTRUMS = ["460", "630", "700", "850", "940", "WHT"]

CASIAMS_SHORT_SPECTRA = ["460", "630", "700", "WHT"]
CASIAMS_LONG_SPECTRA  = ["850", "940"]
XJTU_SHORT_SPECTRA    = ["iPhone/Flash", "huawei/Flash"]
XJTU_LONG_SPECTRA     = ["iPhone/Nature", "huawei/Nature"]

# ══════════════════════════════════════════════════════════════
#  SHARED BASE
# ══════════════════════════════════════════════════════════════
_BASE = {
    "dataset"          : "casiams",
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",
    "n_ids"            : 200,
    "k_test"           : 0.20,
    "gallery_ratio"    : 0.20,
    "img_side"         : 128,

    "eval_protocol"    : "open_set",
    "closed_set_mode"  : "cross_spectrum",
    "closed_set_sample_ratio": 0.20,

    "model"            : "compnet",
    "embedding_dim"    : 512,
    "dropout"          : 0.25,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,

    "n_rounds"         : 100,
    "local_epochs"     : 1,
    "batch_size"       : 64,
    "lr"               : 0.001,
    "lr_step"          : 30,
    "lr_gamma"         : 0.8,
    "M"                : 2,
    "num_workers"      : 4,

    "eval_every"       : 10,
    "random_seed"      : 42,

    # Splits path: None = generate fresh, or path to pkl
    # When set, ALL methods load from this EXACT path
    "splits_path"      : None,
}


# ══════════════════════════════════════════════════════════════
#  PROPOSED
# ══════════════════════════════════════════════════════════════
CONFIG = {
    **_BASE,
    "base_results_dir" : "./rst_proposed_{dataset}_{eval_protocol}",
    "use_moe"          : False,

    "beta"             : 0.15,
    "local_M"          : 2,
    "local_beta"       : 0.15,

    "dp_arch"          : "mlp",
    "dp_input"         : "style",
    "dp_pool_size"     : 16,
    "dp_hidden"        : 128,
    "dp_epochs"        : 100,
    "dp_lr"            : 1e-3,
    "dp_batch_size"    : 64,
    "dp_mode"          : "ideal",
}


# ══════════════════════════════════════════════════════════════
#  FEDPALM
# ══════════════════════════════════════════════════════════════
CONFIG_FEDPALM = {
    **_BASE,
    "base_results_dir" : "./rst_fedpalm_{dataset}_{eval_protocol}",

    "teim_blend_anchor" : 0.8,
    "teim_blend_side"   : 0.2,
    "teim_self_weight"  : 0.8,
    "teim_top1_weight"  : 0.1,
    "teim_top2_weight"  : 0.1,

    "w1"               : 0.8,
    "w2"               : 0.2,
    "temperature"      : 0.07,

    "fedavg_weights"   : "uniform",

    "eval_global"      : True,
    "eval_local_avg"   : True,
    "eval_full"        : True,
    "avg_last_rounds"  : 5,
}


# ══════════════════════════════════════════════════════════════
#  PSFED
# ══════════════════════════════════════════════════════════════
CONFIG_PSFED = {
    **_BASE,
    "base_results_dir" : "./rst_psfed_{dataset}_{eval_protocol}",

    "w1"               : 0.7,
    "w2"               : 0.15,
    "w3"               : 100.0,
    "mu"               : 0.01,
    "temperature"      : 0.07,

    "eval_global"      : True,
    "eval_local_avg"   : True,
    "avg_last_rounds"  : 5,
}


# ══════════════════════════════════════════════════════════════
#  METHOD SELECTOR
# ══════════════════════════════════════════════════════════════

def get_config(method="proposed"):
    if method == "proposed":
        return CONFIG.copy()
    elif method == "fedpalm":
        return CONFIG_FEDPALM.copy()
    elif method == "psfed":
        return CONFIG_PSFED.copy()
    else:
        raise ValueError(f"Unknown method: '{method}'")
