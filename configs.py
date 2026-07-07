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

# X-Palm client domain assignments
XPALM_C1 = ["orange", "pink", "ir"]                    # scanner group A : ["red", "orange", "pink", "magenta", "ir"]
XPALM_C2 = ["green", "white", "yellow"]                          # scanner group B : ["green", "blue", "white", "yellow"]   
XPALM_C3 = ["bf", "jf", "sf", "rnd_1", "rnd_2", "rnd_3", "rnd_4", "rnd_5"]  # phone gestures
XPALM_C4 = ["close", "far", "wet", "text", "fl", "pitch", "roll"]        # phone conditions
XPALM_CLIENTS = [XPALM_C1, XPALM_C2, XPALM_C3, XPALM_C4]
XPALM_CLIENT_NAMES = ["scanner-A", "scanner-B", "phone-gesture", "phone-condition"]

# ══════════════════════════════════════════════════════════════
#  SHARED BASE
# ══════════════════════════════════════════════════════════════
_BASE = {
    "dataset"          : "casiams",
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",
    "xpalm_data_root"  : "/home/pai-ng/Jamal/xpalm",
    "n_ids"            : 200,
    "k_test"           : 0.20,
    "gallery_ratio"    : 0.20,
    "img_side"         : 128,

    "eval_protocol"    : "open_set",
    "closed_set_mode"  : "cross_spectrum",
    "closed_set_sample_ratio": 0.20,
    # local_eval_scope (closed-set only):
    #   "client": each local model on its own client's cross-spectrum data
    #   "global": each local model on the full global test set
    # open-set always uses global test set for both local and global eval
    "local_eval_scope" : "client",     # client | global

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

    # Loss: w1×CE(orig) + w2×CE(FFT-aug) + w3×SupCon + w4×anchor_align
    "w1"               : 0.5,       # CE on original
    "w2"               : 0.5,       # CE on FFT-augmented
    "w3"               : 0.0,       # SupCon on both views
    "w4"               : 0.0,       # anchor alignment
    "anchor_align"     : "mse",     # mse | supcon
    # anchor_level:
    #   feature: anchor = frozen global (resets each round)
    #   model:   anchor = EMA of global (momentum across rounds)
    "anchor_level"     : "feature", # feature | model
    "ema_beta"         : 0.996,     # EMA momentum (only if anchor_level=model)
    "temperature"      : 0.07,

    # FFT augmentation (unique to proposed)
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
