# ==============================================================
#  configs.py — Configuration for Federated Palmprint + Domain Predictor
# ==============================================================

XJTU_VARIATIONS = [
    ("iPhone", "Flash"), ("iPhone", "Nature"),
    ("huawei", "Flash"), ("huawei", "Nature"),
]

CASIA_SPECTRUMS = ["460", "630", "700", "850", "940", "WHT"]

CONFIG = {
    # ── Dataset ──────────────────────────────────────────────
    "dataset"          : "casiams",       # casiams | xjtu
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",
    "base_results_dir" : "./rst_palmfed_{dataset}",
    "splits_path"      : None,            # load existing splits (or None)

    "n_ids"            : 200,
    "k_test"           : 0.20,            # fraction of IDs → test set
    "gallery_ratio"    : 0.20,            # fraction of test samples → gallery
    "img_side"         : 128,

    # ── Model (standard CompNet, no MoE) ─────────────────────
    "model"            : "compnet",       # compnet | ccnet | dinov2
    "embedding_dim"    : 512,
    "dropout"          : 0.25,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,
    "use_moe"          : False,           # disabled — standard CompNet

    # ── Federated Learning ───────────────────────────────────
    "n_rounds"         : 50,
    "local_epochs"     : 1,
    "batch_size"       : 64,
    "lr"               : 0.001,
    "lr_step"          : 30,
    "lr_gamma"         : 0.8,
    "M"                : 2,               # FFT augmentation multiplier
    "beta"             : 0.15,            # FFT swap intensity
    "num_workers"      : 4,

    # ── Domain Predictor ─────────────────────────────────────
    "dp_arch"          : "mlp",           # mlp | cnn | transformer
    "dp_input"         : "style",         # style (low-freq only) | full (all FFT amp)
    "dp_pool_size"     : 16,              # pool FFT amp to this spatial size
    "dp_hidden"        : 128,             # hidden dimension
    "dp_epochs"        : 100,             # training epochs
    "dp_lr"            : 1e-3,
    "dp_batch_size"    : 64,

    # ── Test Domain Toggle ───────────────────────────────────
    # Controls what domain(s) each client sees at test time:
    #   "same"  → test from same domain as training (intra-domain)
    #   "cross" → test from all OTHER domains (cross-domain)
    #   "all"   → test from ALL domains (combined)
    "test_domain"      : "cross",

    # ── Routing Mode ─────────────────────────────────────────
    # Controls how personalized + generalized models are combined:
    #   "soft"      → α·personal + (1-α)·general (domain predictor)
    #   "personal"  → personal model only (baseline)
    #   "general"   → general model only (baseline)
    "routing_mode"     : "soft",

    # ── Evaluation ───────────────────────────────────────────
    "eval_every"       : 5,               # evaluate every N rounds
    "eval_baselines"   : True,            # always show personal/general baselines

    # ── Misc ─────────────────────────────────────────────────
    "random_seed"      : 42,
}
