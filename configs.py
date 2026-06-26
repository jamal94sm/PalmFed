# ==============================================================
#  configs.py — Federated Palmprint + Domain Predictor
# ==============================================================

XJTU_VARIATIONS = [
    ("iPhone", "Flash"), ("iPhone", "Nature"),
    ("huawei", "Flash"), ("huawei", "Nature"),
]

CASIA_SPECTRUMS = ["460", "630", "700", "850", "940", "WHT"]

CONFIG = {
    # ── Dataset ──────────────────────────────────────────────
    "dataset"          : "casiams",
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",
    "base_results_dir" : "./rst_palmfed_{dataset}",
    "splits_path"      : None,

    "n_ids"            : 200,
    "k_test"           : 0.20,
    "gallery_ratio"    : 0.20,
    "img_side"         : 128,

    # ── Evaluation Protocol ──────────────────────────────────
    # open_set:   test IDs ≠ train IDs (disjoint)
    # closed_set: test IDs = train IDs (held-out samples)
    "eval_protocol"    : "open_set",    # open_set | closed_set
    "closed_set_sample_ratio": 0.20,    # fraction of samples held out (closed-set)

    # ── Model (standard CompNet, no MoE) ─────────────────────
    "model"            : "compnet",
    "embedding_dim"    : 512,
    "dropout"          : 0.25,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,
    "use_moe"          : False,

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

    # ── Personal Model Augmentation ──────────────────────────
    # Personal model uses local-only FFT: swap amplitude among
    # samples within the same client (no cross-client knowledge).
    "personal_M"       : 2,              # local FFT aug multiplier
    "personal_beta"    : 0.15,           # local FFT swap intensity

    # ── Domain Predictor ─────────────────────────────────────
    "dp_arch"          : "mlp",           # mlp | cnn | transformer
    "dp_input"         : "style",         # style (low-freq) | full (all FFT amp)
    "dp_pool_size"     : 16,
    "dp_hidden"        : 128,
    "dp_epochs"        : 100,
    "dp_lr"            : 1e-3,
    "dp_batch_size"    : 64,

    # ── Test Domain Toggle ───────────────────────────────────
    "test_domain"      : "cross",         # same | cross | all

    # ── Routing Mode ─────────────────────────────────────────
    "routing_mode"     : "soft",          # soft | personal | general
    "eval_baselines"   : True,            # always show P/G baselines

    # ── Evaluation ───────────────────────────────────────────
    "eval_every"       : 5,

    # ── Misc ─────────────────────────────────────────────────
    "random_seed"      : 42,
}
