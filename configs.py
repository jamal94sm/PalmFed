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

    "n_ids"            : 192,
    "k_test"           : 0.20,
    "gallery_ratio"    : 0.20,
    "img_side"         : 128,

    # ── Evaluation Protocol ──────────────────────────────────
    "eval_protocol"    : "open_set",    # open_set | closed_set
    "closed_set_sample_ratio": 0.20,
    # closed_set_mode:
    #   holdout:        hold out 20% same-spectrum samples per client
    #   cross_spectrum: test on ALL other spectrums for same IDs (no holdout)
    "closed_set_mode"  : "cross_spectrum",  # holdout | cross_spectrum

    # ── Model ────────────────────────────────────────────────
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
    "M"                : 2,
    "beta"             : 0.15,
    "num_workers"      : 4,

    # ── Local Model Augmentation ─────────────────────────────
    "local_M"          : 2,
    "local_beta"       : 0.15,

    # ── Domain Predictor ─────────────────────────────────────
    "dp_arch"          : "mlp",           # mlp | cnn | transformer
    "dp_input"         : "style",         # style | full
    "dp_pool_size"     : 16,
    "dp_hidden"        : 128,
    "dp_epochs"        : 100,
    "dp_lr"            : 1e-3,
    "dp_batch_size"    : 64,

    # ── Domain Prediction Mode ───────────────────────────────
    # ideal:    oracle — uses true domain_id from test sample metadata
    # predicted: uses trained domain predictor
    "dp_mode"          : "ideal",         # ideal | predicted

    # ── Evaluation ───────────────────────────────────────────
    "eval_every"       : 5,

    # ── Misc ─────────────────────────────────────────────────
    "random_seed"      : 42,
}
