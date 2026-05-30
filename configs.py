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
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",   # CASIA-MS
    "xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",        # XJTU

    "base_results_dir" : "./rst_fedavg_casiams",

    # ── Fair comparison: shared splits and init weights ────────
    # Both runs (use_fft_aug=True/False) load from the same files so
    # that data splits and starting weights are identical.
    # {dataset} and {model} are resolved at runtime.
    "splits_path"      : "./rst_fedavg_{dataset}/splits.pkl",
    "init_weights_path": "./rst_fedavg_{dataset}/init_weights_{model}.pth",

    # ── Dataset ────────────────────────────────────────────────
    "n_ids"            : 200,    # number of identities to sample
    "k_test"           : 0.20,   # fraction of IDs allocated to test set
    "gallery_ratio"    : 0.20,   # fraction of test-ID samples → gallery

    # ── Augmentation mode ──────────────────────────────────────
    # Exactly one of the three modes should be active:
    #
    #   use_fft_aug=False, use_mixed_aug=False → spatial aug only (baseline)
    #   use_fft_aug=True,  use_mixed_aug=False → FFT + spatial, all rounds
    #   use_fft_aug=False, use_mixed_aug=True  → spatial first, FFT second
    #
    # use_mixed_aug takes priority over use_fft_aug when both are True.
    "use_fft_aug"      : True,  # True → FFT aug for all rounds
    "use_mixed_aug"    : False,  # True → spatial → FFT switch mid-training
    "mixed_aug_round"  : 10,     # round at which to switch (mixed mode only)

    # FFT augmentation parameters
    "fft_beta"         : 0.15,   # Gaussian mask sigma as fraction of image size
    "M"                : 2,      # samples per original (1 original + M-1 FFT copies)

    # ── FL hyperparameters ─────────────────────────────────────
    "n_rounds"         : 30,    # R: total communication rounds
    "local_epochs"     : 1,      # E: local training epochs per round

    # ── Dual-Expert FC — CompNet only ─────────────────────────
    # When use_moe=True, the single FC(9708→512) is replaced by DualExpertFC:
    #
    #   base_expert   : FC(9708→512) — updated by ALL samples, FedAvg'd.
    #                   Learns domain-invariant projection.
    #   domain_expert : _ResidualExpert(9708→rank→512) — updated ONLY by
    #                   real own-domain samples (NOT FFT sentinel).
    #                   NEVER FedAvg'd — stays local to owning client.
    #                   Learns domain-specific correction.
    #
    # Combination: emb = base_expert(x) + gate * domain_expert(x)
    # FFT samples:  emb = base_expert(x)  (domain_expert skipped via sentinel -1)
    #
    # Two global models evaluated each round:
    #   GlobalBase: base_expert only   (domain_id=None at inference)
    #   GlobalFull: base_expert + domain_expert[k] for sample domain_id=k
    "use_moe"          : True,  # True → DualExpertFC for CompNet
    "lora_rank"        : 64,     # domain_expert bottleneck rank (9708→rank→512)

    # Expert combination weight
    # Fixed scalar w in [0, 1] that controls how the base_expert and
    # domain_expert outputs are blended in the combined embedding.
    #
    # Architecture (feature space):
    #   corrected_feat = gabor_feat + expert_weight * domain_expert(gabor_feat)
    #   emb = base_expert(corrected_feat)
    #
    # domain_expert operates in 9708-d Gabor feature space, outputting a
    # 9708-d correction before the base_expert projection. This means:
    #   - domain_expert corrects what the Gabor filters captured in the raw image
    #   - base_expert then projects the domain-corrected features to 512-d
    #   - GlobalFull mismatch is eliminated: domain correction is always
    #     composed with whatever base_expert is currently active
    #
    # expert_weight = 0.0 → pure base_expert (MoE disabled effectively)
    # expert_weight = 0.5 → equal blend (recommended starting point)
    # expert_weight = 1.0 → full domain correction
    "moe_expert_weight": 0.5,       # fixed combination weight w

    # Domain reconstruction loss
    # Gives the domain_expert an exclusive self-supervised objective:
    # reconstruct the domain-specific component of the Gabor features.
    #
    # For a batch of real own-domain samples from client k:
    #   overall_mean   = mean(gabor_feat, dim=0)          [9708]  all samples
    #   real_feats     = gabor_feat[real_mask]             [R, 9708]
    #   domain_signal  = mean(real_feats, dim=0)           [9708]  domain centroid
    #   target         = domain_signal - overall_mean      [9708]  domain-specific part
    #   prediction     = domain_expert(real_feats)         [R, 9708]
    #   L_recon = MSE(prediction, target.expand_as(prediction))
    #
    # Why this works:
    #   - overall_mean ≈ domain-invariant features (all domains averaged)
    #   - domain_signal = what domain k looks like on average in this batch
    #   - target = the domain-specific deviation from the global average
    #   - The domain_expert must learn to predict this domain centroid offset
    #     from any individual sample of domain k
    #   - This is antagonistic to ArcFace (which wants identity-discriminative
    #     features) — the domain_expert is rewarded for being domain-homogeneous
    #     and identity-invariant, exactly the right inductive bias
    #
    # lambda_domain_recon = 0.0 disables the loss entirely
    "lambda_domain_recon": 0.5,     # reconstruction loss weight

    # MoE warm-up: freeze domain experts for the first moe_warmup_rounds
    # FL rounds so the shared base FC learns a strong domain-invariant
    # initialisation before experts start their domain-specific corrections.
    # After warm-up, experts are unfrozen and trained normally.
    # Setting moe_warmup_rounds=0 disables warm-up (original behaviour).
    "moe_warmup_rounds": 10,     # rounds to train base only; 0 = no warmup

    # ── GRL ─────────────────────
    "use_grl"          : False,  # True → domain adversarial training (GRL)
    "lambda_grl"       : 0.05,   # GRL loss weight (start small, e.g. 0.05–0.2)
    "n_domains"        : 6,     # number of domains = n_clients (6 CASIA-MS, 4 XJTU)
    # GRL shares domain_classifier via FedAvg alongside backbone.
    # domain_ids come from the 3-tuple batch — own domain (aug_idx=0) and
    # donor domain (aug_idx≥1) — so all 6 domains appear in every client's
    # batch after FFT augmentation, giving the GRL meaningful cross-domain signal.

    "use_whitening"    : False,  # True → ZCA whiten gallery+probe at evaluation
    # Whitening matrix estimated from gallery, applied to both gallery and probe.
    # Suppresses domain-induced variance in embedding space before cosine matching.
    # Zero training cost — inference-time only.

    
    # CrossEntropy + ArcFace is always active for all models.
    #
    # Style Consistency Loss — CompNet and DINOv2:
    #   Enforces that FFT-augmented (or spatially-augmented) embeddings
    #   stay close to their original counterparts via cosine similarity.
    #   Loss = 1 - cosine_sim(emb_orig, emb_aug).  Range [0, 2]; 0 = perfect.
    #   No NaN risk — always has exactly one positive pair per sample.
    "lambda_style"     : 1,    # StyleConsistencyLoss weight (0.0 = disabled)

    # ── Supervised Contrastive Loss ────────────────────────────
    # SupCon operates on the paired embeddings [emb_orig, emb_aug] already
    # produced by FFTAugmentedDataset. No dataset change needed.
    # emb_orig = backbone(spatial_aug(fft_styled))
    # emb_aug  = backbone(spatial_aug(clean))   (when aug_idx≥1)
    # SupCon pulls same-identity pairs together across the two views,
    # complementing ArcFace which operates at the classification logit level.
    "use_supcon"       : True,  # True → add SupCon to training loss
    "lambda_supcon"    : 0.5,    # SupCon weight (paper uses 0.15–0.2)
    "temperature"      : 0.07,   # SupCon temperature

    # SupConLoss — CCNet only (between paired same-identity views):
    "ce_weight"        : 0.8,    # CE loss weight (CCNet only; CompNet/DINOv2 use 1.0)
    "con_weight"       : 0.2,    # SupConLoss weight (CCNet only)
    "temperature"      : 0.07,   # SupConLoss temperature (CCNet only)

    # CenterLoss — all models (optional):
    #   Minimises distance between embeddings and per-client class centres.
    #   Centres are kept local and carried over across rounds (never shared).
    "use_center_loss"    : False,  # True → add CenterLoss to training
    "center_loss_weight" : 0.5,  # λ_c — keep small so ArcFace dominates
    "center_loss_lr"     : 0.5,    # SGD lr for centre updates (paper default)

    # ── CompNet hyperparameters ────────────────────────────────
    "img_side"         : 128,
    "embedding_dim"    : 512,
    "dropout"          : 0.25,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,

    # ── CCNet-specific hyperparameters ─────────────────────────
    "comp_weight"      : 0.8,    # channel vs spatial competition weight

    # ── DINOv2-specific hyperparameters ───────────────────────
    # RGB 224×224 + ImageNet normalisation.
    # ViT-S/14 blocks 10 and 11 are unfrozen; all others are frozen.
    "dino_img_side"    : 224,
    "dino_weight_decay": 1e-4,   # AdamW weight decay
    "dino_margin"      : 0.3,    # ArcFace angular margin
    "dino_scale"       : 16,     # ArcFace scale

    # ── Training ───────────────────────────────────────────────
    "batch_size"       : 32,
    "lr"               : 0.001,  # constant lr across all rounds (no scheduler)

    # ── Misc ───────────────────────────────────────────────────
    "random_seed"      : 42,
    "num_workers"      : 4,
    "avg_last_rounds"  : 5,     # number of final rounds to average for reporting
}
