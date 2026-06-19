"""
config.py — Configuration for TENT-based Test-Time Adaptation.

Supports:
  - ImageNet-C (classification) with ViT-Base / ResNet-50 / ResNet-100
  - CASIA-MS (verification) with ArcFace iResNet100
"""

import argparse, math

# ─── Corruption sequences ────────────────────────────────────────────
IMAGENET_C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

# ─── CASIA Multi-Spectral Palmprint ──────────────────────────────────
CASIA_MS_SPECTRUMS = ["460", "630", "700", "850", "940", "WHT"]

CASIA_ORACLE_DOMAINS = {
    "visible": ["WHT", "460"],     # domain 0
    "red_nir": ["630", "700"],     # domain 1
    "nir":     ["850", "940"],     # domain 2
}

CASIA_ORACLE_LOOKUP = {}
for _gid, (_gname, _spectrums) in enumerate(CASIA_ORACLE_DOMAINS.items()):
    for _s in _spectrums:
        CASIA_ORACLE_LOOKUP[_s] = (_gname, _gid)


def get_cfg(args=None):
    p = argparse.ArgumentParser(description="TENT Test-Time Adaptation")

    # ─── Dataset ──────────────────────────────────────────────
    p.add_argument("--dataset", default="imagenet_c",
                   choices=["imagenet_c", "casia_ms"])
    p.add_argument("--data_dir", default="./data/ImageNet-C")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--corruptions", nargs="*", default=None,
                   help="Which corruptions to run (ImageNet-C). None=all 15.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    # ─── CASIA-MS specific ────────────────────────────────────
    p.add_argument("--train_spectrums", nargs="*", default=["WHT", "940"],
                   help="Spectrums for training (source domain). "
                        "Test spectrums = all remaining (target domain).")
    p.add_argument("--test_id_ratio", type=float, default=0.2,
                   help="Fraction of identities held out for testing. "
                        "0.2 = 20%% test IDs, 80%% train IDs.")
    p.add_argument("--gallery_ratio", type=float, default=0.1,
                   help="Fraction of samples per ID for gallery (rest=probe)")
    p.add_argument("--oracle_domains", action="store_true", default=False,
                   help="Use oracle 3-group spectrum assignment for TENT")

    # ─── ArcFace training ─────────────────────────────────────
    p.add_argument("--arcface_epochs", type=int, default=50,
                   help="Epochs to train backbone + train-ID head")
    p.add_argument("--arcface_head_epochs", type=int, default=30,
                   help="Epochs to train test-ID head (backbone frozen)")
    p.add_argument("--arcface_lr", type=float, default=1e-4,
                   help="Learning rate for Phase 1 (backbone + head_A)")
    p.add_argument("--arcface_lr_phase2", type=float, default=1e-2,
                   help="Learning rate for Phase 2 (test-ID head only)")
    p.add_argument("--arcface_wd", type=float, default=5e-4,
                   help="Weight decay for ArcFace training")
    p.add_argument("--arcface_eval_every", type=int, default=5,
                   help="Evaluate on test set every N epochs")
    p.add_argument("--arcface_freeze_ratio", type=float, default=0.75,
                   help="Fraction of backbone params to freeze during "
                        "Phase 1 training. 0.75 = finetune last 25%%.")

    # ─── Backbone ─────────────────────────────────────────────
    p.add_argument("--backbone", default="vit_base",
                   choices=["vit_base", "resnet50", "resnet101",
                            "arcface_r100"])
    p.add_argument("--arcface_onnx", type=str,
                   default="/home/pai-ng/Jamal/NIPS2026/face_models/checkpoints/r100_glint360k.onnx",
                   help="Path to ArcFace iResNet100 ONNX checkpoint")
    p.add_argument("--arcface_ckpt", type=str, default=None,
                   help="Path to trained ArcFace checkpoint (.pth) with "
                        "model + arc state dicts. Required for TENT on "
                        "CASIA-MS (provides classification head for entropy)")
    p.add_argument("--arcface_num_classes", type=int, default=None,
                   help="Number of identity classes for ArcFace head. "
                        "Auto-detected from checkpoint if not set.")
    p.add_argument("--arcface_s", type=float, default=64.0,
                   help="ArcFace scale factor")
    p.add_argument("--arcface_m", type=float, default=0.50,
                   help="ArcFace angular margin for Phase 1 (backbone training)")
    p.add_argument("--arcface_m_phase2", type=float, default=0.10,
                   help="ArcFace angular margin for Phase 2 (test-ID head). "
                        "Lower than Phase 1 since head trains on frozen backbone "
                        "with fewer samples.")
    p.add_argument("--num_classes", type=int, default=1000,
                   help="Number of classes (ImageNet-C)")
    p.add_argument("--img_size", type=int, default=224)

    # ─── TTA method & parameters ─────────────────────────────
    p.add_argument("--tta_method", default="tent",
                   choices=["tent", "bna", "contrastive"],
                   help="TTA method: "
                        "'tent' = entropy min on head_B, "
                        "'bna' = batch norm adaptation (no head), "
                        "'contrastive' = entropy(head_B) + NT-Xent "
                        "on augmented embedding pairs")
    p.add_argument("--tent_lr", type=float, default=1e-3,
                   help="TENT learning rate for BN affine params")
    p.add_argument("--tent_steps", type=int, default=1,
                   help="Gradient steps per batch (1=online TENT)")
    p.add_argument("--tent_episodic", action="store_true", default=False,
                   help="Reset BN params after each domain/spectrum")
    p.add_argument("--contrastive_lambda", type=float, default=1.0,
                   help="Weight of contrastive loss vs entropy loss")
    p.add_argument("--contrastive_temp", type=float, default=0.5,
                   help="Temperature for NT-Xent contrastive loss")

    # ─── Evaluation ───────────────────────────────────────────
    p.add_argument("--eval_backbone", action="store_true", default=False,
                   help="Evaluate frozen backbone before adaptation")

    # ─── Misc ─────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output_tent")

    cfg = p.parse_args(args)

    # ── Auto-config ──
    if cfg.dataset == "casia_ms":
        cfg.is_verification = True
        if cfg.backbone != "arcface_r100":
            print(f"[WARN] CASIA-MS requires arcface_r100 backbone, "
                  f"overriding '{cfg.backbone}'")
            cfg.backbone = "arcface_r100"
        cfg.img_size = 112  # InsightFace convention
    else:
        cfg.is_verification = False

    return cfg
