# ==============================================================
#  main.py — FLClient, FLServer, and federated training loop
# ==============================================================

import os
import time
import random
import pickle
import warnings
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from configs  import CONFIG
from models   import build_model
from datasets import (PalmDataset, AugmentedDataset,
                      PairedDataset, FFTAugmentedDataset,
                      EvalDatasetDINO, get_federated_splits)
from utils    import (extract_style_template, evaluate_model,
                      train_compnet_epoch, train_ccnet_epoch,
                      CenterLoss)


# ══════════════════════════════════════════════════════════════
#  EVAL DATASET HELPER
# ══════════════════════════════════════════════════════════════

def make_eval_dataset(samples, cfg):
    if cfg["model"].strip().lower() == "dinov2":
        return EvalDatasetDINO(samples, cfg.get("dino_img_side", 224))
    return PalmDataset(samples, cfg["img_side"])


# ══════════════════════════════════════════════════════════════
#  STRUCTURED DIAGNOSTIC PRINTER
# ══════════════════════════════════════════════════════════════

def _fmt(v, width=7, decimals=4):
    """Format a float or None for fixed-width table columns."""
    if v is None:
        return " " * width
    return f"{v:.{decimals}f}".rjust(width)

def _bar(counts):
    """Compact ASCII bar for routing counts."""
    if not counts:
        return ""
    mx = max(counts) if max(counts) > 0 else 1
    bars = ""
    for c in counts:
        filled = int(round(5 * c / mx))
        bars += "█" * filled + "░" * (5 - filled) + " "
    return bars.rstrip()

def print_moe_diagnostics_block(rnd, client_records, warmup_active):
    """
    Print a structured MoE diagnostics block for all clients after each round.

    client_records : list of dicts, one per client, each containing:
      client_id, spectrum, weight_diag, act_diag, grad_norms, routing

    Layout
    ──────
    ┌─ MoE [WARMUP|ACTIVE] · Round XXXX ─────────────────────────────────────┐
    │ SIGNAL 1+2  Weight-space norms & diversity (per client)
    │ SIGNAL 3    Routing counts (samples per expert per client)
    │ SIGNAL 4    Gradient norms (base + per-expert, last batch)
    │ SIGNAL 5    Activation residual/base ratio + activation-space diversity
    │ SUMMARY     Cross-client aggregates + collapse warnings
    └─────────────────────────────────────────────────────────────────────────┘
    """
    phase   = "WARMUP" if warmup_active else "ACTIVE"
    n_exp   = len(client_records[0]["weight_diag"]["output_norms"]) \
              if client_records and client_records[0]["weight_diag"] else 0
    sep     = "─" * 74

    print(f"\n  ┌─ MoE [{phase}] · Rnd {rnd:04d} " + "─" * 46 + "┐")

    # ── SIGNALS 1+2 : weight norms + weight-space diversity ─────────────────
    print(f"  │  [S1+S2] B-weight norms & weight-space diversity")
    hdr = "  │   Clt  Spectrum   " + "".join(f"  E{d}   " for d in range(n_exp))
    hdr += "  MnDst  MnMin"
    print(hdr)
    w_mean_dists, w_min_dists = [], []
    for r in client_records:
        wd   = r["weight_diag"]
        norms_str = "".join(_fmt(v, 7, 4) for v in wd["output_norms"])
        print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  "
              f"{norms_str}  {_fmt(wd['mean_pairwise_dist'],6,4)}"
              f"  {_fmt(wd['min_pairwise_dist'],6,4)}")
        w_mean_dists.append(wd["mean_pairwise_dist"])
        w_min_dists.append(wd["min_pairwise_dist"])
    global_w_mean = sum(w_mean_dists) / len(w_mean_dists)
    global_w_min  = min(w_min_dists)
    print(f"  │   {'':>14s}  {'Cross-client avg':>{7*n_exp}s}  "
          f"{_fmt(global_w_mean,6,4)}  {_fmt(global_w_min,6,4)}")

    # ── SIGNAL 3 : routing counts ────────────────────────────────────────────
    print(f"  │  {sep}")
    print(f"  │  [S3] Routing counts  (samples → each expert this epoch)")
    print(f"  │   Clt  Spectrum   " + "".join(f"  E{d}   " for d in range(n_exp)))
    has_routing = any(r["routing"] for r in client_records)
    for r in client_records:
        rt = r["routing"]
        if rt:
            counts_str = "".join(f"{c:>6d} " for c in rt)
            bar        = _bar(rt)
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  {counts_str} {bar}")
        else:
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  (frozen during warmup)")

    # ── SIGNAL 4 : gradient norms ────────────────────────────────────────────
    print(f"  │  {sep}")
    print(f"  │  [S4] Gradient norms  (last batch · None = not routed this batch)")
    print(f"  │   Clt  Spectrum    Base   " +
          "".join(f"  E{d}   " for d in range(n_exp)))
    grad_expert_by_pos = [[] for _ in range(n_exp)]
    for r in client_records:
        gn = r["grad_norms"]
        if gn:
            base_s   = _fmt(gn["base_grad_norm"], 7, 4)
            exp_vals = gn["expert_grad_norms"]
            exp_s    = "".join(_fmt(v, 7, 4) for v in exp_vals)
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  {base_s}  {exp_s}")
            for i, v in enumerate(exp_vals):
                if v is not None:
                    grad_expert_by_pos[i].append(v)
        else:
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  (frozen during warmup)")

    # dominant expert per client heuristic (highest grad norm = own-domain expert)
    if not warmup_active and has_routing:
        dom_str = "  │   Dominant expert per client: "
        for r in client_records:
            gn = r["grad_norms"]
            if gn and gn["expert_grad_norms"]:
                vals = [v if v is not None else -1.0
                        for v in gn["expert_grad_norms"]]
                dom  = int(np.argmax(vals))
                dom_str += f"C{r['client_id']}→E{dom}  "
        print(dom_str)

    # ── SIGNAL 5 : activation-space diagnostics ──────────────────────────────
    print(f"  │  {sep}")
    print(f"  │  [S5] Activation-space  (residual/base ratio · act-dist)")
    print(f"  │   Clt  Spectrum   BaseNm " +
          "".join(f"  R{d}   " for d in range(n_exp)) +
          "  ActMnD  ActMin")
    act_mean_dists, act_min_dists = [], []
    for r in client_records:
        ad = r["act_diag"]
        if ad:
            ratios_str = "".join(_fmt(v, 7, 4) for v in ad["residual_base_ratio"])
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  "
                  f"{_fmt(ad['base_norm'],6,3)}  {ratios_str}  "
                  f"{_fmt(ad['act_mean_dist'],7,4)}  {_fmt(ad['act_min_dist'],6,4)}")
            act_mean_dists.append(ad["act_mean_dist"])
            act_min_dists.append(ad["act_min_dist"])
        else:
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  (frozen during warmup)")

    # ── SUMMARY + WARNINGS ───────────────────────────────────────────────────
    print(f"  │  {sep}")
    print(f"  │  [SUMMARY]")
    if w_mean_dists:
        print(f"  │   Weight-space  : AvgMeanDist={global_w_mean:.4f}  "
              f"GlobalMin={global_w_min:.4f}", end="")
        if not warmup_active and global_w_min < 0.05:
            print("  ⚠ COLLAPSE RISK (weight)", end="")
        print()
    if act_mean_dists:
        g_act_mean = sum(act_mean_dists) / len(act_mean_dists)
        g_act_min  = min(act_min_dists)
        print(f"  │   Activation-sp : AvgMeanDist={g_act_mean:.4f}  "
              f"GlobalMin={g_act_min:.4f}", end="")
        if not warmup_active and g_act_min < 0.05:
            print("  ⚠ COLLAPSE RISK (activation)", end="")
        print()
    if not warmup_active and has_routing:
        # check for dead experts (never routed)
        dead = [d for d in range(n_exp)
                if all((r["routing"] or [0]*n_exp)[d] == 0
                       for r in client_records)]
        if dead:
            print(f"  │   ⚠ DEAD EXPERTS (zero routing): {dead}")
    print(f"  └" + "─" * 73 + "┘")


# ══════════════════════════════════════════════════════════════
#  FL CLIENT
# ══════════════════════════════════════════════════════════════

class FLClient:
    """
    One federated learning client — one spectral/domain, disjoint train IDs.

    Probe batch for activation diagnostics
    ───────────────────────────────────────
    A fixed set of 64 raw Gabor feature vectors is extracted once from the
    first 64 training samples at init time and stored as self.probe_feat.
    This batch is reused every round for activation diagnostics without
    running augmentation or reloading images, ensuring comparability across
    rounds. Features are extracted with no_grad in eval mode.
    """

    def __init__(self, client_id, spectrum, train_samples, label_map,
                 num_classes, cfg, device):
        self.client_id     = client_id
        self.spectrum      = spectrum
        self.train_samples = train_samples
        self.label_map     = label_map
        self.num_classes   = num_classes
        self.cfg           = cfg
        self.device        = device

        self.model = build_model(cfg, num_classes).to(device)

        model_name = cfg["model"].strip().lower()
        if cfg.get("use_center_loss", False):
            embed_dim = (cfg["embedding_dim"] if model_name == "compnet"
                         else 2048 if model_name == "ccnet" else 384)
            self.center_loss      = CenterLoss(num_classes, embed_dim, device)
            self.center_optimizer = optim.SGD(
                self.center_loss.parameters(),
                lr=cfg.get("center_loss_lr", 0.5))
        else:
            self.center_loss      = None
            self.center_optimizer = None

        # fixed probe Gabor features for activation diagnostics
        self.probe_feat = None   # built lazily after first forward pass

        print(f"  Client {client_id} [{spectrum}] [{cfg['model']}] — "
              f"train IDs: {num_classes}  samples: {len(train_samples)}"
              + (f"  [CenterLoss λ={cfg.get('center_loss_weight', 0.003)}]"
                 if cfg.get("use_center_loss", False) else ""))

    # ── weight management ───────────────────────────────────────────────────

    def set_weights(self, backbone_state_dict):
        local_state = self.model.state_dict()
        for key, val in backbone_state_dict.items():
            if key in local_state and local_state[key].shape == val.shape:
                local_state[key] = val.clone()
        self.model.load_state_dict(local_state)

    def get_weights(self):
        return {k: v.cpu().clone()
                for k, v in self.model.state_dict().items()
                if not k.startswith("arc.")}

    # ── probe feature extraction (activation diagnostics) ───────────────────

    @torch.no_grad()
    def _build_probe_feat(self, n=64):
        """
        Extract raw 9708-d Gabor features from the first n training samples.
        Called once after warm-up ends so the base FC is already meaningful.
        Features are fixed across all subsequent rounds for fair comparison.
        """
        from datasets import PalmDataset
        subset = self.train_samples[:n]
        ds     = PalmDataset(subset, self.cfg["img_side"])
        loader = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0)
        imgs, _ = next(iter(loader))
        imgs    = imgs.to(self.device)
        self.model.eval()
        feat = self.model._gabor_feat(imgs)   # [n, 9708]
        self.probe_feat = feat.detach()
        self.model.train()

    # ── local training ──────────────────────────────────────────────────────

    def local_train(self, local_epochs, active_style_bank, M, rnd,
                    mean_bank=None):
        """
        Train local model for local_epochs epochs.
        Returns (avg_loss, accuracy, warmup_active, diagnostics_dict).
        """
        model_name    = self.cfg["model"].strip().lower()
        is_dino       = model_name == "dinov2"
        img_side      = self.cfg.get("dino_img_side", 224) if is_dino \
                        else self.cfg["img_side"]
        grayscale     = not is_dino

        use_moe       = self.cfg.get("use_moe", False) and model_name == "compnet"
        warmup_rounds = self.cfg.get("moe_warmup_rounds", 0)
        warmup_active = use_moe and (warmup_rounds > 0) and (rnd <= warmup_rounds)
        if use_moe:
            self.model.set_moe_warmup(warmup_active)

        # reset routing counters before epoch
        if use_moe and not warmup_active:
            self.model.reset_routing_stats()

        if model_name in ("compnet", "dinov2"):
            if active_style_bank and M > 1:
                dataset = FFTAugmentedDataset(
                    samples              = self.train_samples,
                    style_bank           = active_style_bank,
                    client_id            = self.client_id,
                    M                    = M,
                    beta                 = self.cfg["fft_beta"],
                    img_side             = img_side,
                    grayscale            = grayscale,
                    mean_bank            = mean_bank
                                           if self.cfg.get("domain_aware_mixing", False)
                                           else None,
                    prefer_distant       = self.cfg.get("prefer_distant_domain", True),
                    use_mean_template    = self.cfg.get("use_mean_template", False),
                    deterministic_donors = False,
                )
            else:
                dataset = AugmentedDataset(self.train_samples, img_side,
                                           grayscale=grayscale)
        elif model_name == "ccnet":
            dataset = PairedDataset(
                samples    = self.train_samples,
                img_side   = img_side,
                style_bank = active_style_bank,
                client_id  = self.client_id,
                beta       = self.cfg["fft_beta"],
            )
        else:
            raise ValueError(f"Unknown model: '{self.cfg['model']}'")

        round_seed = self.cfg["random_seed"] + rnd * 1000 + self.client_id
        train_loader = DataLoader(
            dataset,
            batch_size     = self.cfg["batch_size"],
            shuffle        = True,
            num_workers    = self.cfg["num_workers"],
            pin_memory     = True,
            worker_init_fn = lambda wid, s=round_seed: (
                np.random.seed(s + wid),
                random.seed(s + wid),
                torch.manual_seed(s + wid),
            ),
        )

        criterion = nn.CrossEntropyLoss()
        if is_dino:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.cfg["lr"],
                weight_decay=self.cfg.get("dino_weight_decay", 1e-4))
        else:
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.cfg["lr"])

        lambda_c = (self.cfg.get("center_loss_weight", 0.0)
                    if self.cfg.get("use_center_loss", False) else 0.0)

        avg_loss, accuracy, last_grad_norms = 0.0, 0.0, None
        for _ in range(local_epochs):
            if model_name in ("compnet", "dinov2"):
                avg_loss, accuracy, last_grad_norms = train_compnet_epoch(
                    self.model, train_loader, criterion, optimizer, self.device,
                    center_loss         = self.center_loss,
                    center_optimizer    = self.center_optimizer,
                    lambda_center       = lambda_c,
                    lambda_style        = self.cfg.get("lambda_style", 0.0),
                    lambda_grl          = self.cfg.get("lambda_grl", 0.0)
                                          if self.cfg.get("use_grl", False) else 0.0,
                    lambda_load_balance = self.cfg.get("lambda_load_balance", 0.0)
                                          if self.cfg.get("use_moe", False) else 0.0,
                    lambda_supcon       = self.cfg.get("lambda_supcon", 0.0)
                                          if self.cfg.get("use_supcon", False) else 0.0,
                    temperature         = self.cfg.get("temperature", 0.07),
                    collect_grad_norms  = use_moe and not warmup_active,
                )
            elif model_name == "ccnet":
                avg_loss, accuracy, _ = train_ccnet_epoch(
                    self.model, train_loader, criterion, optimizer, self.device,
                    ce_weight        = self.cfg.get("ce_weight",   0.8),
                    con_weight       = self.cfg.get("con_weight",  0.2),
                    temperature      = self.cfg.get("temperature", 0.07),
                    center_loss      = self.center_loss,
                    center_optimizer = self.center_optimizer,
                    lambda_center    = lambda_c,
                )

        # ── collect all diagnostic signals ───────────────────────────────────
        diags = None
        if use_moe:
            # build probe feat once (after warmup ends)
            if not warmup_active and self.probe_feat is None:
                self._build_probe_feat()

            weight_diag = self.model.get_weight_diagnostics()
            act_diag    = (self.model.get_activation_diagnostics(self.probe_feat)
                           if self.probe_feat is not None else None)
            routing     = (self.model.get_routing_stats()
                           if not warmup_active else None)

            diags = {
                "weight_diag": weight_diag,
                "act_diag"   : act_diag,
                "grad_norms" : last_grad_norms,
                "routing"    : routing,
                "warmup"     : warmup_active,
            }

        return avg_loss, accuracy, warmup_active, diags

    # ── style template extraction ───────────────────────────────────────────

    def extract_style_templates(self):
        model_name = self.cfg["model"].strip().lower()
        is_dino    = model_name == "dinov2"
        img_side   = self.cfg.get("dino_img_side", 224) if is_dino \
                     else self.cfg["img_side"]
        mode       = "RGB" if is_dino else "L"
        templates  = []
        for path, _ in self.train_samples:
            img    = Image.open(path).convert(mode).resize(
                (img_side, img_side), Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            templates.append(extract_style_template(img_np))
        print(f"  Client {self.client_id} [{self.spectrum}] "
              f"— extracted {len(templates)} style templates")
        return templates


# ══════════════════════════════════════════════════════════════
#  FL SERVER
# ══════════════════════════════════════════════════════════════

class FLServer:
    """Central server: global model, FedAvg, domain-aware evaluation."""

    def __init__(self, num_classes, gallery_samples, probe_samples, cfg, device):
        self.cfg    = cfg
        self.device = device

        self.global_model = build_model(cfg, num_classes).to(device)

        self.gallery_samples = gallery_samples
        self.probe_samples   = probe_samples

        self.gallery_loader = DataLoader(
            make_eval_dataset(gallery_samples, cfg),
            batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True)
        self.probe_loader = DataLoader(
            make_eval_dataset(probe_samples, cfg),
            batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True)

        # extract domain_id lists from sample tuples  [(path, label, domain_id), ...]
        use_moe = cfg.get("use_moe", False) and cfg["model"].strip().lower() == "compnet"
        if use_moe:
            self.gallery_domain_ids = [s[2] for s in gallery_samples
                                       if len(s) >= 3]
            self.probe_domain_ids   = [s[2] for s in probe_samples
                                       if len(s) >= 3]
            if len(self.gallery_domain_ids) != len(gallery_samples):
                self.gallery_domain_ids = None
            if len(self.probe_domain_ids) != len(probe_samples):
                self.probe_domain_ids = None
        else:
            self.gallery_domain_ids = None
            self.probe_domain_ids   = None

        print(f"  Server [{cfg['model']}] — "
              f"gallery: {len(gallery_samples)}  probe: {len(probe_samples)}"
              + ("  [domain-aware eval]" if use_moe and self.gallery_domain_ids
                 else ""))

    def get_global_weights(self):
        return {k: v.cpu().clone()
                for k, v in self.global_model.state_dict().items()
                if not k.startswith("arc.")}

    def aggregate(self, client_weight_dicts):
        n        = len(client_weight_dicts)
        avg_dict = {}
        for key in client_weight_dicts[0].keys():
            stacked      = torch.stack(
                [client_weight_dicts[i][key].float() for i in range(n)], dim=0)
            avg_dict[key] = stacked.mean(dim=0)
        global_state = self.global_model.state_dict()
        global_state.update(avg_dict)
        self.global_model.load_state_dict(global_state)

    def evaluate(self, use_whitening=False, warmup_active=False):
        """
        Evaluate global model.
        Domain-aware expert routing is used when use_moe=True and warmup
        has ended (during warmup only the base FC is meaningful).
        """
        gal_ids = (None if warmup_active else self.gallery_domain_ids)
        prb_ids = (None if warmup_active else self.probe_domain_ids)
        return evaluate_model(
            self.global_model,
            self.gallery_loader, self.probe_loader, self.device,
            use_whitening       = use_whitening,
            gallery_domain_ids  = gal_ids,
            probe_domain_ids    = prb_ids)


# ══════════════════════════════════════════════════════════════
#  AUGMENTATION MODE HELPERS
# ══════════════════════════════════════════════════════════════

def resolve_aug_mode(cfg):
    if cfg.get("use_mixed_aug", False):
        return "mixed"
    if cfg.get("use_fft_aug", False):
        return "fft"
    return "spatial"


def get_active_style_bank(style_bank_full, rnd, cfg, is_dinov2):
    mode = resolve_aug_mode(cfg)
    if mode == "spatial":
        return {}
    if mode == "fft":
        return style_bank_full
    switch = cfg.get("mixed_aug_round", cfg["n_rounds"] // 2)
    return style_bank_full if rnd > switch else {}


def aug_mode_label(rnd, cfg, is_dinov2):
    mode = resolve_aug_mode(cfg)
    if mode == "spatial":
        return "Spatial"
    if mode == "fft":
        return "FFT"
    switch = cfg.get("mixed_aug_round", cfg["n_rounds"] // 2)
    return "FFT" if rnd > switch else "Spatial"


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    cfg  = CONFIG
    seed = cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir  = cfg["base_results_dir"]
    is_dinov2 = cfg["model"].strip().lower() == "dinov2"
    aug_mode  = resolve_aug_mode(cfg)
    use_moe   = cfg.get("use_moe", False) and cfg["model"].strip().lower() == "compnet"
    warmup_rds = cfg.get("moe_warmup_rounds", 0) if use_moe else 0
    os.makedirs(base_dir, exist_ok=True)

    # ── header ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Federated Learning — Palmprint")
    print(f"  Dataset  : {cfg['dataset'].upper()}")
    print(f"  Protocol : Open-Set, Non-Shared-ID, Cross-Domain")
    print(f"  Model    : {cfg['model'].upper()}")
    print(f"  Device   : {device}")
    print(f"  Rounds   : {cfg['n_rounds']}   "
          f"Local epochs/round : {cfg['local_epochs']}")
    print(f"  IDs : {cfg['n_ids']}   "
          f"Test ID ratio : {cfg['k_test']*100:.0f}%   "
          f"Gallery ratio : {cfg['gallery_ratio']*100:.0f}%")
    if aug_mode == "mixed":
        switch = cfg.get("mixed_aug_round", cfg["n_rounds"] // 2)
        print(f"  Aug mode : MIXED  "
              f"(Spatial rds 1–{switch}, FFT rds {switch+1}–{cfg['n_rounds']})"
              f"  M={cfg['M']}  β={cfg['fft_beta']}")
    else:
        print(f"  Aug mode : {aug_mode.upper()}   M={cfg['M']}  β={cfg['fft_beta']}")
    if use_moe:
        wstr = (f"warm-up rds 1–{warmup_rds}, experts active from rd {warmup_rds+1}"
                if warmup_rds > 0 else "no warm-up")
        print(f"  MoE      : ENABLED  asymmetric routing  ({wstr})")
        print(f"             Inference: base + expert[domain_id]  (known domain labels)")
    print(f"  LR       : {cfg['lr']} (constant)")
    print(f"{'='*62}\n")

    # ── paths ─────────────────────────────────────────────────────────────────
    dataset_key       = cfg["dataset"].strip().lower()
    model_key         = cfg["model"].strip().lower()
    splits_path       = cfg["splits_path"].format(dataset=dataset_key)
    init_weights_path = cfg["init_weights_path"].format(
                            dataset=dataset_key, model=model_key)

    # ── Step 0a: data splits ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    if os.path.exists(splits_path):
        print(f"Loading data splits from: {splits_path}")
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        client_data, gallery_samples, probe_samples, test_label_map, domain_names = splits
        print("  Splits loaded.")
    else:
        print(f"Building federated data splits for {cfg['dataset'].upper()} …")
        splits = get_federated_splits(cfg, seed=seed)
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)
        print(f"  Splits saved to: {splits_path}")
        client_data, gallery_samples, probe_samples, test_label_map, domain_names = splits

    num_classes = client_data[0]["num_classes"]
    n_clients   = len(client_data)
    print(f"\n  Clients: {n_clients}  {domain_names}")
    print(f"  IDs per client: {num_classes}  |  Test classes: {len(test_label_map)}")
    print(f"  Gallery: {len(gallery_samples)}  |  Probe: {len(probe_samples)}\n")

    # ── Step 0b: server ───────────────────────────────────────────────────────
    print("Initialising server …")
    server = FLServer(num_classes, gallery_samples, probe_samples, cfg, device)

    # ── Step 0c: clients ──────────────────────────────────────────────────────
    print("Initialising clients …")
    clients = []
    for i, cd in enumerate(client_data):
        clients.append(FLClient(
            client_id=i, spectrum=cd["spectrum"],
            train_samples=cd["train_samples"], label_map=cd["label_map"],
            num_classes=num_classes, cfg=cfg, device=device))
    cfg["_n_clients"] = list(range(n_clients))

    # ── Step 0d: init weights ─────────────────────────────────────────────────
    if os.path.exists(init_weights_path):
        _probe    = torch.load(init_weights_path, map_location="cpu")
        _ckpt_arc = _probe.get("arc.weight", None)
        _mod_arc  = server.global_model.state_dict()["arc.weight"]
        if _ckpt_arc is None or _ckpt_arc.shape != _mod_arc.shape:
            print(f"\nInit weights arc.weight shape mismatch — regenerating.")
            os.remove(init_weights_path)

    if os.path.exists(init_weights_path):
        print(f"\nLoading initial weights from: {init_weights_path}")
        init_state = torch.load(init_weights_path, map_location=device)
        missing, _ = server.global_model.load_state_dict(init_state, strict=False)
        if missing:
            print(f"  INFO: {len(missing)} new key(s) fresh-init.")
        backbone_init = {k: v for k, v in init_state.items()
                         if not k.startswith("arc.")}
        for client in clients:
            client.set_weights(backbone_init)
        print("  Initial weights loaded.")
    else:
        print(f"\nSaving initial weights to: {init_weights_path}")
        torch.save(server.global_model.state_dict(), init_weights_path)
        backbone_init = {k: v for k, v in server.global_model.state_dict().items()
                         if not k.startswith("arc.")}
        for client in clients:
            client.set_weights(backbone_init)
        print("  Initial weights saved.")

    # ── results file ──────────────────────────────────────────────────────────
    results_path  = os.path.join(base_dir, "results.txt")
    client_header = "\t".join(
        f"Client{i}_EER(%)\tClient{i}_Rank1(%)" for i in range(n_clients))
    with open(results_path, "w") as f:
        f.write(f"Round\tAug_Mode\tGlobal_EER(%)\tGlobal_Rank1(%)\t{client_header}\n")

    # ── Step 0e: style templates ──────────────────────────────────────────────
    print("\nExtracting style templates …")
    style_bank_full = {c.client_id: c.extract_style_templates() for c in clients}
    total = sum(len(v) for v in style_bank_full.values())
    print(f"  Style bank: {total} templates across {len(style_bank_full)} clients")

    mean_bank_full = {
        cid: np.mean(templates, axis=0)
        for cid, templates in style_bank_full.items()
        if len(templates) > 0
    }
    if aug_mode == "mixed":
        switch = cfg.get("mixed_aug_round", cfg["n_rounds"] // 2)
        print(f"  Mixed aug: Spatial rds 1–{switch}, FFT rds {switch+1}–{cfg['n_rounds']}\n")
    elif aug_mode == "fft":
        print("  FFT augmentation ENABLED.\n")
    else:
        print("  Spatial augmentation only.\n")

    use_whitening = cfg.get("use_whitening", False)

    # ── Round 0: random init ──────────────────────────────────────────────────
    print("─" * 62)
    print("  Round 0 (random init)")
    g_eer_0, g_rank1_0 = server.evaluate(use_whitening=use_whitening,
                                          warmup_active=True)
    print(f"  Global → EER={g_eer_0*100:.4f}%  Rank-1={g_rank1_0:.2f}%")
    print("─" * 62)
    with open(results_path, "a") as f:
        f.write(f"0\tInit\t{g_eer_0*100:.4f}\t{g_rank1_0:.2f}\t"
                + "\t".join("-1\t-1" for _ in range(n_clients)) + "\n")

    g_eer, g_rank1 = g_eer_0, g_rank1_0
    recent_history = []

    # ── FL rounds ─────────────────────────────────────────────────────────────
    for rnd in range(1, cfg["n_rounds"] + 1):
        t_start = time.time()

        active_style_bank = get_active_style_bank(
            style_bank_full, rnd, cfg, is_dinov2)
        mode_label    = aug_mode_label(rnd, cfg, is_dinov2)
        rnd_warmup    = use_moe and warmup_rds > 0 and rnd <= warmup_rds

        if use_moe and warmup_rds > 0:
            if rnd == 1:
                print(f"\n  ► MoE WARMUP START: rounds 1–{warmup_rds} "
                      f"(base FC only, experts frozen)")
            elif rnd == warmup_rds + 1:
                print(f"\n  ► MoE WARMUP END: round {rnd} "
                      f"— experts ACTIVE + domain-aware eval ENABLED")

        global_weights = server.get_global_weights()
        client_weights = []
        client_metrics = []
        client_diag_records = []

        # ── Step 1: local training ────────────────────────────────────────────
        for client in clients:
            client.set_weights(global_weights)
            loss, acc, warmup_active, diags = client.local_train(
                cfg["local_epochs"], active_style_bank, cfg["M"], rnd,
                mean_bank=mean_bank_full
                          if cfg.get("domain_aware_mixing", False) else None)

            c_eer, c_rank1 = evaluate_model(
                client.model,
                server.gallery_loader, server.probe_loader, device,
                use_whitening      = use_whitening,
                gallery_domain_ids = (None if warmup_active
                                      else server.gallery_domain_ids),
                probe_domain_ids   = (None if warmup_active
                                      else server.probe_domain_ids))

            client_weights.append(client.get_weights())
            client_metrics.append({
                "client_id"    : client.client_id,
                "spectrum"     : client.spectrum,
                "train_loss"   : round(loss, 4),
                "train_acc"    : round(acc, 1),
                "eer"          : round(c_eer, 6),
                "rank1"        : round(c_rank1, 2),
                "warmup_active": warmup_active,
            })

            if use_moe and diags is not None:
                client_diag_records.append({
                    "client_id"  : client.client_id,
                    "spectrum"   : client.spectrum,
                    **diags,
                })

        # ── Step 2: FedAvg + global eval ─────────────────────────────────────
        server.aggregate(client_weights)
        g_eer, g_rank1 = server.evaluate(use_whitening=use_whitening,
                                          warmup_active=rnd_warmup)
        elapsed = time.time() - t_start

        avg_rounds = cfg.get("avg_last_rounds", 5)
        recent_history.append((g_eer, g_rank1))
        if len(recent_history) > avg_rounds:
            recent_history.pop(0)

        # ── Step 3: print MoE diagnostics block ──────────────────────────────
        if use_moe and client_diag_records:
            print_moe_diagnostics_block(rnd, client_diag_records, rnd_warmup)

        # ── Step 4: print round summary table ────────────────────────────────
        ts = time.strftime("%H:%M:%S")
        print(f"\n  [{ts}] Rnd {rnd:04d}/{cfg['n_rounds']} "
              f"[{mode_label}{'·WARMUP' if rnd_warmup else ''}]  "
              f"Global EER={g_eer*100:.4f}%  Rank-1={g_rank1:.2f}%  "
              f"({elapsed:.1f}s)")

        # compact per-client table
        print(f"  {'Clt':>3}  {'Spectrum':>8}  {'Loss':>7}  {'Acc%':>5}  "
              f"{'EER%':>6}  {'R1%':>5}")
        print(f"  {'─'*3}  {'─'*8}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*5}")
        for cm in client_metrics:
            print(f"  C{cm['client_id']}   {cm['spectrum']:>8s}  "
                  f"{cm['train_loss']:>7.4f}  {cm['train_acc']:>5.1f}  "
                  f"{cm['eer']*100:>6.3f}  {cm['rank1']:>5.2f}")
        print()

        client_cols = "\t".join(
            f"{cm['eer']*100:.4f}\t{cm['rank1']:.2f}"
            for cm in client_metrics)
        with open(results_path, "a") as f:
            f.write(f"{rnd}\t{mode_label}\t{g_eer*100:.4f}\t{g_rank1:.2f}"
                    f"\t{client_cols}\n")

    # ── Final report ──────────────────────────────────────────────────────────
    n_avg     = len(recent_history)
    avg_eer   = sum(e for e, _ in recent_history) / n_avg
    avg_rank1 = sum(r for _, r in recent_history) / n_avg

    print(f"\n{'='*62}")
    print(f"  FL COMPLETE — {cfg['n_rounds']} rounds")
    print(f"  Dataset          : {cfg['dataset'].upper()}")
    print(f"  Model            : {cfg['model'].upper()}")
    print(f"  Aug mode         : {aug_mode.upper()}")
    if use_moe:
        wstr = (f"warm-up rds 1–{warmup_rds}, active from {warmup_rds+1}"
                if warmup_rds > 0 else "no warm-up")
        print(f"  MoE              : ENABLED  ({wstr})")
        print(f"  Inference        : base + expert[domain_id]")
    print(f"  Avg Global EER   : {avg_eer*100:.4f}%  (last {n_avg} rounds)")
    print(f"  Avg Global Rank-1: {avg_rank1:.2f}%  (last {n_avg} rounds)")
    print(f"  Results saved    : {results_path}")
    print(f"{'='*62}")

    with open(results_path, "a") as f:
        f.write(f"\n# Average of last {n_avg} rounds\n")
        f.write(f"avg_{n_avg}\t—\t{avg_eer*100:.4f}\t{avg_rank1:.2f}\n")


if __name__ == "__main__":
    main()
