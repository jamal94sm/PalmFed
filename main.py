# ==============================================================
#  main.py — FLClient, FLServer, and federated training loop
# ==============================================================
#
#  Two global models (MoE mode):
#
#  GlobalBase  — shared base_expert (FedAvg only).
#                Used for domain-unknown evaluation.
#                get_embedding(x, domain_id=None)
#
#  GlobalFull  — shared base_expert + all n_clients domain_experts stacked.
#                After each round the server collects each client's
#                domain_gabor branches, stores them indexed by
#                client_id, then evaluates with:
#                  for sample with domain_id=k:
#                    emb = base(x) + gate_k * domain_expert_k(x)
#                Routing is deterministic by domain label — no learned router.
#                get_embedding(x, domain_id=k) via
#                get_embedding_with_external_expert(x, domain_experts[k],
#                                                   domain_gabors[k])
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
                      extract_features_dual,
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
#  DIAGNOSTIC PRINTER
# ══════════════════════════════════════════════════════════════

def _fmt(v, width=8, dec=4):
    if v is None: return " " * width
    return f"{v:.{dec}f}".rjust(width)

def print_moe_diagnostics(rnd, client_records, warmup_active):
    """
    Structured diagnostic block printed once per round.

    client_records : list of dicts, one per client:
      client_id, spectrum, weight_diag, act_diag, grad_norms,
      routing, warmup

    Signals reported
    ────────────────
    S1  base_weight_norm   — shared base_expert weight magnitude
    S2  domain_weight_norm — local domain_expert B-weight magnitude
    S3  gate_value         — current effective gate (sigmoid*scale)
    S4  routing real/skip  — real samples vs FFT-sentinel samples this epoch
    S5  grad norms         — base, domain_expert, gate after last batch
    S6  gated_base_ratio   — ||gate*domain_expert(x)|| / ||base(x)||
    S7  base_full_cos_sim  — cosine between base-only and full embeddings
                             (1 = domain_expert adds nothing,
                              0 = 90° correction = strong specialisation)
    """
    phase = "WARMUP" if warmup_active else "ACTIVE"
    sep   = "─" * 74

    # Guard: if weight_diag has old-format keys (from a stale models.py),
    # skip the block rather than crashing with a KeyError.
    if client_records:
        wd0 = client_records[0].get("weight_diag") or {}
        if wd0 and "output_norms" in wd0:
            print(f"  [MoE diag] WARNING: stale models.py detected "
                  f"(old MoEFC keys). Replace models.py and restart.")
            return

    print(f"\n  ┌─ MoE [{phase}] · Rnd {rnd:04d} " + "─" * 46 + "┐")

    # ── S1-S3: weight norms and gate ────────────────────────────────────────
    print(f"  │  [S1-S3] Weight norms & gate values")
    print(f"  │   Clt  Spectrum   BaseNorm  DomNorm   Gate")
    for r in client_records:
        wd = r["weight_diag"]
        if wd:
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  "
                  f"{_fmt(wd['base_weight_norm'],9,4)}"
                  f"{_fmt(wd['domain_weight_norm'],9,4)}"
                  f"{_fmt(wd['gate_value'],7,4)}")
        else:
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  (warmup)")

    # ── S4: routing ──────────────────────────────────────────────────────────
    print(f"  │  {sep}")
    print(f"  │  [S4] Routing  (real=domain_expert updated · skip=base only)")
    print(f"  │   Clt  Spectrum    Real    Skip   Real%")
    for r in client_records:
        rt = r["routing"]
        if rt:
            real = rt["real"]; skip = rt["skip"]
            total = real + skip
            pct   = 100.0 * real / total if total > 0 else 0.0
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}"
                  f"  {real:6d}  {skip:6d}  {pct:5.1f}%")
        else:
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  (warmup)")

    # ── S5: gradient norms ───────────────────────────────────────────────────
    print(f"  │  {sep}")
    print(f"  │  [S5] Gradient norms  (last batch)")
    print(f"  │   Clt  Spectrum    BaseGrad  DomGrad   GateGrad")
    for r in client_records:
        gn = r["grad_norms"]
        if gn:
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  "
                  f"{_fmt(gn['base_grad_norm'],9,4)}"
                  f"{_fmt(gn['domain_grad_norm'],9,4)}"
                  f"{_fmt(gn['gate_grad_norm'],9,4)}")
        else:
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  (warmup)")

    # ── S6-S7: activation diagnostics ────────────────────────────────────────
    print(f"  │  {sep}")
    print(f"  │  [S6-S7] Activation space  "
          f"(GatedRatio=||gated_res||/||base||  CosSim=base·full)")
    print(f"  │   Clt  Spectrum   BaseNorm  DomNorm   GatedRat  Gate     CosSim")
    for r in client_records:
        ad = r["act_diag"]
        if ad:
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  "
                  f"{_fmt(ad['base_norm'],9,3)}"
                  f"{_fmt(ad['domain_gated_norm'],9,4)}"
                  f"{_fmt(ad['gated_base_ratio'],9,4)}"
                  f"{_fmt(ad['gate_value'],8,4)}"
                  f"{_fmt(ad['base_full_cos_sim'],8,4)}")
        else:
            print(f"  │   C{r['client_id']}  {r['spectrum']:>8s}  (warmup)")

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"  │  {sep}")
    print(f"  │  [SUMMARY]")
    act_diags = [r["act_diag"] for r in client_records if r["act_diag"]]
    if act_diags:
        avg_ratio  = sum(a["gated_base_ratio"]    for a in act_diags) / len(act_diags)
        avg_cos    = sum(a["base_full_cos_sim"]    for a in act_diags) / len(act_diags)
        avg_gate   = sum(a["gate_value"]           for a in act_diags) / len(act_diags)
        print(f"  │   ExpertWeight (fixed)  : {avg_gate:.4f}")
        print(f"  │   Avg feat-corr/feat    : {avg_ratio:.4f}"
              f"  (>0.1 = meaningful feature correction)")
        print(f"  │   Avg base·full CosSim  : {avg_cos:.4f}"
              f"  (<0.99 = domain_expert shifting embedding direction)")
        recon_vals = [r.get("recon_loss") for r in client_records
                      if r.get("recon_loss") is not None]
        if recon_vals:
            avg_r = sum(recon_vals) / len(recon_vals)
            print(f"  │   Avg domain recon loss : {avg_r:.6f}"
                  f"  (decreasing = domain_expert learning domain signal)")
        if not warmup_active and avg_ratio < 0.05:
            print(f"  │   ⚠ feature correction very small — "
                  f"consider larger moe_expert_weight")
    print(f"  └" + "─" * 73 + "┘")


# ══════════════════════════════════════════════════════════════
#  FL CLIENT
# ══════════════════════════════════════════════════════════════

class FLClient:
    """
    One federated learning client.

    Weight management
    ─────────────────
    get_weights() returns only FedAvg-eligible keys:
      fc.base_expert.* and all Gabor/CB parameters.
    Excludes: arc.*, cb1d.*, cb2d.*, cb3d.*

    set_weights() loads only FedAvg weights, preserving local keys unchanged.

    domain_gabor branches accumulate across ALL rounds without reset.
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
            self.center_loss = None; self.center_optimizer = None

        # fixed probe Gabor features for activation diagnostics
        self.probe_feat = None

        print(f"  Client {client_id} [{spectrum}] [{cfg['model']}] — "
              f"train IDs: {num_classes}  samples: {len(train_samples)}")

    # ── weight management ───────────────────────────────────────────────────

    def get_weights(self):
        """
        FedAvg-eligible weights only:  fc.base_expert.* + Gabor/CB params.
        Local-only keys excluded:      arc.*, cb1d.*, cb2d.*, cb3d.*
        """
        excl = self.model.local_only_keys() \
               if hasattr(self.model, "local_only_keys") else ("arc.",)
        return {k: v.cpu().clone()
                for k, v in self.model.state_dict().items()
                if not any(k.startswith(p) for p in excl)}

    def set_weights(self, backbone_state_dict):
        """Load FedAvg weights; local-only keys preserved unchanged."""
        local_state = self.model.state_dict()
        for key, val in backbone_state_dict.items():
            if key in local_state and local_state[key].shape == val.shape:
                local_state[key] = val.clone()
        self.model.load_state_dict(local_state)

    def get_domain_expert_state(self):
        """
        Return cb1d/cb2d/cb3d state dicts and expert_weight for GlobalFull.
        Used by FLServer to assemble GlobalFull after each round.
        Returns dict with cb1d/cb2d/cb3d state dicts and expert_weight.
        """
        if not (hasattr(self.model, "use_moe") and self.model.use_moe):
            return None
        return {
            "cb1d_state"   : {k: v.cpu().clone() for k,v in self.model.cb1d.state_dict().items()},
            "cb2d_state"   : {k: v.cpu().clone() for k,v in self.model.cb2d.state_dict().items()},
            "cb3d_state"   : {k: v.cpu().clone() for k,v in self.model.cb3d.state_dict().items()},
            "expert_weight": self.model.expert_weight,
            "client_id"    : self.client_id,
        }

    # ── probe feature extraction ────────────────────────────────────────────

    @torch.no_grad()
    def _build_probe_feat(self, n=64):
        subset = self.train_samples[:n]
        ds     = PalmDataset(subset, self.cfg["img_side"])
        loader = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0)
        imgs, _ = next(iter(loader))
        self.probe_feat = imgs.cpu()   # store images; moved to device in diagnostics

    # ── local training ──────────────────────────────────────────────────────

    def local_train(self, local_epochs, active_style_bank, M, rnd,
                    mean_bank=None):
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
                                           grayscale=grayscale,
                                           client_id=self.client_id)
        elif model_name == "ccnet":
            dataset = PairedDataset(
                samples=self.train_samples, img_side=img_side,
                style_bank=active_style_bank,
                client_id=self.client_id, beta=self.cfg["fft_beta"])
        else:
            raise ValueError(f"Unknown model: '{self.cfg['model']}'")

        round_seed = self.cfg["random_seed"] + rnd * 1000 + self.client_id
        train_loader = DataLoader(
            dataset, batch_size=self.cfg["batch_size"], shuffle=True,
            num_workers=self.cfg["num_workers"], pin_memory=True,
            worker_init_fn=lambda wid, s=round_seed: (
                np.random.seed(s+wid), random.seed(s+wid),
                torch.manual_seed(s+wid)),
        )

        criterion = nn.CrossEntropyLoss()
        is_dino_opt = model_name == "dinov2"
        if is_dino_opt:
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
                avg_loss, accuracy, last_grad_norms, avg_recon = train_compnet_epoch(
                    self.model, train_loader, criterion, optimizer, self.device,
                    center_loss        = self.center_loss,
                    center_optimizer   = self.center_optimizer,
                    lambda_center      = lambda_c,
                    lambda_style       = self.cfg.get("lambda_style", 0.0),
                    lambda_grl         = self.cfg.get("lambda_grl", 0.0)
                                         if self.cfg.get("use_grl", False) else 0.0,
                    lambda_supcon      = self.cfg.get("lambda_supcon", 0.0)
                                         if self.cfg.get("use_supcon", False) else 0.0,
                    temperature        = self.cfg.get("temperature", 0.07),
                    collect_grad_norms  = use_moe and not warmup_active,
                    lambda_domain_recon = self.cfg.get("lambda_domain_recon", 0.0)
                                          if use_moe and not warmup_active else 0.0,
                )
            elif model_name == "ccnet":
                avg_loss, accuracy, _, avg_recon = train_ccnet_epoch(
                    self.model, train_loader, criterion, optimizer, self.device,
                    ce_weight=self.cfg.get("ce_weight", 0.8),
                    con_weight=self.cfg.get("con_weight", 0.2),
                    temperature=self.cfg.get("temperature", 0.07),
                    center_loss=self.center_loss,
                    center_optimizer=self.center_optimizer,
                    lambda_center=lambda_c)

        # ── collect diagnostics ──────────────────────────────────────────────
        diags = None
        if use_moe:
            if not warmup_active and self.probe_feat is None:
                self._build_probe_feat()
            diags = {
                "weight_diag": self.model.get_weight_diagnostics(),
                "act_diag"   : (self.model.get_activation_diagnostics(self.probe_feat)
                                if self.probe_feat is not None else None),
                "grad_norms" : last_grad_norms,
                "routing"    : (self.model.get_routing_stats()
                                if not warmup_active else None),
                "recon_loss" : avg_recon if not warmup_active else None,
                "warmup"     : warmup_active,
            }

        return avg_loss, accuracy, warmup_active, diags

    # ── style template extraction ───────────────────────────────────────────

    def extract_style_templates(self):
        model_name = self.cfg["model"].strip().lower()
        is_dino    = model_name == "dinov2"
        img_side   = self.cfg.get("dino_img_side", 224) if is_dino \
                     else self.cfg["img_side"]
        mode = "RGB" if is_dino else "L"
        templates = []
        for path, _ in self.train_samples:
            img    = Image.open(path).convert(mode).resize(
                (img_side, img_side), Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            templates.append(extract_style_template(img_np))
        print(f"  Client {self.client_id} [{self.spectrum}] "
              f"— extracted {len(templates)} style templates")
        return templates


# ══════════════════════════════════════════════════════════════
#  FL SERVER  (two global models)
# ══════════════════════════════════════════════════════════════

class FLServer:
    """
    Central server managing two global models.

    GlobalBase  — base_expert only (FedAvg of fc.base_expert.*).
                  Evaluated with domain_id=None.

    GlobalFull  — base_expert + all n_clients domain_experts stacked.
                  Assembled each round after receiving client domain states.
                  Routing at inference: domain_id=k → use client_k's expert.
                  Evaluated via extract_features_dual() in utils.py.
    """

    def __init__(self, num_classes, gallery_samples, probe_samples,
                 cfg, device, n_clients):
        self.cfg       = cfg
        self.device    = device
        self.n_clients = n_clients

        # GlobalBase model
        self.global_model = build_model(cfg, num_classes).to(device)

        # Domain gabor registry: client_id → {cb1d/cb2d/cb3d states}
        # Populated each round by collect_domain_experts()
        self.domain_expert_registry = {}

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
              + ("  [GlobalBase + GlobalFull]" if use_moe else ""))

    def get_global_weights(self):
        """FedAvg-eligible weights from GlobalBase model."""
        excl = self.global_model.local_only_keys() \
               if hasattr(self.global_model, "local_only_keys") else ("arc.",)
        return {k: v.cpu().clone()
                for k, v in self.global_model.state_dict().items()
                if not any(k.startswith(p) for p in excl)}

    def aggregate(self, client_weight_dicts):
        """FedAvg on base_expert + Gabor params."""
        n        = len(client_weight_dicts)
        avg_dict = {}
        for key in client_weight_dicts[0].keys():
            stacked      = torch.stack(
                [client_weight_dicts[i][key].float() for i in range(n)], dim=0)
            avg_dict[key] = stacked.mean(dim=0)
        gs = self.global_model.state_dict()
        gs.update(avg_dict)
        self.global_model.load_state_dict(gs)

    def collect_domain_experts(self, client_domain_states):
        """
        Store each client's domain_gabor branch states for GlobalFull assembly.
        Called after local training, before GlobalFull evaluation.

        client_domain_states : list of dicts from client.get_domain_expert_state()
        """
        self.domain_expert_registry = {}
        for state in client_domain_states:
            if state is not None:
                cid = state["client_id"]
                self.domain_expert_registry[cid] = {
                    "cb1d_state"   : state["cb1d_state"],
                    "cb2d_state"   : state["cb2d_state"],
                    "cb3d_state"   : state["cb3d_state"],
                    "expert_weight": state.get("expert_weight", 0.5),
                }

    def evaluate_base(self, use_whitening=False, warmup_active=False):
        """Evaluate GlobalBase (base_expert only, domain_id=None)."""
        return evaluate_model(
            self.global_model,
            self.gallery_loader, self.probe_loader, self.device,
            use_whitening      = use_whitening,
            gallery_domain_ids = None,
            probe_domain_ids   = None)

    def evaluate_full(self, use_whitening=False):
        """
        Evaluate GlobalFull.
        For each sample with domain_id=k, uses:
          global_model.base_expert(feat) + gate_k * domain_expert_k(feat)
        where gate_k and domain_expert_k come from client k's local state.
        """
        if not self.domain_expert_registry:
            return None, None
        return extract_features_dual(
            base_model          = self.global_model,
            gallery_loader      = self.gallery_loader,
            probe_loader        = self.probe_loader,
            device              = self.device,
            gallery_domain_ids  = self.gallery_domain_ids,
            probe_domain_ids    = self.probe_domain_ids,
            domain_expert_registry = self.domain_expert_registry,
            use_whitening       = use_whitening,
        )


# ══════════════════════════════════════════════════════════════
#  AUGMENTATION HELPERS
# ══════════════════════════════════════════════════════════════

def resolve_aug_mode(cfg):
    if cfg.get("use_mixed_aug", False): return "mixed"
    if cfg.get("use_fft_aug",   False): return "fft"
    return "spatial"

def get_active_style_bank(style_bank_full, rnd, cfg, is_dinov2):
    mode = resolve_aug_mode(cfg)
    if mode == "spatial": return {}
    if mode == "fft":     return style_bank_full
    switch = cfg.get("mixed_aug_round", cfg["n_rounds"] // 2)
    return style_bank_full if rnd > switch else {}

def aug_mode_label(rnd, cfg, is_dinov2):
    mode = resolve_aug_mode(cfg)
    if mode == "spatial": return "Spatial"
    if mode == "fft":     return "FFT"
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

    device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir   = cfg["base_results_dir"]
    is_dinov2  = cfg["model"].strip().lower() == "dinov2"
    aug_mode   = resolve_aug_mode(cfg)
    use_moe    = cfg.get("use_moe", False) and cfg["model"].strip().lower() == "compnet"
    warmup_rds = cfg.get("moe_warmup_rounds", 0) if use_moe else 0
    os.makedirs(base_dir, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Federated Learning — Palmprint")
    print(f"  Dataset  : {cfg['dataset'].upper()}")
    print(f"  Model    : {cfg['model'].upper()}")
    print(f"  Device   : {device}")
    print(f"  Rounds   : {cfg['n_rounds']}   "
          f"Local epochs: {cfg['local_epochs']}")
    if aug_mode == "mixed":
        switch = cfg.get("mixed_aug_round", cfg["n_rounds"] // 2)
        print(f"  Aug mode : MIXED  (Spatial rds 1–{switch}, "
              f"FFT rds {switch+1}–{cfg['n_rounds']})")
    else:
        print(f"  Aug mode : {aug_mode.upper()}   M={cfg['M']}  β={cfg['fft_beta']}")
    if use_moe:
        gmode = cfg.get("moe_gate_mode", "scalar")
        ginit = cfg.get("moe_gate_init", 1.0)
        gscal = cfg.get("moe_gate_scale", 2.0)
        wstr  = (f"warmup rds 1–{warmup_rds}" if warmup_rds > 0
                 else "no warmup")
        print(f"  MoE      : DualBranchGabor  gate={gmode}(init={ginit},max={gscal})")
        print(f"             base_expert: FedAvg'd | "
              f"domain_expert+gate: local-only ({wstr})")
        print(f"  Eval     : GlobalBase (base only)  +  "
              f"GlobalFull (base + domain_expert[k])")
    print(f"{'='*62}\n")

    # ── data splits ───────────────────────────────────────────────────────────
    dataset_key       = cfg["dataset"].strip().lower()
    model_key         = cfg["model"].strip().lower()
    splits_path       = cfg["splits_path"].format(dataset=dataset_key)
    init_weights_path = cfg["init_weights_path"].format(
                            dataset=dataset_key, model=model_key)

    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    if os.path.exists(splits_path):
        print(f"Loading splits from: {splits_path}")
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
    else:
        print(f"Building splits for {cfg['dataset'].upper()} …")
        splits = get_federated_splits(cfg, seed=seed)
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)
        print(f"  Splits saved.")

    client_data, gallery_samples, probe_samples, test_label_map, domain_names = splits
    num_classes = client_data[0]["num_classes"]
    n_clients   = len(client_data)
    print(f"  Clients: {n_clients}  {domain_names}")
    print(f"  IDs/client: {num_classes}  |  Test classes: {len(test_label_map)}")
    print(f"  Gallery: {len(gallery_samples)}  |  Probe: {len(probe_samples)}\n")

    # ── server + clients ──────────────────────────────────────────────────────
    print("Initialising server …")
    server = FLServer(num_classes, gallery_samples, probe_samples,
                      cfg, device, n_clients)

    print("Initialising clients …")
    clients = [FLClient(
        client_id=i, spectrum=cd["spectrum"],
        train_samples=cd["train_samples"], label_map=cd["label_map"],
        num_classes=num_classes, cfg=cfg, device=device)
        for i, cd in enumerate(client_data)]
    cfg["_n_clients"] = list(range(n_clients))

    # ── init weights ──────────────────────────────────────────────────────────
    if os.path.exists(init_weights_path):
        _probe    = torch.load(init_weights_path, map_location="cpu")
        _ckpt_arc = _probe.get("arc.weight", None)
        _mod_arc  = server.global_model.state_dict()["arc.weight"]
        if _ckpt_arc is None or _ckpt_arc.shape != _mod_arc.shape:
            print("Init weights shape mismatch — regenerating.")
            os.remove(init_weights_path)

    if os.path.exists(init_weights_path):
        print(f"Loading initial weights from: {init_weights_path}")
        init_state = torch.load(init_weights_path, map_location=device)
        server.global_model.load_state_dict(init_state, strict=False)
        backbone_init = server.get_global_weights()
        for client in clients:
            client.set_weights(backbone_init)
        print("  Loaded.")
    else:
        print(f"Saving initial weights to: {init_weights_path}")
        torch.save(server.global_model.state_dict(), init_weights_path)
        backbone_init = server.get_global_weights()
        for client in clients:
            client.set_weights(backbone_init)
        print("  Saved.")

    # ── results file ──────────────────────────────────────────────────────────
    results_path  = os.path.join(base_dir, "results.txt")
    client_header = "\t".join(
        f"C{i}_EER\tC{i}_R1" for i in range(n_clients))
    with open(results_path, "w") as f:
        f.write(f"Rnd\tAug\t"
                f"Base_EER\tBase_R1\tFull_EER\tFull_R1\t"
                f"{client_header}\n")

    # ── style templates ───────────────────────────────────────────────────────
    print("\nExtracting style templates …")
    style_bank_full = {c.client_id: c.extract_style_templates()
                       for c in clients}
    mean_bank_full  = {cid: np.mean(tmpl, axis=0)
                       for cid, tmpl in style_bank_full.items()
                       if len(tmpl) > 0}
    total = sum(len(v) for v in style_bank_full.values())
    print(f"  {total} templates across {len(style_bank_full)} clients\n")

    use_whitening = cfg.get("use_whitening", False)

    # ── Round 0 ───────────────────────────────────────────────────────────────
    print("─" * 62)
    print("  Round 0 (random init)")
    g_eer_b, g_r1_b = server.evaluate_base(use_whitening=use_whitening,
                                            warmup_active=True)
    print(f"  GlobalBase  → EER={g_eer_b*100:.4f}%  Rank-1={g_r1_b:.2f}%")
    print("─" * 62)
    with open(results_path, "a") as f:
        f.write(f"0\tInit\t{g_eer_b*100:.4f}\t{g_r1_b:.2f}\t-\t-\t"
                + "\t".join("-\t-" for _ in range(n_clients)) + "\n")

    recent_base = []; recent_full = []

    # ── FL rounds ─────────────────────────────────────────────────────────────
    for rnd in range(1, cfg["n_rounds"] + 1):
        t_start = time.time()

        active_style_bank = get_active_style_bank(
            style_bank_full, rnd, cfg, is_dinov2)
        mode_label = aug_mode_label(rnd, cfg, is_dinov2)
        rnd_warmup = use_moe and warmup_rds > 0 and rnd <= warmup_rds

        if use_moe and warmup_rds > 0:
            if rnd == 1:
                print(f"\n  ► MoE WARMUP START: rds 1–{warmup_rds} "
                      f"(base_expert only, domain_expert frozen)")
            elif rnd == warmup_rds + 1:
                print(f"\n  ► MoE WARMUP END: rd {rnd} "
                      f"— domain_expert + gate now training")

        global_weights         = server.get_global_weights()
        client_weights         = []
        client_metrics         = []
        client_diag_records    = []
        client_domain_states   = []

        # ── local training ────────────────────────────────────────────────────
        for client in clients:
            client.set_weights(global_weights)
            loss, acc, warmup_active, diags = client.local_train(
                cfg["local_epochs"], active_style_bank, cfg["M"], rnd,
                mean_bank=mean_bank_full
                          if cfg.get("domain_aware_mixing", False) else None)

            # evaluate local model (base only during warmup)
            gal_ids = (None if warmup_active else server.gallery_domain_ids)
            prb_ids = (None if warmup_active else server.probe_domain_ids)
            c_eer, c_rank1 = evaluate_model(
                client.model,
                server.gallery_loader, server.probe_loader, device,
                use_whitening      = use_whitening,
                gallery_domain_ids = gal_ids,
                probe_domain_ids   = prb_ids)

            client_weights.append(client.get_weights())
            client_domain_states.append(client.get_domain_expert_state())

            client_metrics.append({
                "client_id" : client.client_id,
                "spectrum"  : client.spectrum,
                "loss"      : round(loss, 4),
                "acc"       : round(acc, 1),
                "eer"       : round(c_eer, 6),
                "rank1"     : round(c_rank1, 2),
            })
            if use_moe and diags is not None:
                client_diag_records.append({
                    "client_id": client.client_id,
                    "spectrum" : client.spectrum,
                    **diags,
                })

        # ── FedAvg (base_expert only) + collect domain experts ───────────────
        server.aggregate(client_weights)
        if use_moe:
            server.collect_domain_experts(client_domain_states)

        # ── evaluate GlobalBase and GlobalFull ────────────────────────────────
        g_eer_b, g_r1_b = server.evaluate_base(
            use_whitening=use_whitening, warmup_active=rnd_warmup)

        g_eer_f = g_r1_f = None
        if use_moe and not rnd_warmup:
            g_eer_f, g_r1_f = server.evaluate_full(use_whitening=use_whitening)

        elapsed = time.time() - t_start

        avg_rounds = cfg.get("avg_last_rounds", 5)
        recent_base.append((g_eer_b, g_r1_b))
        if len(recent_base) > avg_rounds: recent_base.pop(0)
        if g_eer_f is not None:
            recent_full.append((g_eer_f, g_r1_f))
            if len(recent_full) > avg_rounds: recent_full.pop(0)

        # ── print diagnostics ─────────────────────────────────────────────────
        if use_moe and client_diag_records:
            print_moe_diagnostics(rnd, client_diag_records, rnd_warmup)

        ts = time.strftime("%H:%M:%S")
        print(f"\n  [{ts}] Rnd {rnd:04d}/{cfg['n_rounds']} [{mode_label}]  "
              f"({elapsed:.1f}s)")
        print(f"  GlobalBase  EER={g_eer_b*100:.4f}%  Rank-1={g_r1_b:.2f}%")
        if g_eer_f is not None:
            print(f"  GlobalFull  EER={g_eer_f*100:.4f}%  Rank-1={g_r1_f:.2f}%")
        elif use_moe and rnd_warmup:
            print(f"  GlobalFull  — (warmup, domain_expert not yet active)")

        print(f"\n  {'Clt':>3}  {'Spectrum':>8}  {'Loss':>7}  "
              f"{'Acc%':>5}  {'EER%':>6}  {'R1%':>5}")
        print(f"  {'─'*3}  {'─'*8}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*5}")
        for cm in client_metrics:
            print(f"  C{cm['client_id']}   {cm['spectrum']:>8s}  "
                  f"{cm['loss']:>7.4f}  {cm['acc']:>5.1f}  "
                  f"{cm['eer']*100:>6.3f}  {cm['rank1']:>5.2f}")
        print()

        full_str = (f"{g_eer_f*100:.4f}\t{g_r1_f:.2f}"
                    if g_eer_f is not None else "-\t-")
        client_cols = "\t".join(
            f"{cm['eer']*100:.4f}\t{cm['rank1']:.2f}"
            for cm in client_metrics)
        with open(results_path, "a") as f:
            f.write(f"{rnd}\t{mode_label}\t"
                    f"{g_eer_b*100:.4f}\t{g_r1_b:.2f}\t"
                    f"{full_str}\t{client_cols}\n")

    # ── final summary ─────────────────────────────────────────────────────────
    n_avg = len(recent_base)
    avg_eer_b  = sum(e for e, _ in recent_base) / n_avg
    avg_r1_b   = sum(r for _, r in recent_base) / n_avg

    print(f"\n{'='*62}")
    print(f"  FL COMPLETE — {cfg['n_rounds']} rounds")
    print(f"  Dataset            : {cfg['dataset'].upper()}")
    print(f"  Model              : {cfg['model'].upper()}")
    print(f"  Aug mode           : {aug_mode.upper()}")
    if use_moe:
        print(f"  MoE                : DualBranchGabor  "
              f"gate={cfg.get('moe_gate_mode','scalar')}"
              f"(init={cfg.get('moe_gate_init',1.0)}"
              f",max={cfg.get('moe_gate_scale',2.0)})")
    print(f"  Avg GlobalBase EER : {avg_eer_b*100:.4f}%  (last {n_avg} rds)")
    print(f"  Avg GlobalBase R1  : {avg_r1_b:.2f}%")
    if recent_full:
        n_f = len(recent_full)
        avg_eer_f = sum(e for e, _ in recent_full) / n_f
        avg_r1_f  = sum(r for _, r in recent_full) / n_f
        print(f"  Avg GlobalFull EER : {avg_eer_f*100:.4f}%  (last {n_f} rds)")
        print(f"  Avg GlobalFull R1  : {avg_r1_f:.2f}%")
    print(f"  Results saved      : {results_path}")
    print(f"{'='*62}")

    with open(results_path, "a") as f:
        f.write(f"\n# Average of last {n_avg} rounds\n")
        f.write(f"avg_base\t—\t{avg_eer_b*100:.4f}\t{avg_r1_b:.2f}\t-\t-\n")
        if recent_full:
            f.write(f"avg_full\t—\t-\t-\t{avg_eer_f*100:.4f}\t{avg_r1_f:.2f}\n")


if __name__ == "__main__":
    main()
