# ==============================================================
#  main.py — FLClient, FLServer, and federated training loop
# ==============================================================

import os
import copy
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
                      EvalDatasetDINO,
                      get_federated_splits)
from utils    import (extract_style_template, evaluate_model,
                      train_compnet_epoch, train_ccnet_epoch,
                      CenterLoss)


def make_eval_dataset(samples, cfg):
    """
    Return the correct evaluation dataset for the configured model.
    DINOv2 uses RGB 224×224 with ImageNet normalisation.
    CompNet and CCNet use grayscale 128×128 with NormSingleROI.
    """
    if cfg["model"].strip().lower() == "dinov2":
        return EvalDatasetDINO(samples, cfg.get("dino_img_side", 224))
    return PalmDataset(samples, cfg["img_side"])


# ══════════════════════════════════════════════════════════════
#  FL CLIENT
# ══════════════════════════════════════════════════════════════

class FLClient:
    """
    One federated learning client — owns one spectral/domain with
    a disjoint subset of training identities.

    Model selection
    ───────────────
      cfg["model"] == "compnet" → CompNet, trained with train_compnet_epoch
                                  (grayscale 128×128, ArcFace + CE)
      cfg["model"] == "ccnet"   → CCNet,   trained with train_ccnet_epoch
                                  (grayscale 128×128, ArcFace + CE + SupCon)
      cfg["model"] == "dinov2"  → DINOv2,  trained with train_compnet_epoch
                                  (same single-image forward interface)
                                  Images loaded and augmented as RGB.
                                  FFT augmentation works per-channel on RGB.

    Weight sharing
    ──────────────
      Backbone parameters are shared via FedAvg.
      arc.* is kept local — client-specific identity prototypes.
      center_loss.centres is kept local — per-client class centres
      that are carried over across rounds (never sent to server).

    Optimiser
    ─────────
      Adam (CompNet/CCNet) or AdamW (DINOv2) with constant lr — recreated
      each round since the model resets to the global backbone each round.
      CenterLoss has its own SGD optimiser (center_loss_lr) that persists
      across rounds so centres accumulate knowledge across FL rounds.

    Augmentation modes (controlled by active_style_bank passed per round)
    ──────────────────────────────────────────────────────────────────────
      active_style_bank empty  → AugmentedDataset (spatial aug only)
      active_style_bank filled → FFTAugmentedDataset (FFT + spatial aug)
      Mixed mode is handled in main() by passing an empty bank for the
      first mixed_aug_round rounds and a full bank thereafter — local_train
      needs no special logic for it.
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

        # local model — backbone overwritten each round from global weights
        self.model = build_model(cfg, num_classes).to(device)

        # center loss — kept local, never shared, persists across rounds
        # embed_dim depends on model: 512 (CompNet), 2048 (CCNet), 384 (DINOv2)
        model_name = cfg["model"].strip().lower()
        if cfg.get("use_center_loss", False):
            if model_name == "compnet":
                embed_dim = cfg["embedding_dim"]
            elif model_name == "ccnet":
                embed_dim = 2048
            else:  # dinov2
                embed_dim = 384
            self.center_loss = CenterLoss(num_classes, embed_dim, device)
            self.center_optimizer = optim.SGD(
                self.center_loss.parameters(),
                lr=cfg.get("center_loss_lr", 0.5))
        else:
            self.center_loss      = None
            self.center_optimizer = None

        print(f"  Client {client_id} [{spectrum}] [{cfg['model']}] — "
              f"train IDs: {num_classes}  samples: {len(train_samples)}"
              + (f"  [CenterLoss λ={cfg.get('center_loss_weight', 0.003)}]"
                 if cfg.get("use_center_loss", False) else ""))

    # ── weight management ───────────────────────────────────────────────────

    def set_weights(self, backbone_state_dict):
        """
        Load global backbone weights into local model.
        ArcFace head (arc.*) is never in backbone_state_dict and is
        therefore preserved unchanged — local identity prototypes are kept.
        """
        local_state = self.model.state_dict()
        for key, val in backbone_state_dict.items():
            if key in local_state and local_state[key].shape == val.shape:
                local_state[key] = val.clone()
        self.model.load_state_dict(local_state)

    def get_weights(self):
        """Return backbone weights only — ArcFace head excluded."""
        return {k: v.cpu().clone()
                for k, v in self.model.state_dict().items()
                if not k.startswith("arc.")}

    # ── local training ──────────────────────────────────────────────────────

    def local_train(self, local_epochs, active_style_bank, M, rnd, mean_bank=None):
        """
        Train local model for local_epochs epochs.

        active_style_bank controls augmentation mode this round:
          {}       → AugmentedDataset  (spatial aug only)
          non-empty → FFTAugmentedDataset (FFT + spatial aug)
        This means mixed-mode is transparent here — main() simply passes
        an empty bank for spatial rounds and a full bank for FFT rounds.
        """
        model_name = self.cfg["model"].strip().lower()
        is_dino    = model_name == "dinov2"
        img_side   = self.cfg.get("dino_img_side", 224) if is_dino else self.cfg["img_side"]
        grayscale  = not is_dino

        if model_name in ("compnet", "dinov2"):
            if active_style_bank and M > 1:
                dataset = FFTAugmentedDataset(
                    samples    = self.train_samples,
                    style_bank = active_style_bank,
                    client_id  = self.client_id,
                    beta       = self.cfg["fft_beta"],
                    img_side   = img_side,
                    grayscale  = grayscale,
                    mean_bank       = mean_bank,
                    prefer_distant  = self.cfg.get("prefer_distant_domain", True),
                    use_mean_template = self.cfg.get("use_mean_template", False),
                )
            else:
                dataset = AugmentedDataset(self.train_samples, img_side,
                                           grayscale=grayscale)

        elif model_name == "ccnet":
            # For CCNet, active_style_bank controls FFT on paired views too.
            dataset = PairedDataset(
                samples    = self.train_samples,
                img_side   = img_side,
                style_bank = active_style_bank,
                client_id  = self.client_id,
                beta       = self.cfg["fft_beta"],
            )

        else:
            raise ValueError(f"Unknown model: '{self.cfg['model']}'")

        # deterministic seed per (round, client) — same across all aug runs
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
                self.model.parameters(), lr=self.cfg["lr"],
                weight_decay=self.cfg.get("dino_weight_decay", 1e-4))
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.cfg["lr"])

        lambda_c = self.cfg.get("center_loss_weight", 0.0) \
                   if self.cfg.get("use_center_loss", False) else 0.0

        avg_loss, accuracy = 0.0, 0.0
        for _ in range(local_epochs):
            if model_name in ("compnet", "dinov2"):
                avg_loss, accuracy = train_compnet_epoch(
                    self.model, train_loader, criterion, optimizer, self.device,
                    center_loss      = self.center_loss,
                    center_optimizer = self.center_optimizer,
                    lambda_center    = lambda_c,
                )
            elif model_name == "ccnet":
                avg_loss, accuracy = train_ccnet_epoch(
                    self.model, train_loader, criterion, optimizer, self.device,
                    ce_weight        = self.cfg.get("ce_weight",   0.8),
                    con_weight       = self.cfg.get("con_weight",  0.2),
                    temperature      = self.cfg.get("temperature", 0.07),
                    center_loss      = self.center_loss,
                    center_optimizer = self.center_optimizer,
                    lambda_center    = lambda_c,
                )

        return avg_loss, accuracy

    # ── style template extraction ───────────────────────────────────────────

    def extract_style_templates(self):
        """
        Extract FFT amplitude templates from all local training samples.
        Always called regardless of aug mode so random state is identical
        across spatial, FFT, and mixed runs.

        Grayscale models (CompNet, CCNet): load as L → (H, W) template.
        DINOv2: load as RGB → (H, W, 3) per-channel template.
        Both formats are handled by extract_style_template() in utils.py.
        """
        model_name = self.cfg["model"].strip().lower()
        is_dino    = model_name == "dinov2"
        img_side   = self.cfg.get("dino_img_side", 224) if is_dino \
                     else self.cfg["img_side"]
        mode       = "RGB" if is_dino else "L"

        templates = []
        for path, _ in self.train_samples:
            img    = Image.open(path).convert(mode).resize(
                (img_side, img_side), Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            templates.append(extract_style_template(img_np))
        print(f"  Client {self.client_id} [{self.spectrum}] "
              f"— extracted {len(templates)} {'RGB' if is_dino else 'grayscale'} "
              f"style templates")
        return templates


# ══════════════════════════════════════════════════════════════
#  FL SERVER
# ══════════════════════════════════════════════════════════════

class FLServer:
    """
    Central server:
      - maintains the global model
      - performs FedAvg aggregation on backbone weights only
      - evaluates the global model on the shared gallery/probe test sets
    """

    def __init__(self, num_classes, gallery_samples, probe_samples, cfg, device):
        self.cfg    = cfg
        self.device = device

        self.global_model = build_model(cfg, num_classes).to(device)

        self.gallery_loader = DataLoader(
            make_eval_dataset(gallery_samples, cfg),
            batch_size  = cfg["batch_size"],
            shuffle     = False,
            num_workers = cfg["num_workers"],
            pin_memory  = True,
        )
        self.probe_loader = DataLoader(
            make_eval_dataset(probe_samples, cfg),
            batch_size  = cfg["batch_size"],
            shuffle     = False,
            num_workers = cfg["num_workers"],
            pin_memory  = True,
        )

        print(f"  Server [{cfg['model']}] — "
              f"gallery: {len(gallery_samples)}  probe: {len(probe_samples)}")

    def get_global_weights(self):
        """Return backbone weights only (arc.* excluded)."""
        return {k: v.cpu().clone()
                for k, v in self.global_model.state_dict().items()
                if not k.startswith("arc.")}

    def aggregate(self, client_weight_dicts):
        """FedAvg on backbone parameters only — arc.* never aggregated."""
        n        = len(client_weight_dicts)
        avg_dict = {}
        for key in client_weight_dicts[0].keys():
            stacked      = torch.stack(
                [client_weight_dicts[i][key].float() for i in range(n)], dim=0)
            avg_dict[key] = stacked.mean(dim=0)
        global_state = self.global_model.state_dict()
        global_state.update(avg_dict)
        self.global_model.load_state_dict(global_state)

    def evaluate(self):
        """Evaluate global model on the shared gallery and probe sets."""
        return evaluate_model(
            self.global_model,
            self.gallery_loader,
            self.probe_loader,
            self.device)


# ══════════════════════════════════════════════════════════════
#  AUGMENTATION MODE HELPERS
# ══════════════════════════════════════════════════════════════

def resolve_aug_mode(cfg):
    """
    Determine the augmentation mode from config flags.

    Priority: use_mixed_aug > use_fft_aug > spatial-only (default).

    Returns
    -------
    mode : str   "spatial" | "fft" | "mixed"
    """
    if cfg.get("use_mixed_aug", False):
        return "mixed"
    if cfg.get("use_fft_aug", False):
        return "fft"
    return "spatial"


def get_active_style_bank(style_bank_full, rnd, cfg, is_dinov2):
    """
    Return the style bank to pass to local_train for this round.

    spatial → always {}
    fft     → always style_bank_full
    mixed   → {} for rounds 1..mixed_aug_round
               style_bank_full for rounds mixed_aug_round+1..n_rounds

    DINOv2 now supports FFT augmentation on RGB images (per-channel FFT),
    so no special case is needed.
    """
    mode = resolve_aug_mode(cfg)
    if mode == "spatial":
        return {}
    if mode == "fft":
        return style_bank_full
    # mixed
    switch = cfg.get("mixed_aug_round", cfg["n_rounds"] // 2)
    return style_bank_full if rnd > switch else {}


def aug_mode_label(rnd, cfg, is_dinov2):
    """Short label for console log — shows which mode is active this round."""
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

    device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir   = cfg["base_results_dir"]
    is_dinov2  = cfg["model"].strip().lower() == "dinov2"
    aug_mode   = resolve_aug_mode(cfg)
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
              f"(Spatial rounds 1–{switch}, FFT rounds {switch+1}–{cfg['n_rounds']})"
              f"  M={cfg['M']}  beta={cfg['fft_beta']}")
    else:
        print(f"  Aug mode : {aug_mode.upper()}   "
              f"M={cfg['M']}   beta={cfg['fft_beta']}")
    print(f"  LR       : {cfg['lr']} (constant, no scheduler)")
    print(f"{'='*62}\n")

    # ── resolve dataset- and model-specific paths ─────────────────────────
    dataset_key   = cfg["dataset"].strip().lower()
    model_key     = cfg["model"].strip().lower()
    splits_path       = cfg["splits_path"].format(dataset=dataset_key)
    init_weights_path = cfg["init_weights_path"].format(
                            dataset=dataset_key, model=model_key)

    # ── Step 0a: load or build data splits ────────────────────────────────
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    if os.path.exists(splits_path):
        print(f"Loading existing data splits from: {splits_path}")
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
    print(f"\n  Clients        : {n_clients}  ({domain_names})")
    print(f"  IDs per client : {num_classes}")
    print(f"  Test  classes  : {len(test_label_map)}")
    print(f"  Gallery        : {len(gallery_samples)}  "
          f"Probe : {len(probe_samples)}\n")

    # ── Step 0b: initialise server ────────────────────────────────────────
    print("Initialising server …")
    server = FLServer(num_classes, gallery_samples, probe_samples, cfg, device)

    # ── Step 0c: initialise clients ───────────────────────────────────────
    print("Initialising clients …")
    clients = []
    for i, cd in enumerate(client_data):
        clients.append(FLClient(
            client_id     = i,
            spectrum      = cd["spectrum"],
            train_samples = cd["train_samples"],
            label_map     = cd["label_map"],
            num_classes   = cd["num_classes"],
            cfg           = cfg,
            device        = device,
        ))

    # ── Step 0d: load or save initial model weights ───────────────────────
    if os.path.exists(init_weights_path):
        print(f"\nLoading existing initial weights from: {init_weights_path}")
        init_state = torch.load(init_weights_path, map_location=device)
        server.global_model.load_state_dict(init_state)
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

    # ── results file ──────────────────────────────────────────────────────
    results_path  = os.path.join(base_dir, "results.txt")
    client_header = "\t".join(
        f"Client{i}_EER(%)\tClient{i}_Rank1(%)" for i in range(n_clients))
    with open(results_path, "w") as f:
        f.write(f"Round\tAug_Mode\tGlobal_EER(%)\tGlobal_Rank1(%)\t{client_header}\n")

    # ── Step 0e: extract style templates (always, all runs) ───────────────
    # Always extracted regardless of aug mode so that the random state
    # consumed by extraction is identical across spatial, FFT, and mixed runs.
    # DINOv2 returns empty lists — templates are incompatible with RGB input.
    print("\nExtracting style templates from all clients …")
    style_bank_full = {
        client.client_id: client.extract_style_templates()
        for client in clients
    }
    total = sum(len(v) for v in style_bank_full.values())
    if total > 0:
        print(f"  Style bank ready — {total} templates "
              f"across {len(style_bank_full)} clients")
    if aug_mode == "mixed":
        switch = cfg.get("mixed_aug_round", cfg["n_rounds"] // 2)
        print(f"  Mixed augmentation: Spatial rounds 1–{switch}, "
              f"FFT rounds {switch+1}–{cfg['n_rounds']}.\n")
    elif aug_mode == "fft":
        print("  FFT augmentation ENABLED — style bank will be used.\n")
    else:
        print("  Spatial augmentation only — style bank extracted "
              "but not used during training.\n")

    # compute per-client mean template for domain distance-aware mixing
    # mean_bank: {client_id: mean_amplitude_array}  same shape as one template
    mean_bank = {
        cid: np.mean(templates, axis=0)
        for cid, templates in style_bank_full.items()
        if len(templates) > 0
    }
    print(f"  Mean bank ready — {len(mean_bank)} domain mean templates")

    # ── Round 0: random init evaluation ──────────────────────────────────
    print("\n--- Round 0 (random init) ---")
    g_eer_0, g_rank1_0 = server.evaluate()
    print(f"  [Global init]  EER={g_eer_0*100:.4f}%  Rank-1={g_rank1_0:.2f}%")
    with open(results_path, "a") as f:
        f.write(f"0\tInit\t{g_eer_0*100:.4f}\t{g_rank1_0:.2f}\t"
                + "\t".join("-1\t-1" for _ in range(n_clients)) + "\n")

    g_eer, g_rank1 = g_eer_0, g_rank1_0

    # ── FL rounds ─────────────────────────────────────────────────────────
    for rnd in range(1, cfg["n_rounds"] + 1):
        t_start = time.time()

        # determine which style bank to pass this round
        active_style_bank = get_active_style_bank(
            style_bank_full, rnd, cfg, is_dinov2)
        mode_label = aug_mode_label(rnd, cfg, is_dinov2)

        global_weights = server.get_global_weights()
        client_weights = []
        client_metrics = []

        # ── Step 1: local training ────────────────────────────────────────
        for client in clients:
            client.set_weights(global_weights)
            loss, acc = client.local_train(
                cfg["local_epochs"], active_style_bank, cfg["M"], rnd,
                mean_bank = mean_bank if cfg.get("domain_aware_mixing", False) else None,
            )
            c_eer, c_rank1 = evaluate_model(
                client.model,
                server.gallery_loader, server.probe_loader,
                device)
            client_weights.append(client.get_weights())
            client_metrics.append({
                "client_id" : client.client_id,
                "spectrum"  : client.spectrum,
                "train_loss": round(loss, 6),
                "train_acc" : round(acc, 3),
                "eer"       : round(c_eer, 6),
                "rank1"     : round(c_rank1, 3),
            })

        # ── Step 2: FedAvg (backbone only) + global evaluation ────────────
        server.aggregate(client_weights)
        g_eer, g_rank1 = server.evaluate()
        elapsed = time.time() - t_start

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Round {rnd:04d}/{cfg['n_rounds']} [{mode_label}] | "
              f"Global EER={g_eer*100:.4f}%  Rank-1={g_rank1:.2f}%  "
              f"({elapsed:.1f}s)")
        for cm in client_metrics:
            print(f"  Client {cm['client_id']} [{cm['spectrum']:>14}] | "
                  f"loss={cm['train_loss']:.4f}  acc={cm['train_acc']:.1f}%  "
                  f"EER={cm['eer']*100:.3f}%  R1={cm['rank1']:.1f}%")

        client_cols = "\t".join(
            f"{cm['eer']*100:.4f}\t{cm['rank1']:.2f}"
            for cm in client_metrics)
        with open(results_path, "a") as f:
            f.write(f"{rnd}\t{mode_label}\t{g_eer*100:.4f}\t{g_rank1:.2f}"
                    f"\t{client_cols}\n")

    # ── Final reporting ───────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  FL COMPLETE — {cfg['n_rounds']} rounds")
    print(f"  Dataset            : {cfg['dataset'].upper()}")
    print(f"  Model              : {cfg['model'].upper()}")
    print(f"  Aug mode           : {aug_mode.upper()}")
    print(f"  Final Global EER   : {g_eer*100:.4f}%")
    print(f"  Final Global Rank-1: {g_rank1:.2f}%")
    print(f"  Results saved to   : {results_path}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
