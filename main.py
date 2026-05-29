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
from utils    import (extract_style_template, extract_radial_template,
                      evaluate_model,
                      train_compnet_epoch, train_ccnet_epoch,
                      CenterLoss)


def make_eval_dataset(samples, cfg):
    if cfg["model"].strip().lower() == "dinov2":
        return EvalDatasetDINO(samples, cfg.get("dino_img_side", 224))
    return PalmDataset(samples, cfg["img_side"])


# ══════════════════════════════════════════════════════════════
#  FL CLIENT
# ══════════════════════════════════════════════════════════════

class FLClient:
    """
    One federated learning client.

    Weight sharing
    ──────────────
    Backbone parameters are shared via FedAvg.
    arc.* is kept local — client-specific identity prototypes.
    center_loss.centres is kept local across rounds.

    MoE warmup transparency
    ───────────────────────
    The client is unaware of whether MoE is active — it simply trains
    whatever model state it receives from set_weights().  When the server
    activates MoE at round moe_warmup_round, the upgraded global weights
    are pushed to all clients via set_weights() before the next local_train.
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
            if model_name == "compnet":
                embed_dim = cfg["embedding_dim"]
            elif model_name == "ccnet":
                embed_dim = 2048
            else:
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

    # ── weight management ────────────────────────────────────────────────────

    def set_weights(self, backbone_state_dict):
        """
        Load global backbone weights into local model.

        After MoE activation the global state dict will contain MoEFC /
        MoELayerNorm keys that don't exist in the client model yet.  We do a
        strict=False load so that:
          - New keys (expert weights) are loaded correctly into the upgraded
            client model (the server has already pushed the activated model).
          - Missing keys (e.g. old plain LayerNorm after upgrade) are ignored.
        The client model itself is also upgraded by the server before this is
        called — see FLServer.activate_moe_on_clients().
        """
        local_state = self.model.state_dict()
        for key, val in backbone_state_dict.items():
            if key in local_state and local_state[key].shape == val.shape:
                local_state[key] = val.clone()
        self.model.load_state_dict(local_state)

    def get_weights(self):
        """Return backbone weights for FedAvg — ArcFace head excluded."""
        return {k: v.cpu().clone()
                for k, v in self.model.state_dict().items()
                if not k.startswith("arc.")}

    def upgrade_model_for_moe(self, moe_position):
        """
        Replace the client's local model with an MoE-activated copy.

        Called by the server on each client object immediately after
        server.activate_moe(), before set_weights() distributes the new
        global weights.  This ensures the client model's architecture
        matches the incoming state dict.
        """
        self.model.activate_moe(moe_position)

    # ── local training ───────────────────────────────────────────────────────

    def local_train(self, local_epochs, active_style_bank, M, rnd,
                    mean_bank=None):
        model_name = self.cfg["model"].strip().lower()
        is_dino    = model_name == "dinov2"
        img_side   = self.cfg.get("dino_img_side", 224) if is_dino else self.cfg["img_side"]
        grayscale  = not is_dino

        if model_name in ("compnet", "dinov2"):
            if active_style_bank and M > 1:
                fft_method  = self.cfg.get("fft_aug_method", "amplitude")
                local_only  = self.cfg.get("fft_local_only", False)
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
                    fft_method           = fft_method,
                    local_only           = local_only,
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
                    lambda_style     = self.cfg.get("lambda_style", 0.0),
                    lambda_grl       = self.cfg.get("lambda_grl", 0.0)
                                       if self.cfg.get("use_grl", False) else 0.0,
                    lambda_load_balance = self.cfg.get("lambda_load_balance", 0.0)
                                         if self.cfg.get("use_moe", False) else 0.0,
                    lambda_supcon    = self.cfg.get("lambda_supcon", 0.0)
                                       if self.cfg.get("use_supcon", False) else 0.0,
                    temperature      = self.cfg.get("temperature", 0.07),
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

    # ── style template extraction ─────────────────────────────────────────────

    def extract_style_templates(self):
        model_name = self.cfg["model"].strip().lower()
        is_dino    = model_name == "dinov2"
        img_side   = self.cfg.get("dino_img_side", 224) if is_dino \
                     else self.cfg["img_side"]
        mode       = "RGB" if is_dino else "L"
        fft_method = self.cfg.get("fft_aug_method", "amplitude")

        templates = []
        for path, _ in self.train_samples:
            img    = Image.open(path).convert(mode).resize(
                (img_side, img_side), Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            if fft_method == "radial" and not is_dino:
                templates.append(extract_radial_template(img_np))
            else:
                templates.append(extract_style_template(img_np))

        method_label = f"radial({self.cfg.get('fft_beta',0.1)})" \
                       if fft_method == "radial" else "amplitude"
        print(f"  Client {self.client_id} [{self.spectrum}] "
              f"— {len(templates)} templates  [{method_label}]")
        return templates


# ══════════════════════════════════════════════════════════════
#  FL SERVER
# ══════════════════════════════════════════════════════════════

class FLServer:
    """
    Central server.

    Deferred MoE activation
    ───────────────────────
    After aggregation at round cfg["moe_warmup_round"] the server calls
    activate_moe() which:
      1. Calls global_model.activate_moe() — upgrades the global model
         architecture in-place (MoEFC + MoELayerNorm) and warm-starts
         every expert from the trained shared base.
      2. Calls client.upgrade_model_for_moe() on every client — upgrades
         each client model architecture to match the new global model.
      3. Calls set_weights() on every client with the upgraded global weights
         so all clients start phase 2 from the same warm-started checkpoint.
    """

    def __init__(self, num_classes, gallery_samples, probe_samples,
                 cfg, device, clients):
        self.cfg     = cfg
        self.device  = device
        self.clients = clients   # reference to client list for MoE activation

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
        return {k: v.cpu().clone()
                for k, v in self.global_model.state_dict().items()
                if not k.startswith("arc.")}

    def aggregate(self, client_weight_dicts):
        """FedAvg on backbone parameters — arc.* never aggregated."""
        n        = len(client_weight_dicts)
        avg_dict = {}
        for key in client_weight_dicts[0].keys():
            stacked      = torch.stack(
                [client_weight_dicts[i][key].float() for i in range(n)], dim=0)
            avg_dict[key] = stacked.mean(dim=0)
        global_state = self.global_model.state_dict()
        global_state.update(avg_dict)
        self.global_model.load_state_dict(global_state)

    def activate_moe(self):
        """
        Upgrade global model and all clients to full MoE mode.

        Called once by the FL loop immediately after aggregation at round
        cfg["moe_warmup_round"].  Order matters:
          1. Upgrade global model (activate_moe modifies it in-place).
          2. Upgrade each client model architecture.
          3. Push upgraded global weights to every client.

        After this call, all subsequent rounds run with full MoE routing.
        """
        moe_position = self.cfg.get("moe_position", "both")
        print(f"\n{'─'*56}")
        print(f"  MoE activation — position='{moe_position}'")
        print(f"  Warm-starting {self.cfg.get('n_experts', 6)} experts "
              f"from trained shared base …")

        # 1. upgrade global model
        self.global_model.activate_moe(moe_position)

        # 2. upgrade each client model to match the new architecture
        for client in self.clients:
            client.upgrade_model_for_moe(moe_position)

        # 3. distribute upgraded global weights to all clients
        global_weights = self.get_global_weights()
        for client in self.clients:
            client.set_weights(global_weights)

        print(f"  All clients upgraded and synced.")
        print(f"{'─'*56}\n")

    def evaluate(self, use_whitening=False):
        return evaluate_model(
            self.global_model,
            self.gallery_loader, self.probe_loader, self.device,
            use_whitening=use_whitening)


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

    device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir   = cfg["base_results_dir"]
    is_dinov2  = cfg["model"].strip().lower() == "dinov2"
    aug_mode   = resolve_aug_mode(cfg)
    warmup_rnd = cfg.get("moe_warmup_round", 0)
    use_moe    = cfg.get("use_moe", False)
    os.makedirs(base_dir, exist_ok=True)

    # ── header ────────────────────────────────────────────────────────────────
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
    if use_moe and cfg["model"].strip().lower() == "compnet":
        if warmup_rnd > 0:
            print(f"  MoE      : DEFERRED  position='{cfg.get('moe_position','both')}'  "
                  f"warmup={warmup_rnd} rounds  "
                  f"(shared LoRA base → warm-start experts at round {warmup_rnd})")
        else:
            print(f"  MoE      : ACTIVE from round 1  "
                  f"position='{cfg.get('moe_position','both')}'  "
                  f"experts={cfg.get('n_experts',6)}  rank={cfg.get('lora_rank',64)}")
    print(f"  LR       : {cfg['lr']} (constant, no scheduler)")
    print(f"{'='*62}\n")

    # ── resolve paths ─────────────────────────────────────────────────────────
    dataset_key   = cfg["dataset"].strip().lower()
    model_key     = cfg["model"].strip().lower()
    splits_path       = cfg["splits_path"].format(dataset=dataset_key)
    init_weights_path = cfg["init_weights_path"].format(
                            dataset=dataset_key, model=model_key)

    # ── Step 0a: load or build data splits ────────────────────────────────────
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

    # ── Step 0b: initialise clients ───────────────────────────────────────────
    # Clients are created before the server so the server can hold a reference.
    print("Initialising clients …")
    clients = []
    for i, cd in enumerate(client_data):
        clients.append(FLClient(
            client_id     = i,
            spectrum      = cd["spectrum"],
            train_samples = cd["train_samples"],
            label_map     = cd["label_map"],
            num_classes   = num_classes,
            cfg           = cfg,
            device        = device,
        ))
    cfg["_n_clients"] = list(range(n_clients))

    # ── Step 0c: initialise server ────────────────────────────────────────────
    print("Initialising server …")
    server = FLServer(num_classes, gallery_samples, probe_samples,
                      cfg, device, clients)

    # ── Step 0d: load or save initial model weights ───────────────────────────
    if os.path.exists(init_weights_path):
        _probe    = torch.load(init_weights_path, map_location="cpu")
        _ckpt_arc = _probe.get("arc.weight", None)
        _mod_arc  = server.global_model.state_dict()["arc.weight"]
        if _ckpt_arc is None or _ckpt_arc.shape != _mod_arc.shape:
            print(f"\nInit weights arc.weight shape mismatch — regenerating.")
            os.remove(init_weights_path)

    if os.path.exists(init_weights_path):
        print(f"\nLoading existing initial weights from: {init_weights_path}")
        init_state = torch.load(init_weights_path, map_location=device)
        missing, unexpected = server.global_model.load_state_dict(
            init_state, strict=False)
        if missing:
            print(f"  INFO: {len(missing)} new key(s) not in checkpoint "
                  f"(fresh init): {missing[:4]}{'...' if len(missing)>4 else ''}")
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
        f.write(f"Round\tAug_Mode\tMoE\tGlobal_EER(%)\tGlobal_Rank1(%)\t{client_header}\n")

    # ── Step 0e: extract style templates ──────────────────────────────────────
    print("\nExtracting style templates from all clients …")
    style_bank_full = {
        client.client_id: client.extract_style_templates()
        for client in clients
    }
    total = sum(len(v) for v in style_bank_full.values())
    if total > 0:
        print(f"  Style bank ready — {total} templates "
              f"across {len(style_bank_full)} clients")

    mean_bank_full = {
        cid: np.mean(templates, axis=0)
        for cid, templates in style_bank_full.items()
        if len(templates) > 0
    }

    use_whitening = cfg.get("use_whitening", False)

    # ── Round 0: random init evaluation ───────────────────────────────────────
    print("\n--- Round 0 (random init) ---")
    g_eer_0, g_rank1_0 = server.evaluate(use_whitening=use_whitening)
    print(f"  [Global init]  EER={g_eer_0*100:.4f}%  Rank-1={g_rank1_0:.2f}%")
    with open(results_path, "a") as f:
        f.write(f"0\tInit\tFalse\t{g_eer_0*100:.4f}\t{g_rank1_0:.2f}\t"
                + "\t".join("-1\t-1" for _ in range(n_clients)) + "\n")

    g_eer, g_rank1 = g_eer_0, g_rank1_0
    recent_history = []
    moe_activated  = False   # tracks whether activate_moe() has been called

    # ── FL rounds ─────────────────────────────────────────────────────────────
    for rnd in range(1, cfg["n_rounds"] + 1):
        t_start = time.time()

        active_style_bank = get_active_style_bank(
            style_bank_full, rnd, cfg, is_dinov2)
        mode_label = aug_mode_label(rnd, cfg, is_dinov2)

        # ── MoE deferred activation ──────────────────────────────────────────
        # Triggered AFTER aggregation at the warmup round (i.e. at the start
        # of round warmup_rnd + 1).  We check moe_activated to ensure it
        # runs exactly once regardless of n_rounds configuration.
        if (use_moe and not moe_activated
                and warmup_rnd > 0
                and rnd == warmup_rnd + 1
                and cfg["model"].strip().lower() == "compnet"):
            server.activate_moe()
            moe_activated = True

        moe_label = str(moe_activated or (use_moe and warmup_rnd == 0))

        global_weights = server.get_global_weights()
        client_weights = []
        client_metrics = []

        # ── Step 1: local training ────────────────────────────────────────────
        for client in clients:
            client.set_weights(global_weights)
            loss, acc = client.local_train(
                cfg["local_epochs"], active_style_bank, cfg["M"], rnd,
                mean_bank = mean_bank_full
                            if cfg.get("domain_aware_mixing", False)
                            else None)
            c_eer, c_rank1 = evaluate_model(
                client.model,
                server.gallery_loader, server.probe_loader,
                device,
                use_whitening=use_whitening)
            client_weights.append(client.get_weights())
            client_metrics.append({
                "client_id" : client.client_id,
                "spectrum"  : client.spectrum,
                "train_loss": round(loss, 6),
                "train_acc" : round(acc, 3),
                "eer"       : round(c_eer, 6),
                "rank1"     : round(c_rank1, 3),
            })

        # ── Step 2: FedAvg + global evaluation ───────────────────────────────
        server.aggregate(client_weights)
        g_eer, g_rank1 = server.evaluate(use_whitening=use_whitening)
        elapsed = time.time() - t_start

        avg_rounds = cfg.get("avg_last_rounds", 5)
        recent_history.append((g_eer, g_rank1))
        if len(recent_history) > avg_rounds:
            recent_history.pop(0)

        # ── console log ───────────────────────────────────────────────────────
        moe_tag = ""
        if use_moe and cfg["model"].strip().lower() == "compnet":
            if warmup_rnd > 0 and not moe_activated and rnd <= warmup_rnd:
                moe_tag = f" [warmup {rnd}/{warmup_rnd}]"
            elif moe_activated or warmup_rnd == 0:
                moe_tag = f" [MoE:{cfg.get('moe_position','both')}]"

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Round {rnd:04d}/{cfg['n_rounds']} [{mode_label}]{moe_tag} | "
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
            f.write(f"{rnd}\t{mode_label}\t{moe_label}\t"
                    f"{g_eer*100:.4f}\t{g_rank1:.2f}\t{client_cols}\n")

    # ── Final reporting ───────────────────────────────────────────────────────
    n_avg     = len(recent_history)
    avg_eer   = sum(e for e, _ in recent_history) / n_avg
    avg_rank1 = sum(r for _, r in recent_history) / n_avg

    print(f"\n{'='*62}")
    print(f"  FL COMPLETE — {cfg['n_rounds']} rounds")
    print(f"  Dataset            : {cfg['dataset'].upper()}")
    print(f"  Model              : {cfg['model'].upper()}")
    print(f"  Aug mode           : {aug_mode.upper()}")
    if use_moe and cfg["model"].strip().lower() == "compnet":
        print(f"  MoE position       : {cfg.get('moe_position','both')}")
        print(f"  MoE warmup rounds  : {warmup_rnd}")
    print(f"  Avg Global EER     : {avg_eer*100:.4f}%  (last {n_avg} rounds)")
    print(f"  Avg Global Rank-1  : {avg_rank1:.2f}%  (last {n_avg} rounds)")
    print(f"  Results saved to   : {results_path}")
    print(f"{'='*62}")

    with open(results_path, "a") as f:
        f.write(f"\n# Average of last {n_avg} rounds\n")
        f.write(f"avg_{n_avg}\t—\t—\t{avg_eer*100:.4f}\t{avg_rank1:.2f}\n")


if __name__ == "__main__":
    main()
