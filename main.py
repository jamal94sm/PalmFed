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
                      build_federated_splits)
from utils    import (extract_style_template, evaluate_model,
                      train_compnet_epoch, train_ccnet_epoch)


# ══════════════════════════════════════════════════════════════
#  FL CLIENT
# ══════════════════════════════════════════════════════════════

class FLClient:
    """
    One federated learning client — owns one spectral domain with
    a disjoint subset of training identities.

    Model selection
    ───────────────
      cfg["model"] == "compnet" → CompNet, trained with train_compnet_epoch
      cfg["model"] == "ccnet"   → CCNet,   trained with train_ccnet_epoch
                                  (uses PairedDataset; returns paired images)

    Weight sharing
    ──────────────
      Only backbone parameters are shared via FedAvg.
      The ArcFace head (arc.*) is kept local — it encodes client-specific
      identity prototypes that are semantically different across clients
      even though the weight shapes are identical.

    Optimiser
    ─────────
      Adam with constant lr — recreated each round since the model
      resets to the global backbone checkpoint at the start of every round.
      Carrying over momentum from a previous starting point would be
      meaningless and potentially harmful.
    """

    def __init__(self, client_id, spectrum, train_samples, label_map,
                 num_classes, cfg, device):
        self.client_id     = client_id
        self.spectrum      = spectrum
        self.train_samples = train_samples   # raw list of (path, label)
        self.label_map     = label_map
        self.num_classes   = num_classes
        self.cfg           = cfg
        self.device        = device

        # local model — backbone overwritten each round from global weights
        self.model = build_model(cfg, num_classes).to(device)

        print(f"  Client {client_id} [{spectrum}] [{cfg['model']}] — "
              f"train IDs: {num_classes}  samples: {len(train_samples)}")

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

    def local_train(self, local_epochs, style_bank, M, rnd):
        """
        Train local model for local_epochs epochs using the appropriate
        dataset and training function for the configured model.

        CompNet: AugmentedDataset or FFTAugmentedDataset + train_compnet_epoch
        CCNet  : PairedDataset                           + train_ccnet_epoch

        DataLoader workers are seeded deterministically per (round, client_id)
        so that shuffling order is identical between augmented and baseline runs.
        """
        model_name = self.cfg["model"].strip().lower()
        img_side   = self.cfg["img_side"]

        if model_name == "compnet":
            if self.cfg["use_fft_aug"] and style_bank and M > 1:
                dataset = FFTAugmentedDataset(
                    samples    = self.train_samples,
                    style_bank = style_bank,
                    client_id  = self.client_id,
                    M          = M,
                    beta       = self.cfg["fft_beta"],
                    img_side   = img_side,
                )
            else:
                dataset = AugmentedDataset(self.train_samples, img_side)

        elif model_name == "ccnet":
            # CCNet always uses paired images for SupConLoss
            # FFT augmentation for CCNet can be added as a future extension
            dataset = PairedDataset(self.train_samples, img_side)

        else:
            raise ValueError(f"Unknown model: '{self.cfg['model']}'")

        # deterministic seed per (round, client) — same across both runs
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

        # constant lr — no scheduler (model resets to global weights each round)
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        criterion = nn.CrossEntropyLoss()

        avg_loss, accuracy = 0.0, 0.0
        for _ in range(local_epochs):
            if model_name == "compnet":
                avg_loss, accuracy = train_compnet_epoch(
                    self.model, train_loader, criterion, optimizer, self.device)
            elif model_name == "ccnet":
                avg_loss, accuracy = train_ccnet_epoch(
                    self.model, train_loader, criterion, optimizer, self.device,
                    ce_weight   = self.cfg.get("ce_weight",   0.8),
                    con_weight  = self.cfg.get("con_weight",  0.2),
                    temperature = self.cfg.get("temperature", 0.07),
                )

        return avg_loss, accuracy

    # ── style template extraction ───────────────────────────────────────────

    def extract_style_templates(self):
        """
        Extract low-frequency amplitude templates from all local training samples.
        Safe to share — captures only global style/illumination, not identity info.
        Always called regardless of use_fft_aug to keep random state identical
        between augmented and baseline runs.
        Returns one amplitude array per training sample.
        """
        templates = []
        img_side  = self.cfg["img_side"]
        for path, _ in self.train_samples:
            img = Image.open(path).convert("L").resize(
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
    """
    Central server:
      - maintains the global model
      - performs FedAvg aggregation on backbone weights only
      - evaluates the global model on the shared gallery/probe test sets

    The global model's ArcFace head is never used during evaluation
    (get_embedding() bypasses it). Its weights are a placeholder only.
    """

    def __init__(self, num_classes, gallery_samples, probe_samples, cfg, device):
        self.cfg    = cfg
        self.device = device

        self.global_model = build_model(cfg, num_classes).to(device)

        self.gallery_loader = DataLoader(
            PalmDataset(gallery_samples, cfg["img_side"]),
            batch_size  = cfg["batch_size"],
            shuffle     = False,
            num_workers = cfg["num_workers"],
            pin_memory  = True,
        )
        self.probe_loader = DataLoader(
            PalmDataset(probe_samples, cfg["img_side"]),
            batch_size  = cfg["batch_size"],
            shuffle     = False,
            num_workers = cfg["num_workers"],
            pin_memory  = True,
        )

        print(f"  Server [{cfg['model']}] — "
              f"gallery: {len(gallery_samples)}  probe: {len(probe_samples)}")

    def get_global_weights(self):
        """
        Return backbone weights only (arc.* excluded).
        Clients only load backbone keys, so sending arc.* is unnecessary.
        """
        return {k: v.cpu().clone()
                for k, v in self.global_model.state_dict().items()
                if not k.startswith("arc.")}

    def aggregate(self, client_weight_dicts):
        """
        FedAvg on backbone parameters only.
        ArcFace head is client-specific (different identity prototypes per client)
        and must never be shared or averaged.
        """
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
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    cfg  = CONFIG
    seed = cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir = cfg["base_results_dir"]
    os.makedirs(base_dir, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Federated Learning — Palmprint (CASIA-MS)")
    print(f"  Protocol : Open-Set, Non-Shared-ID, Cross-Domain")
    print(f"  Model    : {cfg['model'].upper()}")
    print(f"  Device   : {device}")
    print(f"  Rounds   : {cfg['n_rounds']}   "
          f"Local epochs/round : {cfg['local_epochs']}")
    print(f"  IDs : {cfg['n_ids']}   "
          f"Test ID ratio : {cfg['k_test']*100:.0f}%   "
          f"Gallery ratio : {cfg['gallery_ratio']*100:.0f}%")
    print(f"  FFT Aug  : {cfg['use_fft_aug']}   "
          f"M={cfg['M']}   beta={cfg['fft_beta']}")
    print(f"  LR       : {cfg['lr']} (constant, no scheduler)")
    print(f"{'='*62}\n")

    # ── Step 0a: load or build data splits ────────────────────────────────
    # Saved on first run; reloaded on subsequent runs so that both
    # augmented and baseline experiments use identical splits.
    splits_path = cfg["splits_path"]
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)

    if os.path.exists(splits_path):
        print(f"Loading existing data splits from: {splits_path}")
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        client_data, gallery_samples, probe_samples, test_label_map, spectra = splits
        print("  Splits loaded.")
    else:
        print("Building federated data splits …")
        splits = build_federated_splits(
            cfg["data_root"], cfg["n_ids"], cfg["k_test"],
            cfg["gallery_ratio"], seed=seed)
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)
        print(f"  Splits saved to: {splits_path}")
        client_data, gallery_samples, probe_samples, test_label_map, spectra = splits

    num_classes = client_data[0]["num_classes"]   # same for all clients (min_ids)
    n_clients   = len(client_data)
    print(f"\n  Clients        : {n_clients}  ({spectra})")
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
    # Both runs start from identical weights so any performance difference
    # is solely attributable to the augmentation strategy.
    init_weights_path = cfg["init_weights_path"]

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
        f.write(f"Round\tGlobal_EER(%)\tGlobal_Rank1(%)\t{client_header}\n")

    # ── Step 0e: extract style templates (always, both runs) ──────────────
    # Always extracted regardless of use_fft_aug — ensures identical random
    # state consumption in both runs. Passed to local_train only when enabled.
    print("\nExtracting style templates from all clients …")
    style_bank_full = {
        client.client_id: client.extract_style_templates()
        for client in clients
    }
    total = sum(len(v) for v in style_bank_full.values())
    print(f"  Style bank ready — {total} templates "
          f"across {len(style_bank_full)} clients")

    style_bank = style_bank_full if cfg["use_fft_aug"] else {}
    if cfg["use_fft_aug"]:
        print("  FFT augmentation ENABLED — style bank will be used.\n")
    else:
        print("  FFT augmentation DISABLED — style bank extracted "
              "but not used during training.\n")

    # ── Round 0: random init evaluation ──────────────────────────────────
    print("\n--- Round 0 (random init) ---")
    g_eer_0, g_rank1_0 = server.evaluate()
    print(f"  [Global init]  EER={g_eer_0*100:.4f}%  Rank-1={g_rank1_0:.2f}%")
    with open(results_path, "a") as f:
        f.write(f"0\t{g_eer_0*100:.4f}\t{g_rank1_0:.2f}\t"
                + "\t".join("-1\t-1" for _ in range(n_clients)) + "\n")

    # pre-initialise so final print is defined even if n_rounds=0
    g_eer, g_rank1 = g_eer_0, g_rank1_0

    # ── FL rounds ─────────────────────────────────────────────────────────
    for rnd in range(1, cfg["n_rounds"] + 1):
        t_start        = time.time()
        global_weights = server.get_global_weights()   # backbone only
        client_weights = []
        client_metrics = []

        # ── Step 1: local training ────────────────────────────────────────
        for client in clients:
            client.set_weights(global_weights)   # load backbone; keep local arc
            loss, acc = client.local_train(
                cfg["local_epochs"], style_bank, cfg["M"], rnd)
            c_eer, c_rank1 = evaluate_model(
                client.model,
                server.gallery_loader, server.probe_loader,
                device)
            client_weights.append(client.get_weights())   # backbone only
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
        print(f"[{ts}] Round {rnd:04d}/{cfg['n_rounds']} | "
              f"Global EER={g_eer*100:.4f}%  Rank-1={g_rank1:.2f}%  "
              f"({elapsed:.1f}s)")
        for cm in client_metrics:
            print(f"  Client {cm['client_id']} [{cm['spectrum']:>6}] | "
                  f"loss={cm['train_loss']:.4f}  acc={cm['train_acc']:.1f}%  "
                  f"EER={cm['eer']*100:.3f}%  R1={cm['rank1']:.1f}%")

        client_cols = "\t".join(
            f"{cm['eer']*100:.4f}\t{cm['rank1']:.2f}"
            for cm in client_metrics)
        with open(results_path, "a") as f:
            f.write(f"{rnd}\t{g_eer*100:.4f}\t{g_rank1:.2f}\t{client_cols}\n")

    # ── Final reporting ───────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  FL COMPLETE — {cfg['n_rounds']} rounds")
    print(f"  Model              : {cfg['model'].upper()}")
    print(f"  Final Global EER   : {g_eer*100:.4f}%")
    print(f"  Final Global Rank-1: {g_rank1:.2f}%")
    print(f"  Results saved to   : {results_path}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
