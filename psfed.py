# ==============================================================
#  train_psfed.py
#  PSFed-Palm adapted for CASIA-MS / XJTU
#  open-set, cross-domain, non-shared-ID evaluation
#
#  Core algorithm kept intact from:
#    Yang et al., "Physics-Driven Spectrum-Consistent FL for
#    Palmprint Verification", IJCV 2024
#
#  Adaptations (dataset loading + evaluation only):
#    - All hyperparameters loaded from configs_psfed.py
#    - Dataset splits via build_federated_splits (shared with main.py)
#    - Global label mapping (identical to FedPalm baseline)
#    - FedPalmDataset with M-fold augmentation for fair comparison
#    - Evaluation: EER + Rank-1 on shared gallery/probe sets
#    - Reports global model + local avg metrics
# ==============================================================

# ──────────────────────────────────────────────────────────────
#  ALGORITHM STEPS (aligned with paper §III and original main.py)
#
#  Spectrum grouping:
#    SHORT clients: visible spectra (460, 630, 700, WHT)
#    LONG  clients: NIR spectra     (850, 940)
#
#  Two sub-anchor models (server-side):
#    visib_net ← FedAvg of SHORT clients' local models
#    invis_net ← FedAvg of LONG  clients' local models
#
#  Step 1 — Local training (per client i, per round r):
#    For each mini-batch with two augmented views (x, x_con):
#      a) output, fe1 = θ_i(x)                  local CE + features
#      b) _,     fe2 = θ_i(x_con)               second view features
#      c) _,     fe3 = anchor(x)                 cross-spectrum anchor
#         [SHORT client → anchor = invis_net (NIR)]
#         [LONG  client → anchor = visib_net (VIS)]
#      d) Loss:
#           L = w1 × CE(output, y)
#             + w2 × SupCon([fe1, fe2], y)
#             + w3 × MSE(fe1, fe3.detach())
#             + mu/2 × ||θ_i − server||
#             + mu/2 × ||θ_i − anchor||
#      e) Update θ_i only.
#
#  Step 2 — Aggregation:
#    visib_net ← FedAvg({θ_i : i ∈ SHORT})   sub-group aggregation
#    invis_net ← FedAvg({θ_i : i ∈ LONG})    sub-group aggregation
#    server    ← FedAvg({θ_i : i ∈ ALL})     global aggregation
#    θ_i       ← server  ∀i                   broadcast global model
#
#  Step 3 — Evaluation:
#    (a) Global model: server.get_embedding(x)
#    (b) Local-avg:    per-client EER → mean
# ──────────────────────────────────────────────────────────────

import os
import copy
import time
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from torchvision import transforms as T
from PIL import Image
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────
from configs import (CONFIG_PSFED as cfg,
                     CASIAMS_SHORT_SPECTRA, CASIAMS_LONG_SPECTRA,
                     XJTU_SHORT_SPECTRA,    XJTU_LONG_SPECTRA)

# ── Our framework ─────────────────────────────────────────────
from datasets import (build_federated_splits, build_federated_splits_xjtu,
                      get_federated_splits,
                      PalmDataset, NormSingleROI)
from utils import compute_eer

# ── PSFed-Palm model (original paper backbone) ─────────────────
from model_fedpalm import compnet_fedpalm

# ── PSFed-Palm loss (original paper) ──────────────────────────
from loss_fedpalm import SupConLoss


# ══════════════════════════════════════════════════════════════
#  DATASET — paired views with M-fold augmentation
# ══════════════════════════════════════════════════════════════

class PSFedDataset(Dataset):
    """
    Paired dataset for PSFed-Palm SupCon training with M-fold augmentation.

    Virtual dataset has M×N entries — each original sample appears M times
    per epoch, each time as a spatially augmented view. With M=2 the effective
    dataset size doubles, matching the proposed method's FFTAugmentedDataset
    M=2 for fair comparison.

    Both img1 and img2 are always spatially augmented — matching the original
    PSFed-Palm MyDataset which applies augmentation unconditionally to all
    samples. img2 is drawn from any of the M×N virtual entries of the same
    identity so both original and augmented occurrences can be paired.

    Returns ([img1, img2], label).
    """
    def __init__(self, samples, img_side=128, M=1):
        self.samples  = samples
        self.img_side = img_side
        self.M        = M

        self.label2idxs = defaultdict(list)
        for i, (_, lab) in enumerate(samples):
            for m in range(M):
                self.label2idxs[lab].append(i * M + m)

        self.transforms = T.Compose([
            T.Resize(img_side),
            T.RandomChoice([
                T.ColorJitter(brightness=0, contrast=0.05,
                              saturation=0, hue=0),
                T.RandomResizedCrop(img_side, scale=(0.8, 1.0),
                                    ratio=(1.0, 1.0)),
                T.RandomPerspective(distortion_scale=0.15, p=1.0),
                T.RandomChoice([
                    T.RandomRotation(10, expand=False,
                                     center=(int(0.5*img_side), 0)),
                    T.RandomRotation(10, expand=False,
                                     center=(0, int(0.5*img_side))),
                ]),
            ]),
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])

    def __len__(self):
        return len(self.samples) * self.M

    def _load(self, path):
        return self.transforms(Image.open(path).convert("L"))

    def __getitem__(self, idx):
        sample_idx  = idx // self.M
        path, label = self.samples[sample_idx]
        img1        = self._load(path)

        idx2        = random.choice(self.label2idxs[label])
        path2, _    = self.samples[idx2 // self.M]
        img2        = self._load(path2)

        return [img1, img2], label


# ══════════════════════════════════════════════════════════════
#  SPECTRUM GROUPING
# ══════════════════════════════════════════════════════════════

def build_spectrum_groups(client_data, dataset):
    """
    Partition client indices into SHORT and LONG spectrum groups.

    CASIA-MS: SHORT = visible (460, 630, 700, WHT); LONG = NIR (850, 940)
    XJTU:     SHORT = Flash lighting;               LONG = Nature lighting

    Returns
    -------
    short_ids : list[int]  client indices in the SHORT group
    long_ids  : list[int]  client indices in the LONG group
    """
    if dataset == "casiams":
        short_set = set(CASIAMS_SHORT_SPECTRA)
        long_set  = set(CASIAMS_LONG_SPECTRA)
    else:
        short_set = set(XJTU_SHORT_SPECTRA)
        long_set  = set(XJTU_LONG_SPECTRA)

    short_ids, long_ids = [], []
    for i, cd in enumerate(client_data):
        sp = cd["spectrum"]
        if sp in short_set:
            short_ids.append(i)
        elif sp in long_set:
            long_ids.append(i)
        else:
            # fallback: assign to nearest group by position
            short_ids.append(i)

    print(f"  Spectrum groups:")
    print(f"    SHORT ({len(short_ids)} clients): "
          f"{[client_data[i]['spectrum'] for i in short_ids]}")
    print(f"    LONG  ({len(long_ids)} clients): "
          f"{[client_data[i]['spectrum'] for i in long_ids]}")
    return short_ids, long_ids


# ══════════════════════════════════════════════════════════════
#  FEDAVG HELPERS
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def fedavg(target_model, source_models):
    """
    FedAvg: target_model ← uniform average of source_models.
    Modifies target_model in-place. source_models unchanged.
    """
    n = len(source_models)
    if n == 0:
        return target_model
    for key in target_model.state_dict().keys():
        agg = torch.zeros_like(target_model.state_dict()[key],
                               dtype=torch.float32)
        for m in source_models:
            agg += (1.0 / n) * m.state_dict()[key].float()
        target_model.state_dict()[key].data.copy_(agg)
    return target_model


@torch.no_grad()
def broadcast(source_model, target_models):
    """Copy source_model weights to all target_models."""
    for m in target_models:
        for key in source_model.state_dict().keys():
            m.state_dict()[key].data.copy_(
                source_model.state_dict()[key])


# ══════════════════════════════════════════════════════════════
#  ONE LOCAL TRAINING ROUND — PSFed-Palm algorithm (verbatim)
# ══════════════════════════════════════════════════════════════

def fit(local_model, anchor_model, server_model,
        loader, optimizer, criterion, con_criterion, device,
        w1, w2, w3, mu):
    """
    One communication round of local PSFed-Palm training.

    Verbatim from original main.py::fit() — only data format adapted
    (uses our paired DataLoader instead of text-file MyDataset).

    Loss:
      L = w1 × CE(output, y)
        + w2 × SupCon([fe1, fe2], y)
        + w3 × MSE(fe1, anchor_fe.detach())   ← cross-spectrum alignment
        + mu/2 × ||θ − server||               ← FedProx to global
        + mu/2 × ||θ − anchor||               ← FedProx to anchor

    Parameters
    ----------
    local_model   : θ_i  (trained in-place)
    anchor_model  : cross-spectrum anchor (visib_net or invis_net, frozen)
    server_model  : Φ (global model, frozen — used for FedProx only)
    loader        : DataLoader returning ([x, x_con], y)
    """
    local_model.train()
    anchor_model.eval()
    server_model.eval()

    cri_mse = nn.MSELoss().to(device)

    running_loss    = 0.0
    running_correct = 0
    total           = 0

    for datas, target in loader:
        x     = datas[0].to(device)
        x_con = datas[1].to(device)
        y     = target.to(device)

        optimizer.zero_grad()

        # local model forward — two views
        output, fe1, _ = local_model(x,     y)
        _,      fe2, _ = local_model(x_con, y)

        # cross-spectrum anchor forward (no gradient)
        with torch.no_grad():
            _, fe3, _ = anchor_model(x, None, None)

        # paired features for SupCon
        fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

        # losses — verbatim from PSFed-Palm
        ce      = criterion(output, y)
        supcon  = con_criterion(fe, y)
        mse     = cri_mse(fe1, fe3.detach())

        # FedProx to global server model (standard: mu/2 * ||θ - Φ||²)
        w_diff_server = torch.tensor(0., device=device)
        for w, w_t in zip(server_model.parameters(),
                          local_model.parameters()):
            w_diff_server += torch.pow(torch.norm(w - w_t), 2)
        prox_server = mu / 2. * w_diff_server

        # FedProx to cross-spectrum anchor model
        w_diff_anchor = torch.tensor(0., device=device)
        for w, w_t in zip(anchor_model.parameters(),
                          local_model.parameters()):
            w_diff_anchor += torch.pow(torch.norm(w - w_t), 2)
        prox_anchor = mu / 2. * w_diff_anchor

        loss = w1 * ce + w2 * supcon + w3 * mse + prox_server + prox_anchor

        loss.backward()
        optimizer.step()

        running_loss    += loss.item() * x.size(0)
        running_correct += output.argmax(1).eq(y).sum().item()
        total           += x.size(0)

    return running_loss / max(total, 1), 100. * running_correct / max(total, 1)


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract(embedding_fn, loader, device):
    feats, labels = [], []
    for imgs, labs in loader:
        feats.append(embedding_fn(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def evaluate_split(embedding_fn, gallery_loader, probe_loader, device):
    gal_f, gal_l = extract(embedding_fn, gallery_loader, device)
    prb_f, prb_l = extract(embedding_fn, probe_loader,   device)
    sim = np.nan_to_num(prb_f @ gal_f.T, nan=0.0)

    scores, lbls = [], []
    for i in range(len(prb_f)):
        for j in range(len(gal_f)):
            scores.append(float(sim[i, j]))
            lbls.append(1 if prb_l[i] == gal_l[j] else -1)

    eer  = compute_eer(np.column_stack([scores, lbls]))
    nn_i = np.argmax(sim, axis=1)
    r1   = 100. * sum(prb_l[i] == gal_l[nn_i[i]]
                      for i in range(len(prb_f))) / max(len(prb_f), 1)
    return eer, r1


@torch.no_grad()
def emb_global(server_model, x):
    """Global model embedding — compnet_fedpalm returns (logits, fe, _)."""
    server_model.eval()
    _, fe, _ = server_model(x, None, None)
    return fe                          # already L2-normed in compnet_fedpalm


def evaluate_local_avg(local_models, gallery_loader, probe_loader, device):
    """Per-client independent evaluation, then average EER and Rank-1."""
    eers, r1s = [], []
    for m in local_models:
        m.eval()
        eer, r1 = evaluate_split(
            lambda x, _m=m: _m(x, None, None)[1],
            gallery_loader, probe_loader, device)
        eers.append(eer)
        r1s.append(r1)
    return sum(eers) / len(eers), sum(r1s) / len(r1s)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():
    set_seed(cfg["random_seed"])
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = cfg["dataset"]

    out_dir     = cfg["base_results_dir"].format(dataset=dataset)
    splits_path = cfg["splits_path"].format(dataset=dataset)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.txt")

    # ── 1. Load / build splits ────────────────────────────────
    if os.path.exists(splits_path):
        print(f"\nLoading existing splits from {splits_path}")
        with open(splits_path, "rb") as f:
            client_data, gallery_samples, probe_samples, _, spectra = \
                pickle.load(f)
    else:
        print(f"\nBuilding {dataset.upper()} splits "
              f"({cfg.get('eval_protocol', 'open_set')}) …")
        result = get_federated_splits(cfg, seed=cfg["random_seed"])
        client_data, gallery_samples, probe_samples, _, spectra = result
        os.makedirs(os.path.dirname(splits_path), exist_ok=True)
        with open(splits_path, "wb") as f:
            pickle.dump(result, f)

    n_clients = len(client_data)

    # ── 2. Global label remapping ─────────────────────────────
    # Identical to FedPalm: all clients share the same ArcFace output
    # dimension so FedAvg is coherent across label spaces.
    all_identities = set()
    for cd in client_data:
        all_identities.update(cd["label_map"].keys())
    global_label_map = {ident: i
                        for i, ident in enumerate(sorted(all_identities))}
    num_classes = len(global_label_map)

    for cd in client_data:
        local_to_ident = {v: k for k, v in cd["label_map"].items()}
        cd["train_samples"] = [
            (path, global_label_map[local_to_ident[local_lab]])
            for path, local_lab in cd["train_samples"]
        ]

    print(f"\n  Clients        : {n_clients}")
    print(f"  Total train IDs: {num_classes}  "
          f"(~{num_classes // n_clients} per client)")
    print(f"  Gallery        : {len(gallery_samples)}  "
          f"Probe: {len(probe_samples)}")
    print(f"  Aug multiplier : M={cfg['M']} (spatial aug only, no FFT)")
    print(f"  Device         : {device}\n")

    # ── 3. Spectrum grouping ──────────────────────────────────
    short_ids, long_ids = build_spectrum_groups(client_data, dataset)

    # ── 4. DataLoaders ───────────────────────────────────────
    M  = cfg["M"]
    bs = cfg["batch_size"]
    nw = cfg["num_workers"]

    train_loaders = [
        DataLoader(
            PSFedDataset(cd["train_samples"], cfg["img_side"], M=M),
            batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
        for cd in client_data
    ]
    gallery_loader = DataLoader(
        PalmDataset(gallery_samples, cfg["img_side"]),
        batch_size=bs, shuffle=False, num_workers=nw)
    probe_loader   = DataLoader(
        PalmDataset(probe_samples,   cfg["img_side"]),
        batch_size=bs, shuffle=False, num_workers=nw)

    # ── 5. Models ─────────────────────────────────────────────
    # server_model : Φ — FedAvg of all local models (global)
    # visib_net    : sub-anchor for SHORT-spectrum group
    # invis_net    : sub-anchor for LONG-spectrum group
    # local_models : θ_i — one per client, never shared
    server_model = compnet_fedpalm(num_classes=num_classes).to(device)
    visib_net    = compnet_fedpalm(num_classes=num_classes).to(device)
    invis_net    = compnet_fedpalm(num_classes=num_classes).to(device)
    local_models = [copy.deepcopy(server_model) for _ in range(n_clients)]

    # cross-spectrum anchor assignment — each client aligns with the
    # OTHER group's anchor to enforce cross-spectrum consistency
    def get_anchor(i):
        """Return the opposing anchor model for client i."""
        return invis_net if i in short_ids else visib_net

    optimizers = [
        torch.optim.Adam(local_models[i].parameters(), lr=cfg["lr"])
        for i in range(n_clients)
    ]
    schedulers = [
        lr_scheduler.StepLR(optimizers[i],
                             step_size=cfg["lr_step"],
                             gamma=cfg["lr_gamma"])
        for i in range(n_clients)
    ]

    criterion     = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=cfg["temperature"],
                               base_temperature=cfg["temperature"])

    # ── 6. Results header ─────────────────────────────────────
    with open(results_path, "w") as f:
        f.write(f"PSFed-Palm — {dataset.upper()}\n")
        f.write(f"{'Round':>6}\t"
                f"{'Short EER':>10}\t{'Short R1':>9}\t"
                f"{'Long EER':>9}\t{'Long R1':>8}\t"
                f"{'Global EER':>11}\t{'Global R1':>10}\n")

    # ── Round 0: random-init baseline ─────────────────────────
    print("\n--- Round 0 (random init) ---")
    server_model.eval()
    g0_eer, g0_r1 = evaluate_split(
        lambda x: emb_global(server_model, x),
        gallery_loader, probe_loader, device)
    print(f"  Global init  EER={g0_eer*100:.4f}%  Rank-1={g0_r1:.2f}%")

    recent = []

    # ── 7. FL rounds ──────────────────────────────────────────
    for rnd in range(1, cfg["n_rounds"] + 1):
        t0 = time.time()

        # Step 1: local training
        round_losses = []
        for i in range(n_clients):
            anchor = get_anchor(i)
            loss, _ = fit(
                local_model   = local_models[i],
                anchor_model  = anchor,
                server_model  = server_model,
                loader        = train_loaders[i],
                optimizer     = optimizers[i],
                criterion     = criterion,
                con_criterion = con_criterion,
                device        = device,
                w1=cfg["w1"], w2=cfg["w2"], w3=cfg["w3"], mu=cfg["mu"],
            )
            schedulers[i].step()
            round_losses.append(loss)

        # Step 2a: sub-group aggregation
        if short_ids:
            fedavg(visib_net, [local_models[i] for i in short_ids])
        if long_ids:
            fedavg(invis_net, [local_models[i] for i in long_ids])

        # Step 2b: global aggregation
        fedavg(server_model, local_models)

        # Step 2c: broadcast global model to all local models
        broadcast(server_model, local_models)

        elapsed = time.time() - t0

        # Step 3: evaluation — 3 pairs
        # (1) Short group model (visib_net)
        sh_eer, sh_r1 = evaluate_split(
            lambda x: emb_global(visib_net, x),
            gallery_loader, probe_loader, device)

        # (2) Long group model (invis_net)
        lo_eer, lo_r1 = evaluate_split(
            lambda x: emb_global(invis_net, x),
            gallery_loader, probe_loader, device)

        # (3) Global model (FedAvg of all clients)
        g_eer, g_r1 = evaluate_split(
            lambda x: emb_global(server_model, x),
            gallery_loader, probe_loader, device)

        recent.append((g_eer, g_r1))
        if len(recent) > cfg["avg_last_rounds"]:
            recent.pop(0)

        avg_loss = sum(round_losses) / len(round_losses)
        ts       = time.strftime("%H:%M:%S")

        print(f"[{ts}] Round {rnd:04d}/{cfg['n_rounds']}  "
              f"loss={avg_loss:.4f}  ({elapsed:.1f}s)")
        print(f"  Short group  EER={sh_eer*100:.4f}%  Rank-1={sh_r1:.2f}%")
        print(f"  Long  group  EER={lo_eer*100:.4f}%  Rank-1={lo_r1:.2f}%")
        print(f"  Global       EER={g_eer*100:.4f}%  Rank-1={g_r1:.2f}%")

        with open(results_path, "a") as f:
            f.write(f"{rnd:>6}\t"
                    f"{sh_eer*100:.4f}\t{sh_r1:.2f}\t"
                    f"{lo_eer*100:.4f}\t{lo_r1:.2f}\t"
                    f"{g_eer*100:.4f}\t{g_r1:.2f}\n")

    # ── 8. Final summary ──────────────────────────────────────
    n_avg   = len(recent)
    avg_eer = sum(e for e, _ in recent) / n_avg
    avg_r1  = sum(r for _, r in recent) / n_avg

    print(f"\n{'='*62}")
    print(f"  FL COMPLETE — {cfg['n_rounds']} rounds")
    print(f"  Dataset            : {dataset.upper()}")
    print(f"  Model              : PSFed-Palm (CompNet backbone)")
    print(f"  Avg Global EER     : {avg_eer*100:.4f}%  (last {n_avg} rounds)")
    print(f"  Avg Global Rank-1  : {avg_r1:.2f}%   (last {n_avg} rounds)")
    print(f"  Results saved to   : {results_path}")
    print(f"{'='*62}")

    with open(results_path, "a") as f:
        f.write(f"\n# Average of last {n_avg} rounds (global model)\n")
        f.write(f"avg_{n_avg}\t—\t—\t—\t—\t"
                f"{avg_eer*100:.4f}\t{avg_r1:.2f}\n")


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--dataset", choices=["casiams", "xjtu"])
    _p.add_argument("--eval_protocol", choices=["open_set", "closed_set"])
    _p.add_argument("--closed_set_mode", choices=["holdout", "cross_spectrum"])
    _p.add_argument("--n_rounds", type=int)
    _p.add_argument("--random_seed", type=int)
    _p.add_argument("--splits_path")
    _p.add_argument("--n_ids", type=int)
    _p.add_argument("--eval_every", type=int)
    _args, _ = _p.parse_known_args()
    for _k, _v in vars(_args).items():
        if _v is not None:
            cfg[_k] = _v
    main()
