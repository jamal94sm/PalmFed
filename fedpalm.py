# ==============================================================
#  train_fedpalm.py
#  FedPalm adapted for CASIA-MS / XJTU
#  open-set, cross-domain, non-shared-ID evaluation
#
#  Core algorithm kept intact from:
#    Yang et al., "FedPalm", TIFS 2026
#
#  Adaptations (dataset loading + evaluation only):
#    - All hyperparameters loaded from configs_fedpalm.py
#    - Dataset splits via build_federated_splits / build_federated_splits_xjtu
#    - Evaluation: EER + Rank-1 via compute_eer (our metrics)
#    - Reports global / local-avg / full FedPalm on gallery+probe
# ==============================================================

# ──────────────────────────────────────────────────────────────
#  ALGORITHM STEPS (aligned with paper §III and original code)
#
#  Step 0 — Initialisation
#    All local models θ_i and anchor models φ_i initialised identically.
#
#  Step 1 — Local training (per client i, per round r):
#    For each mini-batch with two augmented views (x, x'):
#      a) fe_orig    = θ_i(x)               personalized local feature
#      b) fe_j       = θ_j(x) ∀j≠i (frozen) other clients' local features
#         sim_fe     = TEIM(fe_orig, {fe_j})
#                    = 0.8×fe_orig + 0.1×top1_fe + 0.1×top2_fe
#                    [Textural Expert Interaction Module]
#      c) ancho_fe   = φ_i(x)               anchor expert feature
#      d) final_fe   = w_anch×ancho_fe + w_side×sim_fe  [blended embedding]
#      e) Repeat (a-d) for x' → final_fe'
#      f) Loss:
#           L = w1×CE(θ_i(x), y)
#             + w2×SupCon([final_fe, final_fe'], y)
#             + w1×CE(φ_i(x), y)
#      g) Update θ_i and φ_i jointly.
#
#  Step 2 — Aggregation:
#    Φ ← FedAvg({φ_i})   [anchor models only — local θ_i untouched]
#    φ_i ← Φ  ∀i
#
#  Step 3 — Evaluation (three modes):
#    (a) Global model:     embed via Φ(x)
#    (b) Local-avg model:  embed via mean_k(θ_k(x))
#    (c) Full FedPalm:     embed via w_anch×Φ(x) + w_side×TEIM({θ_k(x)})
#    All modes: cosine matching on gallery/probe → EER + Rank-1
# ──────────────────────────────────────────────────────────────

import os
import copy
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from torchvision import transforms as T
from PIL import Image

# ── Config ────────────────────────────────────────────────────
from configs_fedpalm import CONFIG_FEDPALM as cfg

# ── Our framework ─────────────────────────────────────────────
from datasets import (build_federated_splits, build_federated_splits_xjtu,
                      PalmDataset, NormSingleROI)
from utils import compute_eer

# ── FedPalm model — import directly to avoid conflict with local models.py ──
# Copy compnet_original.py from the FedPalm repo into your project root,
# then import it as a top-level module.
from compnet_original import compnet_fedpalm

# ── FedPalm loss (verbatim from original paper repo) ──────────
from loss import SupConLoss


# ══════════════════════════════════════════════════════════════
#  DATASET — paired same-class views for SupCon
# ══════════════════════════════════════════════════════════════

class FedPalmDataset(Dataset):
    """
    Paired dataset matching MyDataset_general_FL from the original paper.

    Returns ([img_anchor, img_pair], label):
      img_anchor — spatially augmented view of sample i
      img_pair   — spatially augmented view of a (possibly same) sample
                   from the SAME identity (positive pair for SupCon)

    Augmentation identical to the original paper's MyDataset_general_FL.
    Input: list of (path, label_int) tuples from build_federated_splits.
    """
    def __init__(self, samples, img_side=128):
        self.samples  = samples
        self.img_side = img_side

        from collections import defaultdict
        self.label2idxs = defaultdict(list)
        for i, (_, lab) in enumerate(samples):
            self.label2idxs[lab].append(i)

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
        return len(self.samples)

    def _load(self, path):
        return self.transforms(Image.open(path).convert("L"))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        idx2   = random.choice(self.label2idxs[label])
        path2, _ = self.samples[idx2]
        return [self._load(path), self._load(path2)], label


# ══════════════════════════════════════════════════════════════
#  TEIM — Textural Expert Interaction Module (verbatim)
# ══════════════════════════════════════════════════════════════

def teim(fe_orig, fes):
    """
    Textural Expert Interaction Module from the FedPalm paper.
    Routes fe_orig through other clients' features using cosine
    similarity. Top-2 most similar experts are selected and blended:
      sim_fe = w_self×fe_orig + w_top1×top1 + w_top2×top2

    fe_orig : [B, D]
    fes     : [n_other, B, D]
    returns : [B, D]
    """
    w_self = cfg["teim_self_weight"]
    w_top1 = cfg["teim_top1_weight"]
    w_top2 = cfg["teim_top2_weight"]

    n_other = fes.shape[0]
    fe_u    = fe_orig.unsqueeze(0)                              # [1, B, D]
    sims    = torch.einsum("nbi,mbi->nb", fes, fe_u)           # [n_other, B]
    sim_t   = sims.T                                            # [B, n_other]

    top_k   = min(2, n_other)
    top_idx = torch.topk(sim_t, top_k, dim=-1).indices         # [B, top_k]
    b_idx   = torch.arange(fes.shape[1]).unsqueeze(1)          # [B, 1]

    fe1 = fes[top_idx[:, 0], b_idx[:, 0], :]                   # [B, D]
    fe2 = fes[top_idx[:, 1], b_idx[:, 0], :] if top_k >= 2 else fe1

    return w_self * fe_orig + w_top1 * fe1 + w_top2 * fe2


# ══════════════════════════════════════════════════════════════
#  FEDAVG — anchor models only (verbatim logic)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def fedavg_anchors(server_model, anchor_models):
    """
    FedAvg over anchor models → server model, then broadcast back.
    Local models (θ_i) are NOT touched — personalized knowledge preserved.
    """
    n = len(anchor_models)
    for key in server_model.state_dict().keys():
        agg = torch.zeros_like(server_model.state_dict()[key],
                               dtype=torch.float32)
        for m in anchor_models:
            agg += (1.0 / n) * m.state_dict()[key].float()
        server_model.state_dict()[key].data.copy_(agg)
        for m in anchor_models:
            m.state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, anchor_models


# ══════════════════════════════════════════════════════════════
#  ONE LOCAL TRAINING ROUND (adapted from fit_fedpalm)
# ══════════════════════════════════════════════════════════════

def train_one_round(local_model, anchor_model, other_models,
                    loader, local_opt, anchor_opt,
                    criterion, con_criterion, device):
    """
    One communication round of local training — original FedPalm algorithm.

    For each mini-batch (x, x') — two augmented views of same-class pairs:
      1. local model forward on x  → fe_orig (local expert feature)
      2. other clients' locals on x (frozen, no_grad) → fes
         TEIM(fe_orig, fes) → sim_fe  (cross-expert routed feature)
      3. anchor model forward on x → ancho_fe
      4. final_fe = w_anch×ancho_fe + w_side×sim_fe
      5. repeat for x' → final_fe'
      6. L = w1×CE(local,y) + w2×SupCon([final_fe,final_fe'],y)
           + w1×CE(anchor,y)
      7. backprop through local_model AND anchor_model jointly.
         Other models are frozen (no_grad) — no gradient flows to them.
    """
    local_model.train()
    anchor_model.train()
    for m in other_models:
        m.eval()

    w1     = cfg["w1"]
    w2     = cfg["w2"]
    w_anch = cfg["teim_blend_anchor"]
    w_side = cfg["teim_blend_side"]

    running_loss    = 0.0
    running_correct = 0
    total           = 0

    for datas, target in loader:
        x     = datas[0].to(device)
        x_con = datas[1].to(device)
        y     = target.to(device)

        # ── view 1 ──────────────────────────────────────────
        out_local, fe_orig, _ = local_model(x, y, None)

        if other_models:
            with torch.no_grad():
                fes = torch.stack([m(x, None, None)[1]
                                   for m in other_models])   # [n-1, B, D]
            sim_fe = teim(fe_orig, fes)
        else:
            sim_fe = fe_orig

        out_anch, ancho_fe, _ = anchor_model(x, y, None)
        final_fe = w_anch * ancho_fe + w_side * sim_fe

        # ── view 2 ──────────────────────────────────────────
        out_local2, fe_orig2, _ = local_model(x_con, y, None)

        if other_models:
            with torch.no_grad():
                fes2 = torch.stack([m(x_con, None, None)[1]
                                    for m in other_models])
            sim_fe2 = teim(fe_orig2, fes2)
        else:
            sim_fe2 = fe_orig2

        out_anch2, ancho_fe2, _ = anchor_model(x_con, y, None)
        final_fe2 = w_anch * ancho_fe2 + w_side * sim_fe2

        # ── losses ──────────────────────────────────────────
        fe_pair     = torch.stack([final_fe, final_fe2], dim=1)  # [B, 2, D]
        loss_local  = criterion(out_local, y)
        loss_anchor = criterion(out_anch,  y)
        loss_contra = con_criterion(fe_pair, y)

        loss = w1 * loss_local + w2 * loss_contra + w1 * loss_anchor

        local_opt.zero_grad()
        anchor_opt.zero_grad()
        loss.backward()
        local_opt.step()
        anchor_opt.step()

        running_loss    += loss.item() * x.size(0)
        running_correct += out_local.argmax(1).eq(y).sum().item()
        total           += x.size(0)

    return running_loss / max(total, 1), 100.0 * running_correct / max(total, 1)


# ══════════════════════════════════════════════════════════════
#  EVALUATION — three modes, our EER + Rank-1 metrics
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract(embedding_fn, loader, device):
    """Extract embeddings using any callable embedding_fn(x) → [B, D]."""
    feats, labels = [], []
    for imgs, labs in loader:
        feats.append(embedding_fn(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def evaluate_split(embedding_fn, gallery_loader, probe_loader, device):
    """EER + Rank-1 from our compute_eer on gallery/probe."""
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
    r1   = 100.0 * sum(prb_l[i] == gal_l[nn_i[i]]
                       for i in range(len(prb_f))) / max(len(prb_f), 1)
    return eer, r1


# ── three embedding functions (one per evaluation mode) ───────

@torch.no_grad()
def score_statistics(embedding_fn, gallery_loader, probe_loader, device):
    """
    Compute genuine vs impostor cosine similarity statistics.
    If genuine mean drops over rounds → backbone is diverging.
    """
    gal_f, gal_l = extract(embedding_fn, gallery_loader, device)
    prb_f, prb_l = extract(embedding_fn, probe_loader,   device)
    sim = np.nan_to_num(prb_f @ gal_f.T, nan=0.0)

    genuine, impostor = [], []
    for i in range(len(prb_f)):
        for j in range(len(gal_f)):
            (genuine if prb_l[i] == gal_l[j] else impostor).append(sim[i, j])

    genuine  = np.array(genuine)
    impostor = np.array(impostor)
    return (genuine.mean(),  genuine.std(),
            impostor.mean(), impostor.std())


@torch.no_grad()
def backbone_weight_norm(model):
    """L2 norm of fc_brand (the identity-projection layer) weights."""
    return model.fc_brand.weight.data.norm().item()


@torch.no_grad()
def arcface_weight_drift(models):
    """
    Max pairwise cosine distance between clients' ArcFace weight matrices.
    High drift = clients' identity spaces have diverged significantly.
    Zero would mean all ArcFace heads are identical (no personalisation).
    """
    ws = [F.normalize(m.arclayer_brand.weight.data.view(-1), p=2, dim=0)
          for m in models]
    dists = []
    for i in range(len(ws)):
        for j in range(i+1, len(ws)):
            dists.append((1 - (ws[i] * ws[j]).sum()).item())
    return max(dists) if dists else 0.0
    """(a) Global model: aggregated anchor Φ only."""
    server_model.eval()
    _, fe, _ = server_model(x, None, None)
    return fe                                     # already L2-normed


def emb_global(server_model, x):
    """(a) Global model: aggregated anchor Φ only."""
    server_model.eval()
    _, fe, _ = server_model(x, None, None)
    return fe   # already L2-normed in compnet_fedpalm


def emb_local_avg(local_models, x):
    """(b) Local-average: mean embedding across all personalized experts."""
    for m in local_models:
        m.eval()
    fes = torch.stack([m(x, None, None)[1] for m in local_models])
    return F.normalize(fes.mean(dim=0), p=2, dim=1)


def emb_full(server_model, local_models, x):
    """(c) Full FedPalm: w_anch×Φ(x) + w_side×TEIM({θ_k(x)})."""
    server_model.eval()
    for m in local_models:
        m.eval()
    w_anch = cfg["teim_blend_anchor"]
    w_side = cfg["teim_blend_side"]
    _, fe_anch, _ = server_model(x, None, None)
    if local_models:
        fes    = torch.stack([m(x, None, None)[1] for m in local_models])
        sim_fe = teim(fe_anch, fes)
        fe_out = w_anch * fe_anch + w_side * sim_fe
    else:
        fe_out = fe_anch
    return F.normalize(fe_out, p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():
    set_seed(cfg["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── resolve paths ────────────────────────────────────────
    dataset    = cfg["dataset"]
    out_dir    = cfg["base_results_dir"].format(dataset=dataset)
    splits_path = cfg["splits_path"].format(dataset=dataset)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.txt")

    data_root  = (cfg["data_root"] if dataset == "casiams"
                  else cfg["xjtu_data_root"])

    # ── 1. Build / load splits ───────────────────────────────
    import pickle
    if os.path.exists(splits_path):
        print(f"\nLoading existing splits from {splits_path}")
        with open(splits_path, "rb") as f:
            client_data, gallery_samples, probe_samples, _, spectra = \
                pickle.load(f)
    else:
        print(f"\nBuilding {dataset.upper()} splits …")
        if dataset == "casiams":
            result = build_federated_splits(
                data_root, cfg["n_ids"], cfg["k_test"],
                cfg["gallery_ratio"], seed=cfg["random_seed"])
        else:
            result = build_federated_splits_xjtu(
                data_root, cfg["n_ids"], cfg["k_test"],
                cfg["gallery_ratio"], seed=cfg["random_seed"])
        client_data, gallery_samples, probe_samples, _, spectra = result
        os.makedirs(os.path.dirname(splits_path), exist_ok=True)
        with open(splits_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Splits saved to {splits_path}")

    n_clients = len(client_data)
    img_side  = cfg["img_side"]
    bs        = cfg["batch_size"]
    nw        = cfg["num_workers"]

    # ── 2. Global label remapping ─────────────────────────────
    # Original FedPalm: num_classes = total training IDs across all clients
    # (e.g. 378 for PolyU with 8 clients). All models share the same ArcFace
    # output dimension so FedAvg is well-defined. Each client only trains on
    # its own ~26 rows; the other rows receive zero gradient from that client.
    #
    # Our splits give each client LOCAL labels 0..n_client-1, meaning
    # client 0's "label 5" and client 1's "label 5" are DIFFERENT people.
    # FedAvg would incorrectly mix their ArcFace prototypes.
    #
    # Fix: create a GLOBAL label map {identity_string: global_int} so that
    # each unique person gets a unique integer across all clients, then
    # remap each client's samples accordingly.
    #
    # label_map in client_data: {identity_string: local_int}
    # We build a reverse map (local_int → identity_string) per client,
    # then assign global integers.

    all_identities = set()
    for cd in client_data:
        all_identities.update(cd["label_map"].keys())
    global_label_map = {ident: i for i, ident in enumerate(sorted(all_identities))}
    num_classes = len(global_label_map)

    # remap each client's train_samples to global labels
    for cd in client_data:
        local_to_ident = {v: k for k, v in cd["label_map"].items()}
        cd["train_samples"] = [
            (path, global_label_map[local_to_ident[local_lab]])
            for path, local_lab in cd["train_samples"]
        ]

    print(f"\n  Clients        : {n_clients}")
    print(f"  Total train IDs: {num_classes}  "
          f"(per-client count: ~{num_classes // n_clients})")
    print(f"  Gallery        : {len(gallery_samples)}  Probe: {len(probe_samples)}")
    print(f"  Device         : {device}\n")

    # ── 3. DataLoaders ───────────────────────────────────────
    train_loaders = [
        DataLoader(FedPalmDataset(cd["train_samples"], img_side),
                   batch_size=bs, shuffle=True, num_workers=nw,
                   pin_memory=True)
        for cd in client_data
    ]
    gallery_loader = DataLoader(PalmDataset(gallery_samples, img_side),
                                batch_size=bs, shuffle=False, num_workers=nw)
    probe_loader   = DataLoader(PalmDataset(probe_samples,   img_side),
                                batch_size=bs, shuffle=False, num_workers=nw)

    server_model  = compnet_fedpalm(num_classes=num_classes).to(device)
    local_models  = [copy.deepcopy(server_model) for _ in range(n_clients)]
    anchor_models = [copy.deepcopy(server_model) for _ in range(n_clients)]

    lr       = cfg["lr"]
    step     = cfg["lr_step"]
    gamma    = cfg["lr_gamma"]
    local_opts   = [torch.optim.Adam(m.parameters(), lr=lr)
                    for m in local_models]
    anchor_opts  = [torch.optim.Adam(m.parameters(), lr=lr)
                    for m in anchor_models]
    local_scheds  = [lr_scheduler.StepLR(o, step_size=step, gamma=gamma)
                     for o in local_opts]
    anchor_scheds = [lr_scheduler.StepLR(o, step_size=step, gamma=gamma)
                     for o in anchor_opts]

    criterion     = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=cfg["temperature"],
                               base_temperature=cfg["temperature"])

    # ── 4. Results header ────────────────────────────────────
    with open(results_path, "w") as f:
        f.write(f"FedPalm — {dataset.upper()}\n")
        f.write(f"{'Round':>6}\t"
                f"{'Global EER':>12}\t{'Global R1':>10}\t"
                f"{'LocalAvg EER':>14}\t{'LocalAvg R1':>12}\t"
                f"{'Full EER':>10}\t{'Full R1':>8}\n")

    # ── Round 0: true random-init baseline ──────────────────
    print("\n--- Round 0 (random init) ---")
    server_model.eval()
    g0_eer, g0_r1 = evaluate_split(
        lambda x: emb_global(server_model, x),
        gallery_loader, probe_loader, device)
    gm, gs, im, is_ = score_statistics(
        lambda x: emb_global(server_model, x),
        gallery_loader, probe_loader, device)
    print(f"  Global init  EER={g0_eer*100:.4f}%  Rank-1={g0_r1:.2f}%")
    print(f"  Score stats  genuine={gm:.4f}\u00b1{gs:.4f}  "
          f"impostor={im:.4f}\u00b1{is_:.4f}  sep={gm-im:.4f}")
    print(f"  fc_brand norm: {backbone_weight_norm(server_model):.4f}")
    server_model.train()

    recent_global = []

    # ── 5. FL rounds ─────────────────────────────────────────
    for rnd in range(1, cfg["n_rounds"] + 1):
        t0 = time.time()

        # Step 1: local training per client — fix train/eval modes
        round_losses, round_accs = [], []
        for i in range(n_clients):
            # set correct modes explicitly before each client
            local_models[i].train()
            anchor_models[i].train()
            for j in range(n_clients):
                if j != i:
                    local_models[j].eval()
                    anchor_models[j].eval()

            others = [local_models[j] for j in range(n_clients) if j != i]
            loss, acc = train_one_round(
                local_models[i], anchor_models[i], others,
                train_loaders[i], local_opts[i], anchor_opts[i],
                criterion, con_criterion, device)
            local_scheds[i].step()
            anchor_scheds[i].step()
            round_losses.append(loss)
            round_accs.append(acc)

        # Step 2: FedAvg of anchor models only
        server_model, anchor_models = fedavg_anchors(server_model,
                                                      anchor_models)
        elapsed = time.time() - t0

        # Step 3: Evaluation — three modes
        g_eer = g_r1 = la_eer = la_r1 = fp_eer = fp_r1 = 0.0

        if cfg["eval_global"]:
            g_eer, g_r1 = evaluate_split(
                lambda x: emb_global(server_model, x),
                gallery_loader, probe_loader, device)

        if cfg["eval_local_avg"]:
            la_eer, la_r1 = evaluate_split(
                lambda x: emb_local_avg(local_models, x),
                gallery_loader, probe_loader, device)

        if cfg["eval_full"]:
            fp_eer, fp_r1 = evaluate_split(
                lambda x: emb_full(server_model, local_models, x),
                gallery_loader, probe_loader, device)

        # restore train mode for next round
        server_model.train()
        for m in local_models + anchor_models:
            m.train()

        recent_global.append((g_eer, g_r1))
        if len(recent_global) > cfg["avg_last_rounds"]:
            recent_global.pop(0)

        # ── Diagnostics ───────────────────────────────────────
        avg_loss = sum(round_losses) / len(round_losses)
        avg_acc  = sum(round_accs)  / len(round_accs)
        ts = time.strftime("%H:%M:%S")

        # per-client training accuracy — rising fast = memorising local IDs
        acc_str = "  ".join(f"c{i}:{round_accs[i]:.0f}%"
                             for i in range(n_clients))

        # genuine/impostor separation — drop = backbone diverging
        gm, gs, im, is_ = score_statistics(
            lambda x: emb_global(server_model, x),
            gallery_loader, probe_loader, device)

        # ArcFace divergence across clients — diagnoses ID-space conflict

        # backbone projection norms
        fc_norms = "  ".join(f"c{i}:{backbone_weight_norm(local_models[i]):.3f}"
                              for i in range(n_clients))

        print(f"[{ts}] Round {rnd:04d}/{cfg['n_rounds']}  "
              f"loss={avg_loss:.4f}  acc={avg_acc:.1f}%  ({elapsed:.1f}s)")
        print(f"  Per-client acc     : {acc_str}")
        print(f"  Score sep (gen-imp): {gm-im:.4f}  "
              f"gen={gm:.4f}\u00b1{gs:.4f}  imp={im:.4f}\u00b1{is_:.4f}")
        print(f"  fc_brand norms     : {fc_norms}")
        print(f"  Global       EER={g_eer*100:.4f}%  Rank-1={g_r1:.2f}%")
        print(f"  Local avg    EER={la_eer*100:.4f}%  Rank-1={la_r1:.2f}%")
        print(f"  Full FedPalm EER={fp_eer*100:.4f}%  Rank-1={fp_r1:.2f}%")

        with open(results_path, "a") as f:
            f.write(f"{rnd:>6}\t"
                    f"{g_eer*100:.4f}\t{g_r1:.2f}\t"
                    f"{la_eer*100:.4f}\t{la_r1:.2f}\t"
                    f"{fp_eer*100:.4f}\t{fp_r1:.2f}\n")

    # ── 6. Final summary ─────────────────────────────────────
    n_avg   = len(recent_global)
    avg_eer = sum(e for e, _ in recent_global) / n_avg
    avg_r1  = sum(r for _, r in recent_global) / n_avg

    print(f"\n{'='*62}")
    print(f"  FL COMPLETE — {cfg['n_rounds']} rounds")
    print(f"  Dataset            : {dataset.upper()}")
    print(f"  Model              : FedPalm (CompNet backbone)")
    print(f"  Avg Global EER     : {avg_eer*100:.4f}%  (last {n_avg} rounds)")
    print(f"  Avg Global Rank-1  : {avg_r1:.2f}%   (last {n_avg} rounds)")
    print(f"  Results saved to   : {results_path}")
    print(f"{'='*62}")

    with open(results_path, "a") as f:
        f.write(f"\n# Average of last {n_avg} rounds (global model)\n")
        f.write(f"avg_{n_avg}\t{avg_eer*100:.4f}\t{avg_r1:.2f}"
                f"\t—\t—\t—\t—\n")


if __name__ == "__main__":
    main()
