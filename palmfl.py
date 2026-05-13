"""
Federated Learning for Palmprint Recognition on CASIA-MS
===========================================================
Each of the 6 clients holds one spectral domain of CASIA-MS as its
local dataset (e.g., client_0 → "940nm", client_1 → "850nm", etc.).

Dataset layout assumed
───────────────────────
  data_root/
    {subjectID}_{handSide}_{spectrum}_{iteration}.jpg
  Example: 001_L_850_1.jpg

Protocol
─────────
  Identities : 200 shared IDs across all 6 clients (closed-set).
  Global test : TEST_RATIO (20%) of each ID's samples per spectrum,
                collected once before training.
                Size: ~TEST_RATIO × n_iters × 200_IDs × 6_spectra
  Local train : remaining 80% of each client's samples.

FL Algorithm (FedAvg)
──────────────────────
  Round 0 : server initialises global model, broadcasts weights to all clients.
  Each round (1 … R):
    Step 1 – every client:
               • loads global weights into local model
               • trains for E local epochs on local training set
               • evaluates local model on global test set
               • returns updated weights to server
    Step 2 – server:
               • FedAvg: simple average of all client weight dicts
               • updates global model
               • evaluates global model on global test set

Results saved to BASE_RESULTS_DIR:
  round_results.json   — EER and Rank-1 per round (local + global)
  summary.txt          — final global model performance
  round_{r:04d}/       — per-round checkpoint and eval scores
"""

# ==============================================================
#  CONFIG  — edit this block only
# ==============================================================
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "base_results_dir" : "./rst_fedavg_casiams",

    # ── Dataset ────────────────────────────────────────────────
    "n_ids"            : 200,       # number of shared identities across clients
    "test_ratio"       : 0.20,      # fraction of each ID's samples → global test set

    # ── FL hyperparameters ─────────────────────────────────────
    "n_rounds"         : 100,       # R: total communication rounds
    "local_epochs"     : 1,         # E: local training epochs per round

    # ── Model (CompNet — unchanged) ────────────────────────────
    "img_side"         : 128,
    "embedding_dim"    : 512,
    "dropout"          : 0.25,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,

    # ── Training ───────────────────────────────────────────────
    "batch_size"       : 32,
    "lr"               : 0.001,
    "lr_step"          : 30,
    "lr_gamma"         : 0.8,

    # ── Misc ───────────────────────────────────────────────────
    "random_seed"      : 42,
    "num_workers"      : 4,
    "save_every"       : 10,        # save global checkpoint every N rounds
}
# ==============================================================

import os
import copy
import json
import math
import time
import random
import warnings
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = CONFIG["random_seed"]
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════
#  MODEL  (CompNet — exact copy, unchanged)
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in  = channel_in; self.channel_out = channel_out
        self.kernel_size = kernel_size; self.stride = stride
        self.padding     = padding
        self.init_ratio  = init_ratio if init_ratio > 0 else 1.0
        self.kernel      = 0
        _S = 9.2 * self.init_ratio; _F = 0.057 / self.init_ratio; _G = 2.0
        self.gamma = nn.Parameter(torch.FloatTensor([_G]))
        self.sigma = nn.Parameter(torch.FloatTensor([_S]))
        self.theta = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([_F]))
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def _gen_bank(self, ksize, c_in, c_out, sigma, gamma, theta, f, psi):
        half = ksize // 2; ksz = 2 * half + 1
        y0 = torch.arange(-half, half + 1).float()
        x0 = torch.arange(-half, half + 1).float()
        y  = y0.view(1,-1).repeat(c_out, c_in, ksz, 1)
        x  = x0.view(-1,1).repeat(c_out, c_in, 1, ksz)
        x  = x.to(sigma.device); y = y.to(sigma.device)
        xt =  x*torch.cos(theta.view(-1,1,1,1)) + y*torch.sin(theta.view(-1,1,1,1))
        yt = -x*torch.sin(theta.view(-1,1,1,1)) + y*torch.cos(theta.view(-1,1,1,1))
        gb = -torch.exp(-0.5*((gamma*xt)**2+yt**2)/(8*sigma.view(-1,1,1,1)**2)
            ) * torch.cos(2*math.pi*f.view(-1,1,1,1)*xt+psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        self.kernel = self._gen_bank(self.kernel_size, self.channel_in,
                                     self.channel_out, self.sigma, self.gamma,
                                     self.theta, self.f, self.psi)
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)


class CompetitiveBlock(nn.Module):
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor   = GaborConv2d(channel_in, n_competitor, ksize,
                                   stride, padding, init_ratio)
        self.a       = nn.Parameter(torch.FloatTensor([1]))
        self.b       = nn.Parameter(torch.FloatTensor([0]))
        self.argmax  = nn.Softmax(dim=1)
        self.conv1   = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor(x)
        x = self.argmax((x - self.b) * self.a)
        return self.conv2(self.maxpool(self.conv1(x)))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.s = s; self.m = m
        self.weight      = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            phi  = (torch.where(cosine > 0, phi, cosine) if self.easy_margin
                    else torch.where(cosine > self.th, phi, cosine - self.mm))
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            return self.s * ((one_hot * phi) + ((1 - one_hot) * cosine))
        return self.s * cosine


class CompNet(nn.Module):
    """CompNet = CB1 // CB2 // CB3 + FC(9708→emb_dim) + Dropout + ArcFace."""
    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25):
        super().__init__()
        self.cb1 = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2 = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3 = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)
        self.fc   = nn.Linear(9708, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.arc  = ArcMarginProduct(embedding_dim, num_classes,
                                     s=arcface_s, m=arcface_m)

    def _backbone(self, x):
        x1 = self.cb1(x).flatten(1); x2 = self.cb2(x).flatten(1)
        x3 = self.cb3(x).flatten(1)
        return self.fc(torch.cat([x1, x2, x3], dim=1))

    def forward(self, x, y=None):
        return self.arc(self.drop(self._backbone(x)), y)

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised embedding for matching."""
        return F.normalize(self._backbone(x), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  NORMALISATION
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    def __init__(self, outchannels=1): self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size(); tensor = tensor.view(c, h * w)
        idx = tensor > 0; t = tensor[idx]
        tensor[idx] = t.sub_(t.mean()).div_(t.std() + 1e-6)
        tensor = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════

class PalmDataset(Dataset):
    """Simple palmprint dataset: list of (path, int_label) pairs."""

    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = T.Compose([
            T.Resize((img_side, img_side)),
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), label


# ══════════════════════════════════════════════════════════════
#  DATA LOADING & PARTITIONING
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root):
    """
    Scan data_root for CASIA-MS ROI images.
    Filename format: {subjectID}_{handSide}_{spectrum}_{iteration}.jpg

    Returns
    -------
    data : dict   spectrum → identity → [path, ...]
    """
    data = defaultdict(lambda: defaultdict(list))
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for fname in sorted(os.listdir(data_root)):
        if os.path.splitext(fname)[1].lower() not in img_exts:
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4:
            continue
        subject_id = parts[0]
        hand_side  = parts[1]
        spectrum   = parts[2]
        identity   = f"{subject_id}_{hand_side}"   # e.g. "001_L"
        data[spectrum][identity].append(os.path.join(data_root, fname))
    return data


def build_federated_splits(data_root, n_ids, test_ratio, seed=42):
    """
    Build per-client local train sets and a shared global test set.

    Strategy
    ─────────
    1. Parse dataset; discover all spectra (clients) and identities.
    2. Select n_ids shared identities present in ALL spectra.
    3. For each (spectrum, identity): shuffle samples, take the first
       test_ratio fraction → global test set; the rest → local train set.

    Returns
    -------
    client_data  : list of dicts, one per spectrum/client
                     {"spectrum": str,
                      "train_samples": [(path, label), ...],
                      "label_map": {identity: int}}
    test_samples : [(path, label), ...]   — shared global test set
    label_map    : {identity: int}        — same for all clients / test set
    spectra      : list of spectrum strings (client ID order)
    """
    rng  = random.Random(seed)
    data = parse_casia_ms(data_root)

    spectra = sorted(data.keys())
    print(f"  Spectra found ({len(spectra)}): {spectra}")

    # ── find identities common to ALL spectra ─────────────────────────────
    common_ids = set(data[spectra[0]].keys())
    for sp in spectra[1:]:
        common_ids &= set(data[sp].keys())
    common_ids = sorted(common_ids)
    print(f"  Identities common to all spectra: {len(common_ids)}")

    if len(common_ids) < n_ids:
        raise ValueError(
            f"Requested {n_ids} shared IDs but only {len(common_ids)} "
            f"identities are present in all {len(spectra)} spectra.")

    # sample n_ids shared identities deterministically
    selected_ids = sorted(rng.sample(common_ids, n_ids))
    label_map    = {ident: i for i, ident in enumerate(selected_ids)}

    # ── split per (spectrum, identity) ───────────────────────────────────
    client_data  = []
    test_samples = []

    for sp in spectra:
        local_train = []
        for ident in selected_ids:
            paths = list(data[sp][ident])
            rng.shuffle(paths)
            label   = label_map[ident]
            n_test  = max(1, round(len(paths) * test_ratio))
            # first n_test → global test set; rest → local training set
            for p in paths[:n_test]:
                test_samples.append((p, label))
            for p in paths[n_test:]:
                local_train.append((p, label))

        client_data.append({
            "spectrum"      : sp,
            "train_samples" : local_train,
            "label_map"     : label_map,   # shared label space
        })

        print(f"    Client [{sp:>6}]  train={len(local_train):5d}  "
              f"test_contribution={len(selected_ids)}")

    print(f"  Global test set size: {len(test_samples)}")
    return client_data, test_samples, label_map, spectra


# ══════════════════════════════════════════════════════════════
#  EVALUATION  (cosine similarity, EER, Rank-1)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    for imgs, labs in loader:
        feats.append(model.get_embedding(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def compute_eer(scores_array):
    """Single EER from an Nx2 array of [score, label(+1/-1)]."""
    ins  = scores_array[scores_array[:, 1] ==  1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0:
        return 1.0
    y = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s = np.concatenate([ins, outs])
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)


def evaluate_model(model, test_loader, device, out_dir=None, tag="eval"):
    """
    Evaluate model on the global test set.
    Uses 50% of test IDs as gallery and 50% as probe (random, seeded split).

    Returns
    -------
    eer   : float  EER in [0, 1]
    rank1 : float  Rank-1 accuracy in [0, 100]
    """
    feats, labels = extract_features(model, test_loader, device)

    # deterministic gallery/probe split on test set (50/50 per identity)
    rng       = random.Random(42)
    id_to_idx = defaultdict(list)
    for i, lbl in enumerate(labels):
        id_to_idx[lbl].append(i)

    gal_idx, prb_idx = [], []
    for lbl, idxs in id_to_idx.items():
        idxs_s = list(idxs); rng.shuffle(idxs_s)
        mid    = max(1, len(idxs_s) // 2)
        gal_idx.extend(idxs_s[:mid])
        prb_idx.extend(idxs_s[mid:])

    gal_feats  = feats[gal_idx];  gal_labels  = labels[gal_idx]
    prb_feats  = feats[prb_idx];  prb_labels  = labels[prb_idx]

    # cosine similarity matrix
    sim_matrix = prb_feats @ gal_feats.T   # embeddings are L2-normalised

    # EER
    scores_list, labels_list = [], []
    for i in range(len(prb_feats)):
        for j in range(len(gal_feats)):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if prb_labels[i] == gal_labels[j] else -1)
    scores_arr = np.column_stack([scores_list, labels_list])
    eer = compute_eer(scores_arr)

    # Rank-1
    nn_idx  = np.argmax(sim_matrix, axis=1)
    correct = sum(prb_labels[i] == gal_labels[nn_idx[i]]
                  for i in range(len(prb_feats)))
    rank1   = 100.0 * correct / max(len(prb_feats), 1)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
            for s, l in zip(scores_list, labels_list):
                f.write(f"{s} {l}\n")

    return eer, rank1


# ══════════════════════════════════════════════════════════════
#  FL CLIENT
# ══════════════════════════════════════════════════════════════

class FLClient:
    """
    Represents one federated learning client.

    Each client holds:
      - a local training dataset (one spectral domain)
      - a local CompNet model
      - a local optimiser + scheduler (re-created each round from global weights)
    """

    def __init__(self, client_id, spectrum, train_samples, label_map,
                 num_classes, cfg, device):
        self.client_id     = client_id
        self.spectrum      = spectrum
        self.label_map     = label_map
        self.num_classes   = num_classes
        self.cfg           = cfg
        self.device        = device

        # local dataset
        self.train_dataset = PalmDataset(train_samples, cfg["img_side"])
        self.train_loader  = DataLoader(
            self.train_dataset,
            batch_size  = cfg["batch_size"],
            shuffle     = True,
            num_workers = cfg["num_workers"],
            pin_memory  = True,
        )

        # local model (initialised to random; will be overwritten each round)
        self.model = CompNet(
            num_classes   = num_classes,
            embedding_dim = cfg["embedding_dim"],
            arcface_s     = cfg["arcface_s"],
            arcface_m     = cfg["arcface_m"],
            dropout       = cfg["dropout"],
        ).to(device)

        print(f"  Client {client_id} [{spectrum}] — "
              f"train samples: {len(train_samples)}")

    # ── public interface ────────────────────────────────────────────────────

    def set_weights(self, state_dict):
        """Load global model weights into the local model."""
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def get_weights(self):
        """Return a CPU copy of the local model state dict."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def local_train(self, local_epochs):
        """
        Train the local model for `local_epochs` epochs.
        Optimiser and scheduler are re-created each round so that
        every client starts from the same lr regardless of round number.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        scheduler = lr_scheduler.StepLR(
            optimizer, self.cfg["lr_step"], self.cfg["lr_gamma"])
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(local_epochs):
            running_loss = 0.0; correct = 0; total = 0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                out  = self.model(imgs, labels)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                correct      += out.argmax(1).eq(labels).sum().item()
                total        += imgs.size(0)
            scheduler.step()

        avg_loss = running_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)
        return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════
#  FL SERVER
# ══════════════════════════════════════════════════════════════

class FLServer:
    """
    Central server that:
      - maintains the global CompNet model
      - performs FedAvg aggregation
      - evaluates the global model on the shared test set
    """

    def __init__(self, num_classes, test_samples, cfg, device):
        self.cfg    = cfg
        self.device = device

        # global model
        self.global_model = CompNet(
            num_classes   = num_classes,
            embedding_dim = cfg["embedding_dim"],
            arcface_s     = cfg["arcface_s"],
            arcface_m     = cfg["arcface_m"],
            dropout       = cfg["dropout"],
        ).to(device)

        # shared global test loader
        self.test_loader = DataLoader(
            PalmDataset(test_samples, cfg["img_side"]),
            batch_size  = cfg["batch_size"],
            shuffle     = False,
            num_workers = cfg["num_workers"],
            pin_memory  = True,
        )

        print(f"  Server — global test samples: {len(test_samples)}")

    def get_global_weights(self):
        """Return a CPU copy of the global model state dict."""
        return {k: v.cpu().clone()
                for k, v in self.global_model.state_dict().items()}

    def aggregate(self, client_weight_dicts):
        """
        FedAvg: simple average of all client weight dicts.
        All clients have equal weight (same number of classes, uniform datasets).
        """
        n = len(client_weight_dicts)
        avg_dict = {}
        for key in client_weight_dicts[0].keys():
            stacked = torch.stack(
                [client_weight_dicts[i][key].float() for i in range(n)], dim=0)
            avg_dict[key] = stacked.mean(dim=0)
        self.global_model.load_state_dict(avg_dict)

    def evaluate(self, out_dir=None, tag="global"):
        """Evaluate the global model on the shared test set."""
        return evaluate_model(
            self.global_model, self.test_loader, self.device, out_dir, tag)

    def save_checkpoint(self, path):
        torch.save(self.global_model.state_dict(), path)


# ══════════════════════════════════════════════════════════════
#  PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════

def plot_fl_curves(round_results, out_dir):
    """Plot global EER and Rank-1 across rounds."""
    rounds  = [r["round"] for r in round_results]
    g_eer   = [r["global_eer"]   * 100 for r in round_results]
    g_rank1 = [r["global_rank1"]       for r in round_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(rounds, g_eer, "b-o", markersize=3, label="Global EER (%)")
    ax1.set_xlabel("Round"); ax1.set_ylabel("EER (%)")
    ax1.set_title("Global Model — EER vs. Round")
    ax1.grid(True); ax1.legend()

    ax2.plot(rounds, g_rank1, "g-o", markersize=3, label="Global Rank-1 (%)")
    ax2.set_xlabel("Round"); ax2.set_ylabel("Rank-1 (%)")
    ax2.set_title("Global Model — Rank-1 vs. Round")
    ax2.grid(True); ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fl_global_curves.png"), dpi=150)
    plt.close(fig)

    # per-client local EER
    spectra = [r["spectrum"] for r in round_results[0]["clients"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    for sp in spectra:
        eer_list = [r["clients"][
            next(i for i,c in enumerate(r["clients"]) if c["spectrum"]==sp)
        ]["eer"] * 100 for r in round_results]
        ax.plot(rounds, eer_list, label=sp)
    ax.set_xlabel("Round"); ax.set_ylabel("Local EER (%) after local training")
    ax.set_title("Per-Client Local EER vs. Round")
    ax.grid(True); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fl_client_eer_curves.png"), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def main():
    cfg  = CONFIG
    seed = cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir = cfg["base_results_dir"]
    os.makedirs(base_dir, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Federated Learning — Palmprint (CASIA-MS)")
    print(f"  Device   : {device}")
    print(f"  Rounds   : {cfg['n_rounds']}   "
          f"Local epochs/round : {cfg['local_epochs']}")
    print(f"  Shared IDs : {cfg['n_ids']}   "
          f"Test ratio : {cfg['test_ratio']*100:.0f}%")
    print(f"{'='*62}\n")

    # ── Step 0a: build data splits ───────────────────────────────────────
    print("Building federated data splits …")
    client_data, test_samples, label_map, spectra = build_federated_splits(
        cfg["data_root"], cfg["n_ids"], cfg["test_ratio"], seed=seed)

    num_classes = len(label_map)
    n_clients   = len(client_data)
    print(f"\n  Clients      : {n_clients}  ({spectra})")
    print(f"  Classes      : {num_classes}")
    print(f"  Global test  : {len(test_samples)} samples\n")

    # ── Step 0b: initialise server ────────────────────────────────────────
    print("Initialising server …")
    server = FLServer(num_classes, test_samples, cfg, device)

    # ── Step 0c: initialise clients ───────────────────────────────────────
    print("Initialising clients …")
    clients = []
    for i, cd in enumerate(client_data):
        clients.append(FLClient(
            client_id    = i,
            spectrum     = cd["spectrum"],
            train_samples= cd["train_samples"],
            label_map    = cd["label_map"],
            num_classes  = num_classes,
            cfg          = cfg,
            device       = device,
        ))

    # evaluate global model at round 0 (random init)
    print("\n--- Round 0 (random init) ---")
    g_eer_0, g_rank1_0 = server.evaluate(
        out_dir=os.path.join(base_dir, "round_0000"),
        tag="global_r0000")
    print(f"  [Global init]  EER={g_eer_0*100:.4f}%  Rank-1={g_rank1_0:.2f}%")

    round_results = []

    # ── FL rounds ─────────────────────────────────────────────────────────
    for rnd in range(1, cfg["n_rounds"] + 1):
        t_start  = time.time()
        rnd_dir  = os.path.join(base_dir, f"round_{rnd:04d}")
        os.makedirs(rnd_dir, exist_ok=True)

        global_weights = server.get_global_weights()
        client_weights = []
        client_metrics = []

        # ── Step 1: local training ────────────────────────────────────────
        for client in clients:
            # load global weights into local model
            client.set_weights(global_weights)

            # train locally
            loss, acc = client.local_train(cfg["local_epochs"])

            # evaluate local model on global test set
            c_eer, c_rank1 = evaluate_model(
                client.model, server.test_loader, device,
                out_dir=rnd_dir,
                tag=f"client{client.client_id}_{client.spectrum}")

            client_weights.append(client.get_weights())
            client_metrics.append({
                "client_id" : client.client_id,
                "spectrum"  : client.spectrum,
                "train_loss": round(loss, 6),
                "train_acc" : round(acc, 3),
                "eer"       : round(c_eer, 6),
                "rank1"     : round(c_rank1, 3),
            })

        # ── Step 2: FedAvg aggregation ────────────────────────────────────
        server.aggregate(client_weights)

        # evaluate global model
        g_eer, g_rank1 = server.evaluate(
            out_dir=rnd_dir, tag=f"global_r{rnd:04d}")

        elapsed = time.time() - t_start

        # logging
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Round {rnd:04d}/{cfg['n_rounds']} | "
              f"Global EER={g_eer*100:.4f}%  Rank-1={g_rank1:.2f}%  "
              f"({elapsed:.1f}s)")
        for cm in client_metrics:
            print(f"  Client {cm['client_id']} [{cm['spectrum']:>6}] | "
                  f"loss={cm['train_loss']:.4f}  acc={cm['train_acc']:.1f}%  "
                  f"EER={cm['eer']*100:.3f}%  R1={cm['rank1']:.1f}%")

        round_results.append({
            "round"       : rnd,
            "global_eer"  : round(g_eer, 6),
            "global_rank1": round(g_rank1, 3),
            "clients"     : client_metrics,
        })

        # periodic checkpoint
        if rnd % cfg["save_every"] == 0 or rnd == cfg["n_rounds"]:
            server.save_checkpoint(
                os.path.join(base_dir, f"global_model_r{rnd:04d}.pth"))

    # ── Final reporting ───────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  FL COMPLETE — {cfg['n_rounds']} rounds")
    final = round_results[-1]
    print(f"  Final Global EER   : {final['global_eer']*100:.4f}%")
    print(f"  Final Global Rank-1: {final['global_rank1']:.2f}%")
    print(f"{'='*62}")

    # save round results JSON
    json_path = os.path.join(base_dir, "round_results.json")
    with open(json_path, "w") as f:
        json.dump(round_results, f, indent=2)
    print(f"  Round results saved to: {json_path}")

    # save summary
    with open(os.path.join(base_dir, "summary.txt"), "w") as f:
        f.write(f"Clients      : {n_clients}  ({spectra})\n")
        f.write(f"Shared IDs   : {num_classes}\n")
        f.write(f"Global test  : {len(test_samples)} samples\n")
        f.write(f"Rounds       : {cfg['n_rounds']}\n")
        f.write(f"Local epochs : {cfg['local_epochs']}\n")
        f.write(f"\nFinal Global EER   : {final['global_eer']*100:.6f}%\n")
        f.write(f"Final Global Rank-1: {final['global_rank1']:.3f}%\n")

    # plot curves
    plot_fl_curves(round_results, base_dir)
    print(f"  Plots saved to: {base_dir}")


if __name__ == "__main__":
    main()
