"""
CompNet on CASIA-MS Dataset
==================================================
Single-file implementation with closed-set and open-set protocols.

PROTOCOL options (edit CONFIG below):
  'closed-set' : 80% of samples per identity → train | 20% → test
                 Evaluation: test probe vs train gallery (Rank-1 + EER)

  'open-set'   : 80% of identities → train | 20% of identities → test
                 Within test identities: 50% samples → gallery, 50% → probe
                 Evaluation: Rank-1 identification + EER

Dataset: CASIA-MS-ROI
  Filename format : {subjectID}_{handSide}_{spectrum}_{iteration}.jpg
  Identity key    : subjectID + handSide  (e.g. "001_L")
  All spectra and iterations are treated as samples of the same identity.

Architecture: CompNet (unchanged from original)
  compnet.py → GaborConv2d + CompetitiveBlock + ArcMarginProduct
  Input: 128×128 grayscale   FC input: 9708   Embedding dim: 512
"""

# ==============================================================
#  CONFIG  — edit this block only
# ==============================================================
CONFIG = {
    "protocol"        : "open-set",   # "closed-set" | "open-set"
    "data_root"       : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "results_dir"     : "./rst_casia_ms",
    "img_side"        : 128,            # input image size (128×128 keeps fc=9708)
    "batch_size"      : 32,
    "num_epochs"      : 100,
    "lr"              : 0.001,
    "lr_step"         : 300,
    "lr_gamma"        : 0.8,
    "dropout"         : 0.25,
    "arcface_s"       : 30.0,
    "arcface_m"       : 0.50,
    "embedding_dim"   : 512,
    "train_ratio"     : 0.50,           # fraction of samples (closed) or IDs (open) for training
    "gallery_ratio"   : 0.10,           # open-set only: fraction of test-ID samples → gallery
    "val_ratio"       : 0.10,           # fraction of train samples held out for validation
    "random_seed"     : 42,
    "save_every"      : 10,             # save model every N epochs
    "eval_every"      : 50,             # run full evaluation every N epochs
    "num_workers"     : 4,
}
# ==============================================================

import os
import sys
import math
import time
import random
import pickle
import warnings
import numpy as np
from collections import defaultdict, Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────
SEED = CONFIG["random_seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════
#  MODEL  (exact copy of models/compnet.py — unchanged)
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    """Learnable Gabor Convolution (LGC) layer."""
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        assert channel_in == 1
        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.init_ratio  = init_ratio if init_ratio > 0 else 1.0
        self.kernel      = 0

        _SIGMA  = 9.2   * self.init_ratio
        _FREQ   = 0.057 / self.init_ratio
        _GAMMA  = 2.0

        self.gamma = nn.Parameter(torch.FloatTensor([_GAMMA]))
        self.sigma = nn.Parameter(torch.FloatTensor([_SIGMA]))
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([_FREQ]))
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out,
                     sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin, ymin = -xmax, -ymax
        ksize = xmax - xmin + 1

        y_0 = torch.arange(ymin, ymax + 1).float()
        x_0 = torch.arange(xmin, xmax + 1).float()

        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize)
        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        x_theta = ( x * torch.cos(theta.view(-1,1,1,1))
                  + y * torch.sin(theta.view(-1,1,1,1)))
        y_theta = (-x * torch.sin(theta.view(-1,1,1,1))
                  + y * torch.cos(theta.view(-1,1,1,1)))

        gb = -torch.exp(
            -0.5 * ((gamma * x_theta)**2 + y_theta**2)
            / (8 * sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2 * math.pi * f.view(-1,1,1,1) * x_theta
                      + psi.view(-1,1,1,1))
        gb = gb - gb.mean(dim=[2, 3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self.genGaborBank(
            self.kernel_size, self.channel_in, self.channel_out,
            self.sigma, self.gamma, self.theta, self.f, self.psi)
        self.kernel = kernel
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


class CompetitiveBlock(nn.Module):
    """CB = LGC + soft-argmax + PPU"""
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super().__init__()
        assert channel_in == 1
        self.channel_in    = 1
        self.n_competitor  = n_competitor
        self.init_ratio    = init_ratio

        self.gabor_conv2d = GaborConv2d(1, n_competitor, ksize, stride, padding, init_ratio)
        self.a      = nn.Parameter(torch.FloatTensor([1]))
        self.b      = nn.Parameter(torch.FloatTensor([0]))
        self.argmax = nn.Softmax(dim=1)
        self.conv1  = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2  = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor_conv2d(x)
        x = (x - self.b) * self.a
        x = self.argmax(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        return x


class ArcMarginProduct(nn.Module):
    """ArcFace angular margin product layer."""
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, inp, label=None):
        cosine = F.linear(F.normalize(inp), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            assert label is None
            output = self.s * cosine
        return output


class CompNet(nn.Module):
    """
    CompNet = CB1 // CB2 // CB3 + FC + Dropout + ArcFace output
    Fixed fc input: 9708  (valid for 128×128 input images)
    """
    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25):
        super().__init__()
        self.num_classes   = num_classes
        self.embedding_dim = embedding_dim

        self.cb1 = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2 = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3 = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)

        self.fc       = nn.Linear(9708, embedding_dim)
        self.drop     = nn.Dropout(p=dropout)
        self.arclayer = ArcMarginProduct(embedding_dim, num_classes,
                                         s=arcface_s, m=arcface_m)

    def _extract_concat(self, x):
        x1 = self.cb1(x).view(x.shape[0], -1)
        x2 = self.cb2(x).view(x.shape[0], -1)
        x3 = self.cb3(x).view(x.shape[0], -1)
        return torch.cat((x1, x2, x3), dim=1)

    def forward(self, x, y=None):
        x = self._extract_concat(x)
        x = self.fc(x)
        x = self.drop(x)
        x = self.arclayer(x, y)
        return x

    def getFeatureCode(self, x):
        """Return L2-normalised 512-d embedding (no grad required)."""
        x = self._extract_concat(x)
        x = self.fc(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x


# ══════════════════════════════════════════════════════════════
#  DATASET UTILITIES
# ══════════════════════════════════════════════════════════════

class CASIAMSDataset(Dataset):
    """
    Dataset for CASIA-MS ROI images.
    Images are loaded as grayscale, resized to img_side × img_side,
    and converted to a float tensor in [0, 1].
    No augmentation or normalisation is applied.

    Parameters
    ----------
    samples  : list of (image_path, int_label)
    img_side : resize target (default 128)
    """
    def __init__(self, samples, img_side=128, **kwargs):
        # **kwargs absorbs any legacy `train=` keyword without error
        self.samples  = samples
        self.img_side = img_side
        self.to_tensor = T.Compose([
            T.Resize((img_side, img_side)),
            T.ToTensor(),           # → float32 in [0, 1], shape [1, H, W]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.to_tensor(img)
        return img, label


# ══════════════════════════════════════════════════════════════
#  DATA LOADING & SPLIT LOGIC
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root):
    """
    Scan data_root for files matching  {subjectID}_{handSide}_{spectrum}_{iter}.jpg
    Returns
    -------
    id2paths : dict  identity_key → sorted list of absolute paths
    """
    id2paths = defaultdict(list)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for fname in sorted(os.listdir(data_root)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in exts:
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4:
            print(f"  [WARN] Skipping unexpected filename: {fname}")
            continue
        subject_id = parts[0]
        hand_side  = parts[1]
        identity   = f"{subject_id}_{hand_side}"   # e.g. "001_L"
        id2paths[identity].append(os.path.join(data_root, fname))

    return dict(id2paths)


def make_label_map(identities_sorted):
    """Map sorted identity keys to consecutive integer labels starting at 0."""
    return {ident: idx for idx, ident in enumerate(sorted(identities_sorted))}


def split_closed_set(id2paths, train_ratio=0.80, seed=42):
    """
    Closed-set split:
      - All identities appear in both train and test.
      - Per identity: first `train_ratio` fraction of samples → train,
        remainder → test.

    Returns
    -------
    train_samples, test_samples : list of (path, int_label)
    label_map                   : dict  identity → int
    """
    rng = random.Random(seed)
    identities = sorted(id2paths.keys())
    label_map  = make_label_map(identities)

    train_samples, test_samples = [], []
    for ident in identities:
        paths = id2paths[ident][:]
        rng.shuffle(paths)
        label   = label_map[ident]
        n_train = max(1, int(len(paths) * train_ratio))
        for p in paths[:n_train]:
            train_samples.append((p, label))
        for p in paths[n_train:]:
            test_samples.append((p, label))

    print(f"  [closed-set] identities: {len(identities)} | "
          f"train: {len(train_samples)} | test: {len(test_samples)}")
    return train_samples, test_samples, label_map


def split_open_set(id2paths, train_ratio=0.80, gallery_ratio=0.50,
                   val_ratio=0.10, seed=42):
    """
    Open-set split:
      - 80% of identities → training.
      - 20% of identities → test, never seen during training.
      - Within train identities: `val_ratio` of samples → validation,
        rest → training.  (FIX: previously val was a copy of train.)
      - Within test identities: `gallery_ratio` of samples → gallery,
        rest → probe.

    Returns
    -------
    train_samples   : list of (path, int_label)  — labels from 0 … N_train-1
    val_samples     : list of (path, int_label)  — same label space as train
    gallery_samples : list of (path, int_label)  — labels from 0 … N_test-1
    probe_samples   : list of (path, int_label)  — same label space as gallery
    train_label_map : dict  identity → int (for training)
    test_label_map  : dict  identity → int (for test, independent range)
    """
    rng = random.Random(seed)
    identities = sorted(id2paths.keys())
    rng_ids    = identities[:]
    rng.shuffle(rng_ids)

    n_train = max(1, int(len(rng_ids) * train_ratio))
    train_ids = sorted(rng_ids[:n_train])
    test_ids  = sorted(rng_ids[n_train:])

    train_label_map = make_label_map(train_ids)
    test_label_map  = make_label_map(test_ids)

    train_samples   = []
    val_samples     = []
    gallery_samples = []
    probe_samples   = []

    # ── FIX: split each train identity's samples into train / val ──
    for ident in train_ids:
        paths = id2paths[ident][:]
        rng.shuffle(paths)
        label = train_label_map[ident]
        n_val = max(1, int(len(paths) * val_ratio))
        for p in paths[:n_val]:
            val_samples.append((p, label))
        for p in paths[n_val:]:
            train_samples.append((p, label))

    for ident in test_ids:
        paths = id2paths[ident][:]
        rng.shuffle(paths)
        label    = test_label_map[ident]
        n_gallery = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gallery]:
            gallery_samples.append((p, label))
        for p in paths[n_gallery:]:
            probe_samples.append((p, label))

    print(f"  [open-set] train IDs: {len(train_ids)} | test IDs: {len(test_ids)}")
    print(f"             train samples: {len(train_samples)} | "
          f"val samples: {len(val_samples)} | "
          f"gallery: {len(gallery_samples)} | probe: {len(probe_samples)}")
    return (train_samples, val_samples, gallery_samples, probe_samples,
            train_label_map, test_label_map)


# ══════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_features(net, data_loader, device):
    """
    Returns
    -------
    feats  : np.ndarray  [N, embedding_dim]
    labels : np.ndarray  [N]
    """
    net.eval()
    feats_list  = []
    labels_list = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            codes = net.getFeatureCode(data)
            feats_list.append(codes.cpu().numpy())
            labels_list.append(target.numpy())
    feats  = np.concatenate(feats_list,  axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return feats, labels


# ══════════════════════════════════════════════════════════════
#  MATCHING & METRICS  (inlined — no subprocess)
# ══════════════════════════════════════════════════════════════

def angular_distance(f1, f2):
    """Normalised angular distance in [0, 1] between two L2-norm vectors."""
    cos = np.dot(f1, f2)
    return np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi


def compute_scores(probe_feats, probe_labels,
                   gallery_feats, gallery_labels):
    """
    Compute all probe × gallery matching scores.

    Returns
    -------
    s : list of float  — angular distances
    l : list of int    — 1 (genuine) or -1 (impostor)
    """
    s, l = [], []
    n_probe   = probe_feats.shape[0]
    n_gallery = gallery_feats.shape[0]
    for i in range(n_probe):
        for j in range(n_gallery):
            d = angular_distance(probe_feats[i], gallery_feats[j])
            s.append(d)
            l.append(1 if probe_labels[i] == gallery_labels[j] else -1)
    return s, l


def compute_eer(scores, labels):
    """
    Compute EER from a list of (score, label) pairs.
    Scores are distances (lower = more similar → genuine scores are smaller).

    Returns
    -------
    eer       : float  — EER in [0, 1]
    thresh    : float  — threshold at EER
    roc_auc   : float
    eer_half  : float  — (FAR + FRR) / 2 at closest operating point
    """
    scores = np.array(scores)
    labels = np.array(labels)

    # Convert to similarity (negate distance) so that genuine > impostor
    sim = -scores

    in_scores  = sim[labels ==  1]
    out_scores = sim[labels == -1]

    y    = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
    sall = np.concatenate([in_scores, out_scores])

    fpr, tpr, thresholds = roc_curve(y, sall, pos_label=1)
    roc_auc = auc(fpr, tpr)

    eer    = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))

    diff    = np.abs(fpr - (1 - tpr))
    idx     = np.argmin(diff)
    eer_half = (fpr[idx] + (1 - tpr[idx])) / 2.0

    return eer, thresh, roc_auc, eer_half, fpr, tpr, thresholds


def compute_rank1(probe_feats, probe_labels,
                  gallery_feats, gallery_labels,
                  scores_matrix=None):
    """
    Rank-1 identification accuracy.

    scores_matrix : optional pre-computed [n_probe × n_gallery] distance array.
    """
    n_probe   = probe_feats.shape[0]
    n_gallery = gallery_feats.shape[0]

    if scores_matrix is None:
        dist = np.zeros((n_probe, n_gallery))
        for i in range(n_probe):
            for j in range(n_gallery):
                dist[i, j] = angular_distance(probe_feats[i], gallery_feats[j])
    else:
        dist = scores_matrix

    correct = 0
    for i in range(n_probe):
        best_j = int(np.argmin(dist[i]))
        if probe_labels[i] == gallery_labels[best_j]:
            correct += 1
    rank1 = correct / n_probe * 100.0
    return rank1, dist


def compute_aggregated_eer(dist_matrix, prb_labels, gal_labels):
    """
    Aggregated EER (best-of-N per gallery class).

    For every probe i and every gallery identity class c, take the MINIMUM
    distance among all gallery samples that belong to class c.  This produces
    one score per (probe, gallery-class) pair — a fairer metric when multiple
    gallery samples per identity exist, matching the aggregation in test.py.

    Parameters
    ----------
    dist_matrix : np.ndarray [n_probe, n_gallery]  — angular distances
    prb_labels  : np.ndarray [n_probe]              — integer class labels
    gal_labels  : np.ndarray [n_gallery]            — integer class labels

    Returns
    -------
    aggr_s : list of float  — aggregated distances
    aggr_l : list of int    — 1 (genuine) or -1 (impostor)
    """
    class_ids = sorted(set(gal_labels.tolist()))
    n_probe   = dist_matrix.shape[0]

    aggr_s, aggr_l = [], []
    for i in range(n_probe):
        for cls in class_ids:
            cls_mask = (gal_labels == cls)          # boolean over gallery axis
            min_dist = dist_matrix[i, cls_mask].min()
            aggr_s.append(min_dist)
            aggr_l.append(1 if prb_labels[i] == cls else -1)

    return aggr_s, aggr_l


# ══════════════════════════════════════════════════════════════
#  PLOTTING & SAVING
# ══════════════════════════════════════════════════════════════

def save_scores_txt(scores, labels, path):
    with open(path, "w") as f:
        for s, l in zip(scores, labels):
            f.write(f"{s} {l}\n")


def plot_and_save(fpr, tpr, fnr, thresholds, eer, rank1, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    fpr_pct = fpr * 100
    tpr_pct = tpr * 100
    fnr_pct = fnr * 100
    thr     = thresholds

    pdf_path = os.path.join(out_dir, f"roc_det_{tag}.pdf")
    with PdfPages(pdf_path) as pdf:
        # ROC
        plt.figure()
        plt.plot(fpr_pct, tpr_pct, "b-^", label="ROC")
        plt.plot(np.linspace(0,100,101), np.linspace(100,0,101), "k-", label="EER")
        plt.xlim([0, 5]); plt.ylim([90, 100])
        plt.legend(); plt.grid(True)
        plt.title(f"ROC  |  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
        plt.xlabel("FAR (%)"); plt.ylabel("GAR (%)")
        plt.savefig(os.path.join(out_dir, f"ROC_{tag}.png"))
        pdf.savefig(); plt.close()

        # DET
        plt.figure()
        plt.plot(fpr_pct, fnr_pct, "b-^", label="DET")
        plt.plot(np.linspace(0,100,101), np.linspace(0,100,101), "k-", label="EER")
        plt.xlim([0, 5]); plt.ylim([0, 5])
        plt.legend(); plt.grid(True)
        plt.title("DET curve")
        plt.xlabel("FAR (%)"); plt.ylabel("FRR (%)")
        plt.savefig(os.path.join(out_dir, f"DET_{tag}.png"))
        pdf.savefig(); plt.close()

        # FAR/FRR vs threshold
        plt.figure()
        plt.plot(thr, fpr_pct, "r-.", label="FAR")
        plt.plot(thr, fnr_pct, "b-^", label="FRR")
        plt.legend(); plt.grid(True)
        plt.title("FAR and FRR Curves")
        plt.xlabel("Threshold"); plt.ylabel("FAR / FRR (%)")
        plt.savefig(os.path.join(out_dir, f"FAR_FRR_{tag}.png"))
        pdf.savefig(); plt.close()


def plot_loss_acc(train_losses, val_losses, train_acc, val_acc, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ep = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(ep, train_losses, "b", label="train loss")
    plt.plot(ep, val_losses,   "r", label="val loss")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.savefig(os.path.join(out_dir, "losses.png")); plt.close()

    plt.figure()
    plt.plot(ep, train_acc, "b", label="train acc")
    plt.plot(ep, val_acc,   "r", label="val acc")
    plt.legend(); plt.grid(True)
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)")
    plt.savefig(os.path.join(out_dir, "accuracy.png")); plt.close()


def plot_gi_histogram(in_scores, out_scores, out_dir, tag):
    """Genuine-Impostor matching score distribution."""
    os.makedirs(out_dir, exist_ok=True)
    samples = 100
    in_arr  = np.array(in_scores)
    out_arr = np.array(out_scores)

    def normalise_hist(arr):
        lo, hi = arr.min(), arr.max()
        idx_arr = np.round((arr - lo) / (hi - lo + 1e-10) * samples).astype(int)
        h = np.zeros(samples + 1)
        for v in idx_arr:
            h[v] += 1
        h = h / h.sum() * 100
        x = np.linspace(0, 1, samples + 1) * (hi - lo) + lo
        return x, h

    xi, hi  = normalise_hist(in_arr)
    xo, ho  = normalise_hist(out_arr)

    plt.figure()
    plt.plot(xo, ho, "r", label="Impostor")
    plt.plot(xi, hi, "b", label="Genuine")
    plt.legend(fontsize=13)
    plt.xlabel("Matching Score", fontsize=13)
    plt.ylabel("Percentage (%)", fontsize=13)
    plt.ylim([0, 1.2 * max(hi.max(), ho.max())])
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"GI_{tag}.png")); plt.close()


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def run_one_epoch(epoch, model, loader, criterion, optimizer, device,
                  phase="training"):
    if phase == "training":
        model.train()
    else:
        model.eval()

    running_loss    = 0.0
    running_correct = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        if phase == "training":
            optimizer.zero_grad()
            output = model(data, target)
        else:
            with torch.no_grad():
                output = model(data, None)

        loss = criterion(output, target)
        running_loss += loss.item()

        preds = output.argmax(dim=1, keepdim=True)
        running_correct += preds.eq(target.view_as(preds)).sum().item()

        if phase == "training":
            loss.backward()
            optimizer.step()

    total    = len(loader.dataset)
    avg_loss = running_loss / total
    acc      = 100.0 * running_correct / total
    return avg_loss, acc


# ══════════════════════════════════════════════════════════════
#  EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════



def evaluate(net, train_loader, test_probe_loader, test_gallery_loader,
             device, out_dir, tag="eval"):
    """
    Shared evaluation for both protocols.

    For closed-set  : train_loader = train split,  test loaders = test split
                      (probe and gallery are the SAME test split,
                       matched against the training gallery)
    For open-set    : train_loader = None (not used for matching),
                      gallery = open-set gallery, probe = open-set probe

    FIX: Always compute both pairwise EER and aggregated EER.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- extract features ---
    print("  Extracting gallery features …")
    gal_feats, gal_labels = extract_features(net, test_gallery_loader, device)

    print("  Extracting probe features …")
    prb_feats, prb_labels = extract_features(net, test_probe_loader, device)

    n_probe   = prb_feats.shape[0]
    n_gallery = gal_feats.shape[0]
    print(f"  probe: {n_probe}  gallery: {n_gallery}")

    # --- compute all pairwise scores ---
    print("  Computing pairwise distances …")
    s, l = [], []
    dist_matrix = np.zeros((n_probe, n_gallery))
    for i in range(n_probe):
        for j in range(n_gallery):
            d = angular_distance(prb_feats[i], gal_feats[j])
            dist_matrix[i, j] = d
            s.append(d)
            l.append(1 if prb_labels[i] == gal_labels[j] else -1)

    # save raw scores
    scores_path = os.path.join(out_dir, f"scores_{tag}.txt")
    save_scores_txt(s, l, scores_path)

    # --- Pairwise EER ---
    eer, thresh, roc_auc, eer_half, fpr, tpr, thresholds = compute_eer(s, l)
    fnr = 1 - tpr
    print(f"  Pairwise EER: {eer*100:.4f}%  |  thresh: {thresh:.4f}  |  AUC: {roc_auc:.6f}")
    print(f"  Pairwise EER½: {eer_half*100:.4f}%")

    # --- Rank-1 ---
    rank1, _ = compute_rank1(prb_feats, prb_labels,
                              gal_feats, gal_labels,
                              scores_matrix=dist_matrix)
    print(f"  Rank-1 acc: {rank1:.3f}%")

    # --- GI histogram ---
    in_scores  = [s[k] for k in range(len(s)) if l[k] ==  1]
    out_scores = [s[k] for k in range(len(s)) if l[k] == -1]
    plot_gi_histogram(in_scores, out_scores, out_dir, tag)

    # --- plots ---
    plot_and_save(fpr, tpr, fnr, thresholds, eer, rank1, out_dir, tag)

    # save text summary
    with open(os.path.join(out_dir, f"rst_{tag}.txt"), "w") as f:
        f.write(f"Pairwise EER  : {eer*100:.6f}%\n")
        f.write(f"Pairwise EER½ : {eer_half*100:.6f}%\n")
        f.write(f"Threshold     : {thresh:.4f}\n")
        f.write(f"AUC           : {roc_auc:.10f}\n")
        f.write(f"Rank-1        : {rank1:.3f}%\n")

    # --- FIX: Always compute aggregated EER (min-distance per gallery class) ---
    n_gallery_classes = len(set(gal_labels.tolist()))
    if n_gallery_classes < n_gallery:
        # multiple gallery samples per class exist → aggregation is meaningful
        print("  Computing aggregated EER …")
        aggr_s, aggr_l = compute_aggregated_eer(dist_matrix, prb_labels, gal_labels)
        (aggr_eer, aggr_thresh, aggr_auc,
         aggr_eer_half, aggr_fpr, aggr_tpr, _) = compute_eer(aggr_s, aggr_l)
        print(f"  Aggregated EER: {aggr_eer*100:.4f}%  |  AUC: {aggr_auc:.6f}")
        with open(os.path.join(out_dir, f"rst_{tag}.txt"), "a") as f:
            f.write(f"\nAggregated EER      : {aggr_eer*100:.6f}%\n")
            f.write(f"Aggregated EER_half : {aggr_eer_half*100:.6f}%\n")
            f.write(f"Aggregated AUC      : {aggr_auc:.10f}\n")
    else:
        aggr_eer = eer   # single sample per class → aggregated = pairwise

    return eer, aggr_eer, rank1


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # ---------- unpack config ----------
    protocol       = CONFIG["protocol"]
    data_root      = CONFIG["data_root"]
    results_dir    = CONFIG["results_dir"]
    img_side       = CONFIG["img_side"]
    batch_size     = CONFIG["batch_size"]
    num_epochs     = CONFIG["num_epochs"]
    lr             = CONFIG["lr"]
    lr_step        = CONFIG["lr_step"]
    lr_gamma       = CONFIG["lr_gamma"]
    dropout        = CONFIG["dropout"]
    arc_s          = CONFIG["arcface_s"]
    arc_m          = CONFIG["arcface_m"]
    emb_dim        = CONFIG["embedding_dim"]
    train_ratio    = CONFIG["train_ratio"]
    gallery_ratio  = CONFIG["gallery_ratio"]
    val_ratio      = CONFIG["val_ratio"]
    seed           = CONFIG["random_seed"]
    save_every     = CONFIG["save_every"]
    eval_every     = CONFIG["eval_every"]
    nw             = CONFIG["num_workers"]

    assert protocol in ("closed-set", "open-set"), \
        f"Unknown protocol: {protocol}. Use 'closed-set' or 'open-set'."

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Protocol : {protocol}")
    print(f"  Device   : {device}")
    print(f"  Data     : {data_root}")
    print(f"{'='*60}\n")

    # ---------- parse dataset ----------
    print("Scanning dataset …")
    id2paths = parse_casia_ms(data_root)
    n_total_ids = len(id2paths)
    n_total_imgs = sum(len(v) for v in id2paths.values())
    print(f"  Found {n_total_ids} identities, {n_total_imgs} images total.\n")

    # ---------- protocol-specific split ----------
    if protocol == "closed-set":
        train_samples, test_samples, label_map = split_closed_set(
            id2paths, train_ratio=train_ratio, seed=seed)
        num_classes = len(label_map)

        train_dataset   = CASIAMSDataset(train_samples, img_side=img_side)
        test_dataset    = CASIAMSDataset(test_samples,  img_side=img_side)
        train_gal_data  = CASIAMSDataset(train_samples, img_side=img_side)

        train_loader    = DataLoader(train_dataset,  batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=True)
        val_loader      = DataLoader(test_dataset,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        gallery_loader  = DataLoader(train_gal_data, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        probe_loader    = val_loader

        train_label_counts = Counter(l for _, l in train_samples)
        train_num_per_class = int(np.median(list(train_label_counts.values())))
        print(f"  [closed-set] #classes={num_classes} | "
              f"~{train_num_per_class} train samples/class\n")

    else:  # open-set
        # ── FIX: split_open_set now returns a proper val split ──
        (train_samples, val_samples, gallery_samples, probe_samples,
         train_label_map, test_label_map) = split_open_set(
            id2paths, train_ratio=train_ratio,
            gallery_ratio=gallery_ratio,
            val_ratio=val_ratio, seed=seed)
        num_classes = len(train_label_map)

        train_dataset   = CASIAMSDataset(train_samples,   img_side=img_side)
        val_dataset     = CASIAMSDataset(val_samples,      img_side=img_side)
        gallery_dataset = CASIAMSDataset(gallery_samples,  img_side=img_side)
        probe_dataset   = CASIAMSDataset(probe_samples,    img_side=img_side)

        train_loader    = DataLoader(train_dataset,   batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=True)
        val_loader      = DataLoader(val_dataset,     batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        gallery_loader  = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        probe_loader    = DataLoader(probe_dataset,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

        print(f"  [open-set] #train_classes={num_classes}\n")

    # ---------- model ----------
    print(f"Building CompNet — num_classes={num_classes} …")
    net = CompNet(num_classes=num_classes, embedding_dim=emb_dim,
                  arcface_s=arc_s, arcface_m=arc_m, dropout=dropout)
    net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
        net = DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # ---------- training loop ----------
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0
    best_eer     = 1.0

    # cached quick-eval metrics shown in every-10-epoch print
    last_eer   = float("nan")
    last_rank1 = float("nan")

    print(f"\nStarting training for {num_epochs} epochs …")
    print(f"  EER / Rank-1 computed every {eval_every} epochs and shown in every 10-epoch log.\n")

    for epoch in range(num_epochs):
        t_loss, t_acc = run_one_epoch(
            epoch, net, train_loader, criterion, optimizer, device, "training")
        v_loss, v_acc = run_one_epoch(
            epoch, net, val_loader,   criterion, optimizer, device, "testing")
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        _net = net.module if isinstance(net, DataParallel) else net

        # ── periodic evaluation (EER / Rank-1) ───────────────────────────────
        # runs at epoch 0 and every eval_every epochs thereafter
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            eval_net = _net
            tag = f"ep{epoch:04d}_{protocol.replace('-','')}"
            cur_eer, cur_aggr_eer, cur_rank1 = evaluate(
                eval_net,
                train_loader,
                probe_loader,
                gallery_loader,
                device,
                out_dir=rst_eval,
                tag=tag,
            )
            last_eer   = cur_eer
            last_rank1 = cur_rank1

            if cur_eer < best_eer:
                best_eer = cur_eer
                torch.save(_net.state_dict(),
                           os.path.join(results_dir, "net_params_best_eer.pth"))
                print(f"  *** New best EER: {best_eer*100:.4f}% ***")

        # ── every-10-epoch console print ──────────────────────────────────────
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            ts = time.strftime("%H:%M:%S")
            eer_str   = f"{last_eer*100:.4f}%"   if not math.isnan(last_eer)   else "N/A"
            rank1_str = f"{last_rank1:.2f}%"      if not math.isnan(last_rank1) else "N/A"
            print(
                f"[{ts}] ep {epoch:04d} | "
                f"loss  train={t_loss:.5f}  val={v_loss:.5f} | "
                f"cls-acc  train={t_acc:.2f}%  val={v_acc:.2f}% | "
                f"EER={eer_str}  Rank-1={rank1_str}"
            )

        # ── save best classification model ────────────────────────────────────
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params_best.pth"))

        # ── periodic checkpoint + loss/acc plots ──────────────────────────────
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params.pth"))
            plot_loss_acc(train_losses, val_losses, train_accs, val_accs, results_dir)

    # ---------- final evaluation with best model ----------
    print("\n=== Final evaluation with best EER model ===")
    best_model_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(results_dir, "net_params_best.pth")

    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_model_path, map_location=device))

    final_eer, final_aggr_eer, final_rank1 = evaluate(
        eval_net,
        train_loader,
        probe_loader,
        gallery_loader,
        device,
        out_dir=rst_eval,
        tag=f"FINAL_{protocol.replace('-','')}",
    )

    print(f"\n{'='*60}")
    print(f"  PROTOCOL : {protocol}")
    print(f"  FINAL Pairwise EER   : {final_eer*100:.4f}%")
    print(f"  FINAL Aggregated EER : {final_aggr_eer*100:.4f}%")
    print(f"  FINAL Rank-1         : {final_rank1:.3f}%")
    print(f"  Results saved to: {results_dir}")
    print(f"{'='*60}\n")

    # save final summary
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Protocol  : {protocol}\n")
        f.write(f"Data root : {data_root}\n")
        f.write(f"Identities: {n_total_ids}\n")
        f.write(f"Images    : {n_total_imgs}\n")
        f.write(f"Classes (train): {num_classes}\n")
        f.write(f"Best val acc       : {best_val_acc:.3f}%\n")
        f.write(f"Final Pairwise EER : {final_eer*100:.6f}%\n")
        f.write(f"Final Aggreg. EER  : {final_aggr_eer*100:.6f}%\n")
        f.write(f"Final Rank-1       : {final_rank1:.3f}%\n")


if __name__ == "__main__":
    main()
