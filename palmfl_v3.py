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

Protocol (Open-Set, Shared-ID)
────────────────────────────────
  Identities : N total IDs shared across all 6 clients.
  ID split   : K_TEST (20%) of IDs → test set  |  remaining 80% → training.
               Train and test IDs are fully disjoint (open-set).
  Local train: all samples of train IDs in each client's spectral domain.
               Train IDs and label space are identical across all clients.
  Global test: all samples of test IDs across all 6 spectra, split into:
               • Gallery : GALLERY_RATIO (50%) of each (spectrum, identity)
                           pair's samples.
               • Probe   : remaining 50% of each (spectrum, identity) pair.
               Both gallery and probe contain all test IDs and all spectra.
               The gallery/probe split is fixed before training and shared
               across all rounds and clients.

FL Algorithm (FedAvg)
──────────────────────
  Round 0 : server initialises global model, broadcasts weights to all clients.
  Each round (1 … R):
    Step 1 – every client:
               • loads global weights into local model
               • trains for E local epochs on local training set
               • evaluates local model on shared gallery/probe test sets
               • returns updated weights to server
    Step 2 – server:
               • FedAvg: simple average of all client weight dicts
               • updates global model
               • evaluates global model on shared gallery/probe test sets

Results saved to BASE_RESULTS_DIR:
  results.txt  — tab-separated EER and Rank-1 per round (global + per-client)
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
    "k_test"        : 0.20,   # fraction of IDs allocated to test set
    "gallery_ratio" : 0.20,   # fraction of test-ID samples → gallery

    "fft_beta"  : 0.05,   # Gaussian mask sigma fraction (cut-off frequency for fft swapping)
    "M"         : 3,      # total augmented images per sample (1 original + M-1 synthetic)
    "use_fft_aug" : True,   # True → FFT style augmentation | False → standard training
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
#  Knowledge Sharing and Augmentation
# ══════════════════════════════════════════════════════════════
def gaussian_mask(H, W, beta):
    """Soft Gaussian low-frequency mask centred at DC."""
    sigma = min(H, W) * beta
    cy, cx = H // 2, W // 2
    ys = np.arange(H) - cy
    xs = np.arange(W) - cx
    xs, ys = np.meshgrid(xs, ys)
    return np.exp(-(xs**2 + ys**2) / (2 * sigma**2)).astype(np.float32)


def extract_style_template(img_np, beta):
    """
    Extract the low-frequency amplitude template from an image.
    Returns the fftshifted amplitude array (H, W) or (H, W, C).
    This is the shareable style descriptor — no texture/identity info.
    """
    def _extract_channel(ch):
        amp = np.abs(np.fft.fft2(ch))
        return np.fft.fftshift(amp)

    if img_np.ndim == 2:
        return _extract_channel(img_np)
    return np.stack([_extract_channel(img_np[..., c])
                     for c in range(img_np.shape[2])], axis=-1)


def apply_style_template(img_np, amp_template, beta):
    """
    Swap the low-frequency amplitude of img_np with amp_template
    using a Gaussian soft mask. Phase (texture/structure) is preserved.
    Returns a float32 image in [0, 1].
    """
    H, W = img_np.shape[:2]
    mask = gaussian_mask(H, W, beta)

    def _apply_channel(ch, amp_tpl_ch):
        fft   = np.fft.fft2(ch)
        amp_s = np.fft.fftshift(np.abs(fft))
        pha   = np.angle(fft)
        # soft blend: low-freq from template, high-freq from original
        amp_syn = (1 - mask) * amp_s + mask * amp_tpl_ch
        amp_syn = np.fft.ifftshift(amp_syn)
        return np.clip(np.fft.ifft2(amp_syn * np.exp(1j * pha)).real, 0, 1)

    if img_np.ndim == 2:
        return _apply_channel(img_np, amp_template).astype(np.float32)
    return np.stack([_apply_channel(img_np[..., c], amp_template[..., c])
                     for c in range(img_np.shape[2])], axis=-1).astype(np.float32)

# ══════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════
class PalmDataset(Dataset):
    """Plain dataset for gallery/probe evaluation — no augmentation."""
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


class FFTAugmentedDataset(Dataset):
    """
    Training dataset with FFT style augmentation.
    For each original sample, M-1 synthetic images are created by swapping
    the low-frequency amplitude with randomly selected style templates from
    M-1 other clients. The dataset returns M images per original sample.
    """
    def __init__(self, samples, style_bank, client_id, M, beta, img_side):
        self.samples         = samples
        self.style_bank      = style_bank   # {client_id: [amp_array, ...]}
        self.client_id       = client_id
        self.M               = M
        self.beta            = beta
        self.img_side        = img_side
        self.other_ids       = [cid for cid in style_bank if cid != client_id]
        self.transform       = T.Compose([
            T.Resize((img_side, img_side)),
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])

    def __len__(self):
        return len(self.samples) * self.M

    def _load_np(self, path):
        img = Image.open(path).convert("L")
        img = img.resize((self.img_side, self.img_side), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _to_tensor(self, img_np):
        pil = Image.fromarray((img_np * 255).astype(np.uint8), mode="L")
        return self.transform(pil)

    def __getitem__(self, idx):
        sample_idx = idx // self.M    # which original sample
        aug_idx    = idx  % self.M    # 0 = original, 1..M-1 = synthetic

        path, label = self.samples[sample_idx]
        img_np = self._load_np(path)

        if aug_idx == 0 or not self.other_ids:
            # original sample — always included
            return self._to_tensor(img_np), label

        # select one random client (different each aug_idx) and one random template
        rand_client   = random.choice(self.other_ids)
        rand_template = random.choice(self.style_bank[rand_client])
        img_syn       = apply_style_template(img_np, rand_template, self.beta)
        return self._to_tensor(img_syn), label


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


def build_federated_splits(data_root, n_ids, k_test, gallery_ratio, seed=42):
    rng  = random.Random(seed)
    data = parse_casia_ms(data_root)

    spectra   = sorted(data.keys())
    n_clients = len(spectra)
    print(f"  Spectra found ({n_clients}): {spectra}")

    # identities common to ALL spectra
    common_ids = set(data[spectra[0]].keys())
    for sp in spectra[1:]:
        common_ids &= set(data[sp].keys())
    common_ids = sorted(common_ids)
    print(f"  Identities common to all spectra: {len(common_ids)}")

    if len(common_ids) < n_ids:
        raise ValueError(
            f"Requested {n_ids} IDs but only {len(common_ids)} "
            f"are present in all spectra.")

    # select n_ids and split into test / train
    selected_ids = sorted(rng.sample(common_ids, n_ids))
    rng.shuffle(selected_ids)
    n_test_ids     = max(1, round(n_ids * k_test))
    test_ids       = sorted(selected_ids[:n_test_ids])
    train_ids      = sorted(selected_ids[n_test_ids:])
    test_label_map = {ident: i for i, ident in enumerate(test_ids)}

    print(f"  Total train IDs: {len(train_ids)}  |  Test IDs: {len(test_ids)}")

    # uniformly partition train IDs among clients — no overlap
    rng.shuffle(train_ids)
    client_id_splits = [[] for _ in range(n_clients)]
    for i, ident in enumerate(train_ids):
        client_id_splits[i % n_clients].append(ident)

    # equalise client sizes by trimming to the smallest partition
    min_ids = min(len(ids) for ids in client_id_splits)
    client_id_splits = [sorted(ids[:min_ids]) for ids in client_id_splits]
    n_dropped = len(train_ids) - min_ids * n_clients
    print(f"  IDs per client : {min_ids}  (dropped {n_dropped} to equalise)")

    # per-client local train sets — each client has its OWN IDs and label space
    client_data = []
    for i, sp in enumerate(spectra):
        c_ids       = client_id_splits[i]
        c_label_map = {ident: j for j, ident in enumerate(c_ids)}
        local_train = [
            (p, c_label_map[ident])
            for ident in c_ids
            for p in data[sp][ident]
        ]
        client_data.append({
            "spectrum"      : sp,
            "train_samples" : local_train,
            "label_map"     : c_label_map,
            "num_classes"   : min_ids,
        })
        print(f"    Client {i} [{sp:>6}]  "
              f"train IDs={min_ids}  samples={len(local_train)}")

    # fixed global test set — test IDs, all spectra, gallery/probe split
    # split within each (spectrum, identity) pair by gallery_ratio
    gallery_samples, probe_samples = [], []
    for sp in spectra:
        for ident in test_ids:
            paths = list(data[sp][ident])
            rng.shuffle(paths)
            label = test_label_map[ident]
            n_gal = max(1, round(len(paths) * gallery_ratio))
            for p in paths[:n_gal]:
                gallery_samples.append((p, label))
            for p in paths[n_gal:]:
                probe_samples.append((p, label))

    print(f"  Gallery: {len(gallery_samples)}  |  Probe: {len(probe_samples)}")
    return (client_data, gallery_samples, probe_samples,
            test_label_map, spectra)

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


def evaluate_model(model, gallery_loader, probe_loader, device,
                   out_dir=None, tag="eval"):
    gal_feats, gal_labels = extract_features(model, gallery_loader, device)
    prb_feats, prb_labels = extract_features(model, probe_loader,   device)

    sim_matrix  = prb_feats @ gal_feats.T
    scores_list, labels_list = [], []
    for i in range(len(prb_feats)):
        for j in range(len(gal_feats)):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if prb_labels[i] == gal_labels[j] else -1)

    scores_arr = np.column_stack([scores_list, labels_list])
    eer        = compute_eer(scores_arr)

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
        self.train_samples = train_samples   # raw list — no PalmDataset wrapper
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
              f"train IDs: {num_classes}  samples: {len(train_samples)}")

    # ── public interface ────────────────────────────────────────────────────

    def set_weights(self, backbone_state_dict):
        """Load global backbone weights. ArcFace head kept local."""
        local_state = self.model.state_dict()
        for key, val in backbone_state_dict.items():
            if key in local_state and local_state[key].shape == val.shape:
                local_state[key] = val.clone()
        self.model.load_state_dict(local_state)
    
    def get_weights(self):
        """Return backbone weights only — exclude ArcFace head."""
        return {k: v.cpu().clone()
                for k, v in self.model.state_dict().items()
                if not k.startswith("arc.")}

    def local_train(self, local_epochs, style_bank, M):
        if self.cfg["use_fft_aug"] and style_bank and M > 1:
            dataset = FFTAugmentedDataset(
                samples    = self.train_samples,   # ← was self.train_dataset.samples
                style_bank = style_bank,
                client_id  = self.client_id,
                M          = M,
                beta       = self.cfg["fft_beta"],
                img_side   = self.cfg["img_side"],
            )
        else:
            dataset = PalmDataset(self.train_samples, self.cfg["img_side"])
    
        train_loader = DataLoader(
            dataset,
            batch_size     = self.cfg["batch_size"],
            shuffle        = True,
            num_workers    = self.cfg["num_workers"],
            pin_memory     = True,
            worker_init_fn = worker_init_fn,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        scheduler = lr_scheduler.StepLR(
            optimizer, self.cfg["lr_step"], self.cfg["lr_gamma"])
        criterion = nn.CrossEntropyLoss()
    
        self.model.train()
        running_loss = 0.0; correct = 0; total = 0
        for epoch in range(local_epochs):
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                out  = self.model(imgs, labels)
                loss = criterion(out, labels)
                loss.backward(); optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                correct      += out.argmax(1).eq(labels).sum().item()
                total        += imgs.size(0)
            scheduler.step()
    
        return running_loss / max(total, 1), 100.0 * correct / max(total, 1)
      
    def extract_style_templates(self):
        """
        Extract low-frequency amplitude templates from all local training samples.
        These are safe to share — they capture only global style/illumination,
        not identity-discriminative texture.
        Returns a list of amplitude arrays, one per training sample.
        """
        templates = []
        img_side  = self.cfg["img_side"]
        beta      = self.cfg["fft_beta"]
        for path, _ in self.train_samples:   # ← was self.train_dataset.samples
            img = Image.open(path).convert("L")
            img = img.resize((img_side, img_side), Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            templates.append(extract_style_template(img_np, beta))
        print(f"  Client {self.client_id} [{self.spectrum}] "
              f"— extracted {len(templates)} style templates")
        return templates


# ══════════════════════════════════════════════════════════════
#  FL SERVER
# ══════════════════════════════════════════════════════════════

class FLServer:
    """
    Central server that:
      - maintains the global CompNet model
      - performs FedAvg aggregation
      - evaluates the global model on the shared gallery/probe test sets
    """
    def __init__(self, num_classes, gallery_samples, probe_samples, cfg, device):
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

        # shared gallery and probe loaders (fixed, created once before training)
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

        print(f"  Server — gallery: {len(gallery_samples)}  "
              f"probe: {len(probe_samples)}")

    def get_global_weights(self):
        """Return a CPU copy of the global model state dict."""
        return {k: v.cpu().clone()
                for k, v in self.global_model.state_dict().items()}

    def aggregate(self, client_weight_dicts):
        """FedAvg on backbone parameters only (arc.* excluded)."""
        n        = len(client_weight_dicts)
        avg_dict = {}
        for key in client_weight_dicts[0].keys():
            stacked      = torch.stack(
                [client_weight_dicts[i][key].float() for i in range(n)], dim=0)
            avg_dict[key] = stacked.mean(dim=0)
        global_state = self.global_model.state_dict()
        global_state.update(avg_dict)
        self.global_model.load_state_dict(global_state)

    def evaluate(self, out_dir=None, tag="global"):
        """Evaluate the global model on the shared gallery and probe sets."""
        return evaluate_model(
            self.global_model,
            self.gallery_loader,
            self.probe_loader,
            self.device, out_dir, tag)

    def save_checkpoint(self, path):
        torch.save(self.global_model.state_dict(), path)


# ══════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
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
    print(f"  Device   : {device}")
    print(f"  Rounds   : {cfg['n_rounds']}   "
          f"Local epochs/round : {cfg['local_epochs']}")
    print(f"  IDs : {cfg['n_ids']}   "
          f"Test ID ratio : {cfg['k_test']*100:.0f}%   "
          f"Gallery ratio : {cfg['gallery_ratio']*100:.0f}%")
    print(f"  FFT Aug  : {cfg['use_fft_aug']}   M={cfg['M']}   beta={cfg['fft_beta']}")
    print(f"{'='*62}\n")

    # ── Step 0a: build data splits ────────────────────────────────────────
    print("Building federated data splits …")
    (client_data, gallery_samples, probe_samples,
     test_label_map, spectra) = build_federated_splits(
        cfg["data_root"], cfg["n_ids"], cfg["k_test"],
        cfg["gallery_ratio"], seed=seed)

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
            num_classes   = cd["num_classes"],   # per-client (all equal to min_ids)
            cfg           = cfg,
            device        = device,
        ))

    # ── results file (tab-separated, one row per round) ───────────────────
    results_path = os.path.join(base_dir, "results.txt")
    client_header = "\t".join(
        f"Client{i}_EER(%)\tClient{i}_Rank1(%)" for i in range(n_clients))
    with open(results_path, "w") as f:
        f.write(f"Round\tGlobal_EER(%)\tGlobal_Rank1(%)\t{client_header}\n")

    # ── style template extraction and style bank creation ─────────────────
    if cfg["use_fft_aug"]:
        print("\nExtracting style templates from all clients …")
        style_bank = {
            client.client_id: client.extract_style_templates()
            for client in clients
        }
        total = sum(len(v) for v in style_bank.values())
        print(f"  Style bank ready — {total} templates "
              f"across {len(style_bank)} clients\n")
    else:
        style_bank = {}
        print("\nFFT augmentation disabled — using original samples only.\n")

    # ── Round 0: random init evaluation ──────────────────────────────────
    print("\n--- Round 0 (random init) ---")
    g_eer_0, g_rank1_0 = server.evaluate()
    print(f"  [Global init]  EER={g_eer_0*100:.4f}%  Rank-1={g_rank1_0:.2f}%")
    with open(results_path, "a") as f:
        f.write(f"0\t{g_eer_0*100:.4f}\t{g_rank1_0:.2f}\t"
                + "\t".join("-1\t-1" for _ in range(n_clients)) + "\n")

    # ── FL rounds ─────────────────────────────────────────────────────────
    g_eer, g_rank1 = g_eer_0, g_rank1_0   # FIX: ensures defined if n_rounds=0

    for rnd in range(1, cfg["n_rounds"] + 1):
        t_start        = time.time()
        global_weights = server.get_global_weights()
        client_weights = []
        client_metrics = []

        # ── Step 1: local training ────────────────────────────────────────
        for client in clients:
            # load global backbone weights (ArcFace head kept local)
            client.set_weights(global_weights)
            loss, acc = client.local_train(cfg["local_epochs"], style_bank, cfg["M"])
            c_eer, c_rank1 = evaluate_model(
                client.model,
                server.gallery_loader, server.probe_loader,
                device)
            # return backbone weights only (ArcFace head excluded)
            client_weights.append(client.get_weights())
            client_metrics.append({
                "client_id" : client.client_id,
                "spectrum"  : client.spectrum,
                "train_loss": round(loss, 6),
                "train_acc" : round(acc, 3),
                "eer"       : round(c_eer, 6),
                "rank1"     : round(c_rank1, 3),
            })

        # ── Step 2: FedAvg aggregation (backbone only) + global evaluation
        server.aggregate(client_weights)
        g_eer, g_rank1 = server.evaluate()
        elapsed = time.time() - t_start

        # console log
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Round {rnd:04d}/{cfg['n_rounds']} | "
              f"Global EER={g_eer*100:.4f}%  Rank-1={g_rank1:.2f}%  "
              f"({elapsed:.1f}s)")
        for cm in client_metrics:
            print(f"  Client {cm['client_id']} [{cm['spectrum']:>6}] | "
                  f"loss={cm['train_loss']:.4f}  acc={cm['train_acc']:.1f}%  "
                  f"EER={cm['eer']*100:.3f}%  R1={cm['rank1']:.1f}%")

        # append to results file
        client_cols = "\t".join(
            f"{cm['eer']*100:.4f}\t{cm['rank1']:.2f}"
            for cm in client_metrics)
        with open(results_path, "a") as f:
            f.write(f"{rnd}\t{g_eer*100:.4f}\t{g_rank1:.2f}\t{client_cols}\n")

    # ── Final reporting ───────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  FL COMPLETE — {cfg['n_rounds']} rounds")
    print(f"  Final Global EER   : {g_eer*100:.4f}%")
    print(f"  Final Global Rank-1: {g_rank1:.2f}%")
    print(f"  Results saved to   : {results_path}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
