# ==============================================================
#  datasets.py — dataset classes and federated data partitioning
# ==============================================================
#
#  Normalisation:
#    NormSingleROI
#
#  Dataset classes:
#    PalmDataset          — gallery/probe evaluation (no augmentation)
#    AugmentedDataset     — standard spatial aug (CompNet baseline)
#    PairedDataset        — paired same-class images (CCNet training)
#    FFTAugmentedDataset  — FFT style swap + spatial aug (CompNet FFT run)
#
#  Data loading:
#    parse_casia_ms
#    build_federated_splits
# ==============================================================

import os
import random
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils import extract_style_template, apply_style_template


# ══════════════════════════════════════════════════════════════
#  NORMALISATION
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    """
    Per-image normalisation over foreground pixels only (value > 0).
    Background zero-padding is excluded so it does not distort the
    mean/std of the actual palm ROI region.
    """
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
#  DATASET CLASSES
# ══════════════════════════════════════════════════════════════

class PalmDataset(Dataset):
    """
    Plain dataset for gallery/probe evaluation — no augmentation.
    Used by FLServer gallery/probe loaders and evaluate_model().
    """
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


class AugmentedDataset(Dataset):
    """
    Training dataset with standard spatial/photometric augmentation.
    Used as the baseline (use_fft_aug=False) for CompNet training so
    that the baseline is not disadvantaged by unaugmented data.
    Returns a single (augmented_image, label) per sample.
    """
    def __init__(self, samples, img_side=128):
        self.samples = samples
        self.transform = T.Compose([
            T.Resize((img_side, img_side)),
            T.RandomChoice([
                T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                T.RandomResizedCrop(img_side, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
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

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), label


class PairedDataset(Dataset):
    """
    Paired same-class dataset for CCNet contrastive training.
    Returns ([img1, img2], label) where img2 is a different sample
    of the same identity — required to form the two views for SupConLoss.
    Spatial augmentation is applied to both images independently.
    """
    def __init__(self, samples, img_side=128):
        self.samples     = samples
        self.label2idxs  = defaultdict(list)
        for i, (_, lab) in enumerate(samples):
            self.label2idxs[lab].append(i)

        self.transform = T.Compose([
            T.Resize((img_side, img_side)),
            T.RandomChoice([
                T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                T.RandomResizedCrop(img_side, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
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

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path1, label = self.samples[idx]
        # pick a different sample from the same identity
        idxs = self.label2idxs[label]
        idx2 = idx
        while idx2 == idx and len(idxs) > 1:
            idx2 = random.choice(idxs)
        path2, _ = self.samples[idx2]
        img1 = self.transform(Image.open(path1).convert("L"))
        img2 = self.transform(Image.open(path2).convert("L"))
        return [img1, img2], label


class FFTAugmentedDataset(Dataset):
    """
    Training dataset with FFT style augmentation + spatial augmentation.
    Used for CompNet when use_fft_aug=True.

    For each original sample x_i, M-1 synthetic copies are produced by
    replacing its low-frequency amplitude with a randomly chosen template
    from a randomly chosen other client's style bank.

    Index layout: indices [i*M .. i*M+M-1] map to sample i.
      aug_idx == 0  → original   + spatial augmentation
      aug_idx >= 1  → FFT styled + spatial augmentation

    Notes
    ─────
    • T.Resize absent from self.to_tensor — _load_np already resizes.
    • Spatial augmentation applied to ALL samples for fair comparison.
    • Foreground mask saved before FFT and re-applied after reconstruction
      so NormSingleROI computes statistics on palm ROI pixels only
      (inverse FFT leaves small non-zero residuals in background).
    """

    def __init__(self, samples, style_bank, client_id, M, beta, img_side):
        self.samples    = samples
        self.style_bank = style_bank   # {client_id: [amp_array, ...]}
        self.client_id  = client_id
        self.M          = M
        self.beta       = beta
        self.img_side   = img_side
        # a client never borrows its own style (no domain shift otherwise)
        self.other_ids  = [cid for cid in style_bank if cid != client_id]

        # spatial augmentation — applied to all samples after FFT processing
        self.spatial_aug = T.RandomChoice([
            T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
            T.RandomResizedCrop(img_side, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            T.RandomPerspective(distortion_scale=0.15, p=1.0),
            T.RandomChoice([
                T.RandomRotation(10, expand=False,
                                 center=(int(0.5 * img_side), 0)),
                T.RandomRotation(10, expand=False,
                                 center=(0, int(0.5 * img_side))),
            ]),
        ])

        # no T.Resize — _load_np already resizes to img_side
        self.to_tensor = T.Compose([
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])

    def __len__(self):
        return len(self.samples) * self.M

    def _load_np(self, path):
        """Load, resize, return float32 in [0, 1]."""
        img = Image.open(path).convert("L").resize(
            (self.img_side, self.img_side), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _to_tensor(self, img_np):
        """numpy → spatially augmented PIL → normalised tensor."""
        pil = Image.fromarray((img_np * 255).astype(np.uint8), mode="L")
        pil = self.spatial_aug(pil)
        return self.to_tensor(pil)

    def __getitem__(self, idx):
        sample_idx = idx // self.M   # which original sample
        aug_idx    = idx  % self.M   # 0 = original, 1..M-1 = synthetic

        path, label = self.samples[sample_idx]
        img_np = self._load_np(path)

        if aug_idx == 0 or not self.other_ids:
            return self._to_tensor(img_np), label

        # save foreground mask before FFT
        fg_mask = img_np > 0

        rand_client   = random.choice(self.other_ids)
        rand_template = random.choice(self.style_bank[rand_client])
        img_syn       = apply_style_template(img_np, rand_template, self.beta)

        # re-apply foreground mask
        img_syn[~fg_mask] = 0.0
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
    data : dict  spectrum → identity → [path, ...]
    """
    data     = defaultdict(lambda: defaultdict(list))
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for fname in sorted(os.listdir(data_root)):
        if os.path.splitext(fname)[1].lower() not in img_exts:
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4:
            continue
        identity = f"{parts[0]}_{parts[1]}"   # e.g. "001_L"
        spectrum = parts[2]
        data[spectrum][identity].append(os.path.join(data_root, fname))
    return data


def build_federated_splits(data_root, n_ids, k_test, gallery_ratio, seed=42):
    """
    Build per-client training sets and a fixed shared gallery/probe test set.

    Protocol: Open-Set, Non-Shared-ID, Cross-Domain.
      - n_ids identities are sampled from those common to all spectra.
      - k_test fraction → test IDs; remainder → train IDs (disjoint).
      - Train IDs are partitioned round-robin across clients (no overlap).
      - All clients are trimmed to min partition size (equalised).
      - Test IDs across all spectra form the gallery/probe split.

    Returns
    -------
    client_data     : list of dicts {spectrum, train_samples, label_map, num_classes}
    gallery_samples : list of (path, label)
    probe_samples   : list of (path, label)
    test_label_map  : {identity: int}
    spectra         : list of spectrum strings
    """
    rng  = random.Random(seed)
    data = parse_casia_ms(data_root)

    spectra   = sorted(data.keys())
    n_clients = len(spectra)
    print(f"  Spectra found ({n_clients}): {spectra}")

    # identities present in ALL spectra
    common_ids = set(data[spectra[0]].keys())
    for sp in spectra[1:]:
        common_ids &= set(data[sp].keys())
    common_ids = sorted(common_ids)
    print(f"  Identities common to all spectra: {len(common_ids)}")

    if len(common_ids) < n_ids:
        raise ValueError(
            f"Requested {n_ids} IDs but only {len(common_ids)} "
            f"are present in all spectra.")

    # sample n_ids, shuffle, split into test / train ID sets
    selected_ids = sorted(rng.sample(common_ids, n_ids))
    rng.shuffle(selected_ids)
    n_test_ids     = max(1, round(n_ids * k_test))
    test_ids       = sorted(selected_ids[:n_test_ids])
    train_ids      = sorted(selected_ids[n_test_ids:])
    test_label_map = {ident: i for i, ident in enumerate(test_ids)}

    print(f"  Total train IDs: {len(train_ids)}  |  Test IDs: {len(test_ids)}")

    # round-robin partition — no two clients share a train ID
    rng.shuffle(train_ids)
    client_id_splits = [[] for _ in range(n_clients)]
    for i, ident in enumerate(train_ids):
        client_id_splits[i % n_clients].append(ident)

    # equalise by trimming to smallest partition
    min_ids = min(len(ids) for ids in client_id_splits)
    client_id_splits = [sorted(ids[:min_ids]) for ids in client_id_splits]
    n_dropped = len(train_ids) - min_ids * n_clients
    print(f"  IDs per client : {min_ids}  (dropped {n_dropped} to equalise)")

    # build per-client local train sets — each has its own label space
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

    # fixed global test set — split within each (spectrum, identity) pair
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
