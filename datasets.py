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
#  Data loading — CASIA-MS:
#    parse_casia_ms
#    build_federated_splits          (6 clients, one per spectrum)
#
#  Data loading — XJTU:
#    parse_xjtu_domains
#    build_federated_splits_xjtu     (4 clients, one per device+lighting)
#
#  Dispatcher:
#    get_federated_splits            (calls correct builder from cfg["dataset"])
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
from configs import XJTU_VARIATIONS


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
    Returns a single (augmented_image, label) per sample.

    Parameters
    ----------
    grayscale : bool
        True  → grayscale + NormSingleROI  (CompNet, CCNet)
        False → RGB + ImageNet norm        (DINOv2)
    """
    def __init__(self, samples, img_side=128, grayscale=True):
        self.samples   = samples
        self.grayscale = grayscale

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
            NormSingleROI(outchannels=1) if grayscale else
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mode = "L" if self.grayscale else "RGB"
        return self.transform(Image.open(path).convert(mode)), label


class PairedDataset(Dataset):
    """
    Paired same-class dataset for CCNet contrastive training.

    Returns ([img1, img2], label) where img2 is a different sample of the
    same identity — required to form the two views for SupConLoss.

    Augmentation modes (controlled by style_bank argument):
      style_bank empty → standard spatial augmentation on both views
      style_bank filled → FFT style swap + spatial aug on both views
                          (each view independently gets a random template
                           from a random other client)
    """
    def __init__(self, samples, img_side=128,
                 style_bank=None, client_id=None, beta=0.15):
        self.samples    = samples
        self.img_side   = img_side
        self.style_bank = style_bank or {}
        self.client_id  = client_id
        self.beta       = beta
        self.other_ids  = [cid for cid in self.style_bank
                           if cid != client_id] if self.style_bank else []
        self.use_fft    = bool(self.other_ids)

        self.label2idxs = defaultdict(list)
        for i, (_, lab) in enumerate(samples):
            self.label2idxs[lab].append(i)

        self.spatial_aug = T.RandomChoice([
            T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
            T.RandomResizedCrop(img_side, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            T.RandomPerspective(distortion_scale=0.15, p=1.0),
            T.RandomChoice([
                T.RandomRotation(10, expand=False,
                                 center=(int(0.5*img_side), 0)),
                T.RandomRotation(10, expand=False,
                                 center=(0, int(0.5*img_side))),
            ]),
        ])

        self.to_tensor = T.Compose([
            T.Resize((img_side, img_side)),
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])
        # no T.Resize — _load_np already resizes
        self.to_tensor_fft = T.Compose([
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])

    def __len__(self): return len(self.samples)

    def _load_np(self, path):
        """Load, resize, return grayscale float32 in [0, 1]."""
        img = Image.open(path).convert("L").resize(
            (self.img_side, self.img_side), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _augment_image(self, path):
        """
        Load one image and apply the configured augmentation.
        FFT mode: FFT style swap from random other client + spatial aug.
        Spatial mode: spatial augmentation only.
        """
        if self.use_fft:
            img_np        = self._load_np(path)
            rand_client   = random.choice(self.other_ids)
            rand_template = random.choice(self.style_bank[rand_client])
            img_np        = apply_style_template(img_np, rand_template, self.beta)
            pil = Image.fromarray((img_np * 255).astype(np.uint8), mode="L")
            pil = self.spatial_aug(pil)
            return self.to_tensor_fft(pil)
        else:
            img = Image.open(path).convert("L")
            img = self.spatial_aug(img)
            return self.to_tensor(img)

    def __getitem__(self, idx):
        path1, label = self.samples[idx]
        idxs = self.label2idxs[label]
        idx2 = idx
        while idx2 == idx and len(idxs) > 1:
            idx2 = random.choice(idxs)
        path2, _ = self.samples[idx2]
        img1 = self._augment_image(path1)
        img2 = self._augment_image(path2)
        return [img1, img2], label


class FFTAugmentedDataset(Dataset):
    """
    Training dataset with FFT style augmentation + spatial augmentation.
    Used for CompNet and DINOv2 when use_fft_aug=True.

    For each original sample x_i, M-1 synthetic copies are produced by
    replacing its low-frequency amplitude with a randomly chosen template
    from a randomly chosen other client's style bank.

    Index layout: indices [i*M .. i*M+M-1] map to sample i.
      aug_idx == 0  → original   + spatial augmentation
      aug_idx >= 1  → FFT styled + spatial augmentation

    Notes
    ─────
    • Grayscale (H, W) for CompNet/CCNet; RGB (H, W, 3) for DINOv2.
    • Foreground mask and per-channel logic live in apply_style_template.
    """

    def __init__(self, samples, style_bank, client_id, beta, img_side,
             grayscale=True, mean_bank=None,
             prefer_distant=True, use_mean_template=False):
        self.samples    = samples
        self.style_bank = style_bank
        self.client_id  = client_id
        self.beta       = beta
        self.img_side   = img_side
        self.grayscale  = grayscale
        self.other_ids  = [cid for cid in style_bank if cid != client_id]
        self.mean_bank         = mean_bank
        self.prefer_distant    = prefer_distant
        self.use_mean_template = use_mean_template
    
        # pre-compute donor order by L2 distance of low-frequency mean templates
        # done once at dataset construction — not repeated per sample
        self.donor_order = self._rank_donors()

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

        # no T.Resize — _load_np already resizes
        if grayscale:
            self.norm = T.Compose([T.ToTensor(), NormSingleROI(outchannels=1)])
        else:
            self.norm = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def _load_np(self, path):
        """Load and resize. Returns float32 in [0, 1].
        Grayscale (H, W) for CompNet/CCNet; RGB (H, W, 3) for DINOv2."""
        mode = "L" if self.grayscale else "RGB"
        img  = Image.open(path).convert(mode).resize(
            (self.img_side, self.img_side), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _to_tensor(self, img_np):
        """numpy array → PIL (correct mode) → spatial aug → normalised tensor."""
        if self.grayscale:
            pil = Image.fromarray((img_np * 255).astype(np.uint8), mode="L")
        else:
            pil = Image.fromarray((img_np * 255).astype(np.uint8), mode="RGB")
        pil = self.spatial_aug(pil)
        return self.norm(pil)

    def _rank_donors(self):
        """
        Rank other clients by L2 distance of their mean template to this
        client's mean template. Uses only the low-frequency region (centre
        patch of the fftshifted amplitude — same region the Gaussian mask
        targets) to measure domain distance.
    
        Returns sorted list of client IDs:
          prefer_distant=True  → most different first
          prefer_distant=False → most similar first
        Falls back to random order if mean_bank is unavailable.
        """
        if not self.mean_bank or self.client_id not in self.mean_bank:
            return list(self.other_ids)   # fallback: random order
    
        own_mean = self.mean_bank[self.client_id]
        H, W     = own_mean.shape[:2]
    
        # extract centre low-frequency patch (inner 25% of H and W)
        ch, cw   = H // 2, W // 2
        ph, pw   = H // 8, W // 8          # patch half-size
        def _lf_patch(arr):
            patch = arr[ch-ph:ch+ph, cw-pw:cw+pw]
            return patch.flatten()
    
        own_lf = _lf_patch(own_mean)
    
        distances = {}
        for cid in self.other_ids:
            if cid in self.mean_bank:
                other_lf     = _lf_patch(self.mean_bank[cid])
                distances[cid] = np.linalg.norm(own_lf - other_lf)
            else:
                distances[cid] = 0.0
    
        # sort: largest distance first if prefer_distant, else smallest first
        return sorted(self.other_ids,
                      key=lambda c: distances[c],
                      reverse=self.prefer_distant)
        
    def __getitem__(self, idx):
        sample_idx = idx // self.M
        aug_idx    = idx  % self.M
    
        path, label = self.samples[sample_idx]
        img_np = self._load_np(path)
    
        if aug_idx == 0 or not self.other_ids:
            return self._to_tensor(img_np), label
    
        # donor client selection
        if self.donor_order and self.mean_bank:
            # domain-aware: pre-ranked by L2 distance of mean low-freq templates
            rand_client = self.donor_order[0]
        else:
            # default: uniform random — original behaviour
            rand_client = random.choice(self.other_ids)
    
        # template selection from chosen donor
        if self.use_mean_template and self.mean_bank and rand_client in self.mean_bank:
            # domain-aware: use the donor's mean amplitude template
            rand_template = self.mean_bank[rand_client]
        else:
            # default: random sample from donor's bank — original behaviour
            rand_template = random.choice(self.style_bank[rand_client])
    
        img_syn = apply_style_template(img_np, rand_template, self.beta)
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

# ══════════════════════════════════════════════════════════════
#  XJTU DATA LOADING & PARTITIONING
# ══════════════════════════════════════════════════════════════

def parse_xjtu_domains(data_root, seed=42):
    """
    Scan data_root for XJTU-UP images organised by (device, condition) domain.

    Directory layout:
      data_root/
        {device}/
          {condition}/
            {L|R}_{subjectID}/
              *.jpg

    Domain keys: (device, condition) tuples from XJTU_VARIATIONS, e.g.
      ("iPhone", "Flash"), ("iPhone", "Nature"),
      ("huawei", "Flash"), ("huawei", "Nature")

    Identity key: the folder name, e.g. "L_001" or "R_002".

    Returns
    -------
    data : dict  (device, condition) → identity → [path, ...]
           Same structure as parse_casia_ms (spectrum → identity → paths)
           so build_federated_splits_xjtu mirrors build_federated_splits.
    """
    IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}
    data     = defaultdict(lambda: defaultdict(list))

    for device, condition in XJTU_VARIATIONS:
        var_dir = os.path.join(data_root, device, condition)
        if not os.path.isdir(var_dir):
            print(f"  [XJTU] WARNING: {var_dir} not found")
            continue
        for id_folder in sorted(os.listdir(var_dir)):
            id_dir = os.path.join(var_dir, id_folder)
            if not os.path.isdir(id_dir):
                continue
            parts = id_folder.split("_")
            if len(parts) < 2 or parts[0].upper() not in ("L", "R"):
                continue
            for fname in sorted(os.listdir(id_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                    continue
                data[(device, condition)][id_folder].append(
                    os.path.join(id_dir, fname))

    for domain in XJTU_VARIATIONS:
        n_ids  = len(data[domain])
        n_imgs = sum(len(v) for v in data[domain].values())
        print(f"  [XJTU] {domain[0]}/{domain[1]:6s}  "
              f"IDs={n_ids}  images={n_imgs}")
    return data


def build_federated_splits_xjtu(data_root, n_ids, k_test, gallery_ratio, seed=42):
    """
    Build per-client training sets and a fixed shared gallery/probe test set
    for the XJTU dataset.

    Protocol: Open-Set, Non-Shared-ID, Cross-Domain.
      - 4 clients, one per (device, condition) domain.
      - n_ids identities are sampled from those common to all 4 domains.
      - k_test fraction → test IDs; remainder → train IDs (fully disjoint).
      - Train IDs are partitioned round-robin across clients (no overlap).
      - All clients are trimmed to min partition size (equalised).
      - Test IDs across all 4 domains form the gallery/probe split.

    Returns
    -------
    client_data     : list of dicts {domain_label, train_samples,
                                     label_map, num_classes}
    gallery_samples : list of (path, label)
    probe_samples   : list of (path, label)
    test_label_map  : {identity: int}
    domain_labels   : list of "{device}/{condition}" strings (client names)
    """
    rng  = random.Random(seed)
    data = parse_xjtu_domains(data_root, seed=seed)

    domains       = XJTU_VARIATIONS          # 4 (device, condition) tuples
    n_clients     = len(domains)
    domain_labels = [f"{d}/{c}" for d, c in domains]

    # identities present in ALL 4 domains
    common_ids = set(data[domains[0]].keys())
    for dom in domains[1:]:
        common_ids &= set(data[dom].keys())
    common_ids = sorted(common_ids)
    print(f"  Identities common to all {n_clients} domains: {len(common_ids)}")

    if len(common_ids) < n_ids:
        raise ValueError(
            f"Requested {n_ids} IDs but only {len(common_ids)} "
            f"are present in all {n_clients} XJTU domains.")

    # sample n_ids, shuffle, split into test / train ID sets
    selected_ids   = sorted(rng.sample(common_ids, n_ids))
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
    for i, dom in enumerate(domains):
        c_ids       = client_id_splits[i]
        c_label_map = {ident: j for j, ident in enumerate(c_ids)}
        local_train = [
            (p, c_label_map[ident])
            for ident in c_ids
            for p in data[dom][ident]
        ]
        client_data.append({
            "spectrum"      : domain_labels[i],   # reuse "spectrum" key for compatibility
            "domain"        : dom,                # (device, condition) tuple
            "train_samples" : local_train,
            "label_map"     : c_label_map,
            "num_classes"   : min_ids,
        })
        print(f"    Client {i} [{domain_labels[i]:>14}]  "
              f"train IDs={min_ids}  samples={len(local_train)}")

    # fixed global test set — split within each (domain, identity) pair
    gallery_samples, probe_samples = [], []
    for dom in domains:
        for ident in test_ids:
            paths = list(data[dom][ident])
            rng.shuffle(paths)
            label = test_label_map[ident]
            n_gal = max(1, round(len(paths) * gallery_ratio))
            for p in paths[:n_gal]:
                gallery_samples.append((p, label))
            for p in paths[n_gal:]:
                probe_samples.append((p, label))

    print(f"  Gallery: {len(gallery_samples)}  |  Probe: {len(probe_samples)}")
    return (client_data, gallery_samples, probe_samples,
            test_label_map, domain_labels)


# ══════════════════════════════════════════════════════════════
#  DISPATCHER
# ══════════════════════════════════════════════════════════════

def get_federated_splits(cfg, seed=42):
    """
    Call the correct split builder based on cfg["dataset"].

    Returns the same 5-tuple for both datasets:
      (client_data, gallery_samples, probe_samples, test_label_map, domain_names)

    This uniform interface means main.py needs no dataset-specific branching
    beyond this single call.
    """
    dataset = cfg["dataset"].strip().lower()

    if dataset == "casiams":
        return build_federated_splits(
            cfg["data_root"], cfg["n_ids"], cfg["k_test"],
            cfg["gallery_ratio"], seed=seed)

    elif dataset == "xjtu":
        return build_federated_splits_xjtu(
            cfg["xjtu_data_root"], cfg["n_ids"], cfg["k_test"],
            cfg["gallery_ratio"], seed=seed)

    else:
        raise ValueError(
            f"Unknown dataset: '{cfg['dataset']}'. "
            f"Choose 'casiams' or 'xjtu'.")



# ══════════════════════════════════════════════════════════════
#  DINOV2 EVALUATION DATASET  (RGB + ImageNet normalisation)
# ══════════════════════════════════════════════════════════════
# DINOv2 training uses AugmentedDataset / FFTAugmentedDataset with
# grayscale=False — same augmentation policy as CompNet.  Only the
# gallery/probe evaluation loader needs a dedicated class because it
# must convert the grayscale palmprint to RGB and apply ImageNet norm.

def _dino_eval_transform(img_side=224):
    """Eval transform for DINOv2: resize + RGB + ImageNet norm."""
    return T.Compose([
        T.Resize((img_side, img_side)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])


class EvalDatasetDINO(Dataset):
    """
    Evaluation dataset for DINOv2 gallery/probe sets.
    No augmentation — resize + ImageNet norm only.
    Grayscale palmprint images are replicated to 3 channels via
    convert("RGB") to match the ImageNet-pretrained backbone's input.
    """
    def __init__(self, samples, img_side=224):
        self.samples   = samples
        self.transform = _dino_eval_transform(img_side)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label
