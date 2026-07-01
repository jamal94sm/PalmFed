# ==============================================================
#  datasets.py — dataset classes and federated data partitioning
# ==============================================================
# SENTINEL VALUE: domain_id == -1 means "base FC only, no expert update"
# Used for FFT-augmented samples so experts only see real own-domain data.
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
    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = T.Compose([
            T.Resize((img_side, img_side)),
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx][0], self.samples[idx][1]
        return self.transform(Image.open(path).convert("L")), label
    def get_paths(self):
        return [s[0] for s in self.samples]
    def get_domain_ids(self):
        if len(self.samples[0]) >= 3:
            return [s[2] for s in self.samples]
        return None


class AugmentedDataset(Dataset):
    """
    Training dataset with standard spatial/photometric augmentation.
    Returns ([orig, aug], label, client_id).
    domain_id == client_id for all samples (real own-domain data).
    """
    def __init__(self, samples, img_side=128, grayscale=True, client_id=0):
        self.samples   = samples
        self.grayscale = grayscale
        self.client_id = client_id

        aug_transforms = [
            T.Resize((img_side, img_side)),
            T.RandomChoice([
                T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                T.RandomResizedCrop(img_side, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                T.RandomPerspective(distortion_scale=0.15, p=1.0),
                T.RandomChoice([
                    T.RandomRotation(10, expand=False, center=(int(0.5*img_side), 0)),
                    T.RandomRotation(10, expand=False, center=(0, int(0.5*img_side))),
                ]),
            ]),
        ]
        if grayscale:
            norm = T.Compose([T.ToTensor(), NormSingleROI(outchannels=1)])
        else:
            norm = T.Compose([T.ToTensor(),
                              T.Normalize(mean=[0.485, 0.456, 0.406],
                                          std =[0.229, 0.224, 0.225])])
        self.transform_orig = T.Compose([T.Resize((img_side, img_side))] + [norm])
        self.transform_aug  = T.Compose(aug_transforms + [norm])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx][0], self.samples[idx][1]
        mode = "L" if self.grayscale else "RGB"
        img  = Image.open(path).convert(mode)
        # domain_id = client_id: real own-domain data → routes to own expert
        return ([self.transform_orig(img), self.transform_aug(img)],
                label,
                self.client_id)


class PairedDataset(Dataset):
    """Paired same-class dataset for CCNet contrastive training."""
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
                T.RandomRotation(10, expand=False, center=(int(0.5*img_side), 0)),
                T.RandomRotation(10, expand=False, center=(0, int(0.5*img_side))),
            ]),
        ])
        self.to_tensor = T.Compose([
            T.Resize((img_side, img_side)), T.ToTensor(), NormSingleROI(outchannels=1)])
        self.to_tensor_fft = T.Compose([T.ToTensor(), NormSingleROI(outchannels=1)])

    def __len__(self): return len(self.samples)

    def _load_np(self, path):
        img = Image.open(path).convert("L").resize(
            (self.img_side, self.img_side), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _augment_image(self, path):
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
    Training dataset with FFT style augmentation.

    Asymmetric expert routing via domain_id sentinel
    ─────────────────────────────────────────────────
    aug_idx == 0  (real own-domain sample):
      domain_id = client_id   → routes to expert[client_id] + base
      This is the only signal that trains domain-specific experts.

    aug_idx >= 1  (FFT-styled synthetic sample):
      domain_id = -1  (SENTINEL)   → base FC only, NO expert update
      FFT styling approximates the target domain's amplitude but retains
      the source domain's identity/texture. Routing these to an expert
      would contaminate it with cross-domain noise. The base FC still
      receives their gradient and learns domain-invariant features.

    This replaces the previous scheme where FFT samples routed to the
    donor's expert[donor_id], which was the primary source of contamination.
    """

    EXPERT_SKIP = -1   # sentinel: skip expert branch, base FC only

    def __init__(self, samples, style_bank, client_id, M, beta, img_side,
                 grayscale=True, mean_bank=None,
                 prefer_distant=True, use_mean_template=False,
                 deterministic_donors=False):
        self.samples              = samples
        self.style_bank           = style_bank
        self.client_id            = client_id
        self.M                    = M
        self.beta                 = beta
        self.img_side             = img_side
        self.grayscale            = grayscale
        self.other_ids            = [cid for cid in style_bank if cid != client_id]
        self.mean_bank            = mean_bank
        self.prefer_distant       = prefer_distant
        self.use_mean_template    = use_mean_template
        self.deterministic_donors = deterministic_donors

        self.donor_order = self._rank_donors()

        self.spatial_aug = T.RandomChoice([
            T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
            T.RandomResizedCrop(img_side, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            T.RandomPerspective(distortion_scale=0.15, p=1.0),
            T.RandomChoice([
                T.RandomRotation(10, expand=False, center=(int(0.5 * img_side), 0)),
                T.RandomRotation(10, expand=False, center=(0, int(0.5 * img_side))),
            ]),
        ])

        if grayscale:
            norm_base = T.Compose([T.ToTensor(), NormSingleROI(outchannels=1)])
        else:
            norm_base = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.norm_orig = T.Compose([T.Resize((img_side, img_side)), norm_base])
        self.norm_aug  = norm_base

    def __len__(self):
        return len(self.samples) * self.M

    def _load_np(self, path):
        mode = "L" if self.grayscale else "RGB"
        img  = Image.open(path).convert(mode).resize(
            (self.img_side, self.img_side), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _to_orig_tensor(self, path):
        mode = "L" if self.grayscale else "RGB"
        return self.norm_orig(Image.open(path).convert(mode))

    def _to_aug_tensor(self, img_np):
        mode = "L" if self.grayscale else "RGB"
        pil  = Image.fromarray((img_np * 255).astype(np.uint8), mode=mode)
        return self.norm_aug(self.spatial_aug(pil))

    def _rank_donors(self):
        if not self.mean_bank or self.client_id not in self.mean_bank:
            return list(self.other_ids)
        own_mean = self.mean_bank[self.client_id]
        H, W     = own_mean.shape[:2]
        ch, cw   = H // 2, W // 2
        ph, pw   = H // 8, W // 8
        def _lf_patch(arr):
            return arr[ch-ph:ch+ph, cw-pw:cw+pw].flatten()
        own_lf    = _lf_patch(own_mean)
        distances = {cid: float(np.linalg.norm(
                         own_lf - _lf_patch(self.mean_bank[cid])))
                     for cid in self.other_ids if cid in self.mean_bank}
        return sorted(self.other_ids,
                      key=lambda c: distances.get(c, 0.0),
                      reverse=self.prefer_distant)

    def __getitem__(self, idx):
        sample_idx = idx // self.M
        aug_idx    = idx  % self.M

        path, label = self.samples[sample_idx][0], self.samples[sample_idx][1]
        img_np = self._load_np(path)

        orig = self._to_aug_tensor(img_np)

        if aug_idx == 0 or not self.other_ids:
            # Real own-domain sample → domain_id = client_id → updates expert[k] + base
            return [orig, self._to_aug_tensor(img_np)], label, self.client_id

        # FFT-styled synthetic sample → domain_id = SENTINEL (-1) → base FC only
        # Expert branch is completely skipped for this sample.
        # The base FC still receives gradient and learns cross-domain invariance.
        if self.deterministic_donors:
            rand_client = self.donor_order[(aug_idx - 1) % len(self.donor_order)]
        elif self.donor_order and self.mean_bank:
            rand_client = self.donor_order[0]
        else:
            rand_client = random.choice(self.other_ids)

        if self.use_mean_template and self.mean_bank \
                and rand_client in self.mean_bank:
            rand_template = self.mean_bank[rand_client]
        else:
            rand_template = random.choice(self.style_bank[rand_client])

        img_syn = apply_style_template(img_np, rand_template, self.beta)
        # Sentinel -1: expert branch skipped in MoEFC.forward()
        return [orig, self._to_aug_tensor(img_syn)], label, self.EXPERT_SKIP


# ══════════════════════════════════════════════════════════════
#  DATA LOADING — CASIA-MS
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root):
    data     = defaultdict(lambda: defaultdict(list))
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for fname in sorted(os.listdir(data_root)):
        if os.path.splitext(fname)[1].lower() not in img_exts:
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4:
            continue
        identity = f"{parts[0]}_{parts[1]}"
        spectrum = parts[2]
        data[spectrum][identity].append(os.path.join(data_root, fname))
    return data


def build_federated_splits(data_root, n_ids, k_test, gallery_ratio, seed=42):
    rng  = random.Random(seed)
    data = parse_casia_ms(data_root)
    spectra   = sorted(data.keys())
    n_clients = len(spectra)
    print(f"  Spectra found ({n_clients}): {spectra}")

    common_ids = set(data[spectra[0]].keys())
    for sp in spectra[1:]:
        common_ids &= set(data[sp].keys())
    common_ids = sorted(common_ids)
    print(f"  Identities common to all spectra: {len(common_ids)}")

    if len(common_ids) < n_ids:
        raise ValueError(f"Requested {n_ids} IDs but only {len(common_ids)} present.")

    selected_ids   = sorted(rng.sample(common_ids, n_ids))
    rng.shuffle(selected_ids)
    n_test_ids     = max(1, round(n_ids * k_test))
    test_ids       = sorted(selected_ids[:n_test_ids])
    train_ids      = sorted(selected_ids[n_test_ids:])
    test_label_map = {ident: i for i, ident in enumerate(test_ids)}
    print(f"  Total train IDs: {len(train_ids)}  |  Test IDs: {len(test_ids)}")

    rng.shuffle(train_ids)
    client_id_splits = [[] for _ in range(n_clients)]
    for i, ident in enumerate(train_ids):
        client_id_splits[i % n_clients].append(ident)

    min_ids = min(len(ids) for ids in client_id_splits)
    client_id_splits = [sorted(ids[:min_ids]) for ids in client_id_splits]
    n_dropped = len(train_ids) - min_ids * n_clients
    print(f"  IDs per client : {min_ids}  (dropped {n_dropped} to equalise)")

    client_data = []
    for i, sp in enumerate(spectra):
        c_ids       = client_id_splits[i]
        c_label_map = {ident: j for j, ident in enumerate(c_ids)}
        local_train = [(p, c_label_map[ident])
                       for ident in c_ids for p in data[sp][ident]]
        client_data.append({
            "spectrum"      : sp,
            "train_samples" : local_train,
            "label_map"     : c_label_map,
            "num_classes"   : min_ids,
        })
        print(f"    Client {i} [{sp:>6}]  train IDs={min_ids}  samples={len(local_train)}")

    gallery_samples, probe_samples = [], []
    for sp_idx, sp in enumerate(spectra):
        for ident in test_ids:
            paths = list(data[sp][ident])
            rng.shuffle(paths)
            label = test_label_map[ident]
            n_gal = max(1, round(len(paths) * gallery_ratio))
            for p in paths[:n_gal]:
                gallery_samples.append((p, label, sp_idx))
            for p in paths[n_gal:]:
                probe_samples.append((p, label, sp_idx))

    print(f"  Gallery: {len(gallery_samples)}  |  Probe: {len(probe_samples)}")
    return (client_data, gallery_samples, probe_samples, test_label_map, spectra)


# ══════════════════════════════════════════════════════════════
#  DATA LOADING — XJTU
# ══════════════════════════════════════════════════════════════

def parse_xjtu_domains(data_root, seed=42):
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
        print(f"  [XJTU] {domain[0]}/{domain[1]:6s}  IDs={n_ids}  images={n_imgs}")
    return data


def build_federated_splits_xjtu(data_root, n_ids, k_test, gallery_ratio, seed=42):
    rng  = random.Random(seed)
    data = parse_xjtu_domains(data_root, seed=seed)
    domains       = XJTU_VARIATIONS
    n_clients     = len(domains)
    domain_labels = [f"{d}/{c}" for d, c in domains]

    common_ids = set(data[domains[0]].keys())
    for dom in domains[1:]:
        common_ids &= set(data[dom].keys())
    common_ids = sorted(common_ids)
    print(f"  Identities common to all {n_clients} domains: {len(common_ids)}")

    if len(common_ids) < n_ids:
        raise ValueError(f"Requested {n_ids} IDs but only {len(common_ids)} present.")

    selected_ids   = sorted(rng.sample(common_ids, n_ids))
    rng.shuffle(selected_ids)
    n_test_ids     = max(1, round(n_ids * k_test))
    test_ids       = sorted(selected_ids[:n_test_ids])
    train_ids      = sorted(selected_ids[n_test_ids:])
    test_label_map = {ident: i for i, ident in enumerate(test_ids)}
    print(f"  Total train IDs: {len(train_ids)}  |  Test IDs: {len(test_ids)}")

    rng.shuffle(train_ids)
    client_id_splits = [[] for _ in range(n_clients)]
    for i, ident in enumerate(train_ids):
        client_id_splits[i % n_clients].append(ident)

    min_ids = min(len(ids) for ids in client_id_splits)
    client_id_splits = [sorted(ids[:min_ids]) for ids in client_id_splits]
    n_dropped = len(train_ids) - min_ids * n_clients
    print(f"  IDs per client : {min_ids}  (dropped {n_dropped} to equalise)")

    client_data = []
    for i, dom in enumerate(domains):
        c_ids       = client_id_splits[i]
        c_label_map = {ident: j for j, ident in enumerate(c_ids)}
        local_train = [(p, c_label_map[ident])
                       for ident in c_ids for p in data[dom][ident]]
        client_data.append({
            "spectrum"      : domain_labels[i],
            "domain"        : dom,
            "train_samples" : local_train,
            "label_map"     : c_label_map,
            "num_classes"   : min_ids,
        })
        print(f"    Client {i} [{domain_labels[i]:>14}]  "
              f"train IDs={min_ids}  samples={len(local_train)}")

    gallery_samples, probe_samples = [], []
    for dom_idx, dom in enumerate(domains):
        for ident in test_ids:
            paths = list(data[dom][ident])
            rng.shuffle(paths)
            label = test_label_map[ident]
            n_gal = max(1, round(len(paths) * gallery_ratio))
            for p in paths[:n_gal]:
                gallery_samples.append((p, label, dom_idx))
            for p in paths[n_gal:]:
                probe_samples.append((p, label, dom_idx))

    print(f"  Gallery: {len(gallery_samples)}  |  Probe: {len(probe_samples)}")
    return (client_data, gallery_samples, probe_samples, test_label_map, domain_labels)


# ══════════════════════════════════════════════════════════════
#  DISPATCHER
# ══════════════════════════════════════════════════════════════

def get_federated_splits(cfg, seed=42):
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
        raise ValueError(f"Unknown dataset: '{cfg['dataset']}'. Choose 'casiams' or 'xjtu'.")


# ══════════════════════════════════════════════════════════════
#  DINOV2 EVAL DATASET
# ══════════════════════════════════════════════════════════════

def _dino_eval_transform(img_side=224):
    return T.Compose([
        T.Resize((img_side, img_side)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class EvalDatasetDINO(Dataset):
    def __init__(self, samples, img_side=224):
        self.samples   = samples
        self.transform = _dino_eval_transform(img_side)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx][0], self.samples[idx][1]
        return self.transform(Image.open(path).convert("RGB")), label



# ══════════════════════════════════════════════════════════════
#  CLOSED-SET SPLITS
#  All IDs are train IDs. Per client: hold out 20% of samples.
#  Local test = held-out per client. Global test = union of all.
# ══════════════════════════════════════════════════════════════

def build_federated_splits_closed_set(data_root, n_ids, gallery_ratio,
                                       sample_holdout=0.20, seed=42):
    """
    Closed-set, non-shared-ID, same-domain split for CASIA-MS.

    Returns 6-tuple:
      client_data    : list[{spectrum, train_samples, local_test_gal,
                             local_test_prb, label_map, num_classes}]
      gallery_samples: global gallery (union of local held-out, split per ID)
      probe_samples  : global probe
      test_label_map : {identity: int}
      spectra        : list of spectrum strings
    """
    rng  = random.Random(seed)
    data = parse_casia_ms(data_root)

    spectra   = sorted(data.keys())
    n_clients = len(spectra)

    common_ids = set(data[spectra[0]].keys())
    for sp in spectra[1:]:
        common_ids &= set(data[sp].keys())
    common_ids = sorted(common_ids)

    if len(common_ids) < n_ids:
        raise ValueError(f"Requested {n_ids} IDs but only {len(common_ids)} common.")

    selected_ids = sorted(rng.sample(common_ids, n_ids))
    rng.shuffle(selected_ids)

    # ALL IDs are train IDs (closed-set)
    # Global label map for test set
    test_label_map = {ident: i for i, ident in enumerate(sorted(selected_ids))}

    # Round-robin partition (non-shared)
    client_id_splits = [[] for _ in range(n_clients)]
    for i, ident in enumerate(selected_ids):
        client_id_splits[i % n_clients].append(ident)

    min_ids = min(len(ids) for ids in client_id_splits)
    client_id_splits = [sorted(ids[:min_ids]) for ids in client_id_splits]

    print(f"  [Closed-set] {n_ids} IDs, {min_ids} per client, "
          f"holdout={sample_holdout:.0%}")

    # Build per-client: train + local test (held-out samples)
    client_data = []
    all_held_out = []  # (path, global_label, spectrum_idx)

    for ci, sp in enumerate(spectra):
        c_ids = client_id_splits[ci]
        c_label_map = {ident: j for j, ident in enumerate(c_ids)}

        # Gather all samples for this client
        all_samples = []
        for ident in c_ids:
            for p in data[sp][ident]:
                all_samples.append((p, c_label_map[ident], ident))

        # Hold out per identity (ensures each ID has gallery+probe)
        train_samples = []
        held_out = []
        by_id = defaultdict(list)
        for s in all_samples:
            by_id[s[2]].append(s)

        for ident, id_samples in by_id.items():
            rng.shuffle(id_samples)
            n_hold = max(2, int(len(id_samples) * sample_holdout))
            n_hold = min(n_hold, len(id_samples) - 1)  # keep at least 1 for train
            held_out.extend(id_samples[:n_hold])
            train_samples.extend(id_samples[n_hold:])

        # Local test: split held-out 50/50 gallery/probe per ID
        local_by_id = defaultdict(list)
        for p, local_label, ident in held_out:
            local_by_id[local_label].append((p, local_label))

        local_gal, local_prb = [], []
        for label, samples in local_by_id.items():
            rng.shuffle(samples)
            n_gal = max(1, len(samples) // 2)
            local_gal.extend(samples[:n_gal])
            local_prb.extend(samples[n_gal:])

        # Track held-out for global test set
        for p, _, ident in held_out:
            all_held_out.append((p, test_label_map[ident], ci))

        client_data.append({
            "spectrum"       : sp,
            "train_samples"  : [(p, l) for p, l, _ in train_samples],
            "local_test_gal" : local_gal,
            "local_test_prb" : local_prb,
            "label_map"      : c_label_map,
            "num_classes"    : min_ids,
        })
        print(f"    Client {ci} [{sp:>6}]  train={len(train_samples)}  "
              f"held_out={len(held_out)}  "
              f"local_gal={len(local_gal)}  local_prb={len(local_prb)}")

    # Global test: gallery/probe from all held-out samples
    global_by_id = defaultdict(list)
    for s in all_held_out:
        global_by_id[s[1]].append(s)

    gallery_samples, probe_samples = [], []
    for label, samples in global_by_id.items():
        rng.shuffle(samples)
        n_gal = max(1, int(len(samples) * gallery_ratio))
        gallery_samples.extend(samples[:n_gal])
        probe_samples.extend(samples[n_gal:])

    print(f"  Global test: Gal={len(gallery_samples)} Prb={len(probe_samples)}")
    return (client_data, gallery_samples, probe_samples,
            test_label_map, spectra)


def build_federated_splits_closed_set_xjtu(data_root, n_ids, gallery_ratio,
                                            sample_holdout=0.20, seed=42):
    """Closed-set split for XJTU dataset."""
    rng  = random.Random(seed)
    data = parse_xjtu_domains(data_root, seed=seed)

    domains       = XJTU_VARIATIONS
    n_clients     = len(domains)
    domain_labels = [f"{d}/{c}" for d, c in domains]

    common_ids = set(data[domains[0]].keys())
    for dom in domains[1:]:
        common_ids &= set(data[dom].keys())
    common_ids = sorted(common_ids)

    if len(common_ids) < n_ids:
        raise ValueError(f"Requested {n_ids} IDs but only {len(common_ids)} common.")

    selected_ids = sorted(rng.sample(common_ids, n_ids))
    rng.shuffle(selected_ids)
    test_label_map = {ident: i for i, ident in enumerate(sorted(selected_ids))}

    client_id_splits = [[] for _ in range(n_clients)]
    for i, ident in enumerate(selected_ids):
        client_id_splits[i % n_clients].append(ident)

    min_ids = min(len(ids) for ids in client_id_splits)
    client_id_splits = [sorted(ids[:min_ids]) for ids in client_id_splits]

    print(f"  [Closed-set XJTU] {n_ids} IDs, {min_ids} per client")

    client_data = []
    all_held_out = []

    for ci, dom in enumerate(domains):
        c_ids = client_id_splits[ci]
        c_label_map = {ident: j for j, ident in enumerate(c_ids)}

        all_samples = []
        for ident in c_ids:
            for p in data[dom][ident]:
                all_samples.append((p, c_label_map[ident], ident))

        train_samples = []
        held_out = []
        by_id = defaultdict(list)
        for s in all_samples:
            by_id[s[2]].append(s)

        for ident, id_samples in by_id.items():
            rng.shuffle(id_samples)
            n_hold = max(2, int(len(id_samples) * sample_holdout))
            n_hold = min(n_hold, len(id_samples) - 1)
            held_out.extend(id_samples[:n_hold])
            train_samples.extend(id_samples[n_hold:])

        local_by_id = defaultdict(list)
        for p, local_label, ident in held_out:
            local_by_id[local_label].append((p, local_label))

        local_gal, local_prb = [], []
        for label, samples in local_by_id.items():
            rng.shuffle(samples)
            n_gal = max(1, len(samples) // 2)
            local_gal.extend(samples[:n_gal])
            local_prb.extend(samples[n_gal:])

        for p, _, ident in held_out:
            all_held_out.append((p, test_label_map[ident], ci))

        client_data.append({
            "spectrum"       : domain_labels[ci],
            "domain"         : dom,
            "train_samples"  : [(p, l) for p, l, _ in train_samples],
            "local_test_gal" : local_gal,
            "local_test_prb" : local_prb,
            "label_map"      : c_label_map,
            "num_classes"    : min_ids,
        })
        print(f"    Client {ci} [{domain_labels[ci]:>14}]  "
              f"train={len(train_samples)}  held={len(held_out)}")

    global_by_id = defaultdict(list)
    for s in all_held_out:
        global_by_id[s[1]].append(s)

    gallery_samples, probe_samples = [], []
    for label, samples in global_by_id.items():
        rng.shuffle(samples)
        n_gal = max(1, int(len(samples) * gallery_ratio))
        gallery_samples.extend(samples[:n_gal])
        probe_samples.extend(samples[n_gal:])

    print(f"  Global test: Gal={len(gallery_samples)} Prb={len(probe_samples)}")
    return (client_data, gallery_samples, probe_samples,
            test_label_map, domain_labels)


# ══════════════════════════════════════════════════════════════
#  UNIFIED DISPATCHER (replaces original get_federated_splits)
# ══════════════════════════════════════════════════════════════

def get_federated_splits(cfg, seed=42):
    dataset  = cfg["dataset"].strip().lower()
    protocol = cfg.get("eval_protocol", "open_set").strip().lower()

    if dataset == "casiams":
        if protocol == "closed_set":
            return build_federated_splits_closed_set(
                cfg["data_root"], cfg["n_ids"], cfg["gallery_ratio"],
                cfg.get("closed_set_sample_ratio", 0.20), seed=seed)
        else:
            return build_federated_splits(
                cfg["data_root"], cfg["n_ids"], cfg["k_test"],
                cfg["gallery_ratio"], seed=seed)

    elif dataset == "xjtu":
        if protocol == "closed_set":
            return build_federated_splits_closed_set_xjtu(
                cfg["xjtu_data_root"], cfg["n_ids"], cfg["gallery_ratio"],
                cfg.get("closed_set_sample_ratio", 0.20), seed=seed)
        else:
            return build_federated_splits_xjtu(
                cfg["xjtu_data_root"], cfg["n_ids"], cfg["k_test"],
                cfg["gallery_ratio"], seed=seed)

    else:
        raise ValueError(f"Unknown dataset: '{dataset}'")
