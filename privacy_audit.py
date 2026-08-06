# ==============================================================
# privacy_audit.py — Empirical privacy audit of the Style Bank's
# low-frequency amplitude descriptors (addresses Reviewer 1 & 2).
#
# Three experiments, all driven by the SAME masked-amplitude
# descriptor used for clustering (see clustering.py::crop_lowfreq),
# so results here directly characterize what is actually shared
# in the Style Bank at a given alpha.
#
#   Exp 1 — Identity classification / verification from amplitude
#           alone (masked amplitude only, no phase).
#   Exp 2 — Simple reconstruction attack: combine shared amplitude
#           with a phase the attacker DOES have access to (mean/
#           random phase), inverse-FFT, and measure re-identification
#           rate against the trained verifier.
#   Exp 3 — Membership inference: can an attacker tell whether a
#           given amplitude descriptor was in a client's training set?
#
# alpha is adjustable via --alpha (or a --alpha_sweep list) so the
# leakage-vs-alpha trade-off can be reported directly.
#
# Reuses: datasets.py (splits + PalmDataset), utils.py (extract_style_
# template, compute_eer, evaluate_split), configs.py (get_config),
# clustering.py (crop_lowfreq), model_fedpalm.py (compnet_fedpalm, for
# the full-image reference baseline only).
# ==============================================================

'''
# Single alpha, all three datasets
python privacy_audit.py --datasets casiams xjtu xpalm --alpha 0.15

# Sweep alpha to see the leakage/utility trade-off (recommended)
python privacy_audit.py --datasets casiams --alpha_sweep 0.05 0.10 0.15 0.25 0.40

# With a real trained verifier for a meaningful Exp 2 number
python privacy_audit.py --datasets casiams --alpha 0.15 --verifier_ckpt ./checkpoints/global_model.pt
'''

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from configs import get_config
from datasets import get_federated_splits, PalmDataset
from utils import extract_style_template, compute_eer, evaluate_split
from clustering import crop_lowfreq
from model_fedpalm import compnet_fedpalm


# ══════════════════════════════════════════════════════════════
# DESCRIPTOR EXTRACTION — shared across all 3 experiments
# ══════════════════════════════════════════════════════════════
def load_gray(path, img_side):
    img = Image.open(path).convert("L").resize(
        (img_side, img_side), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def full_fft(img_np):
    """Full complex FFT (fftshift'ed) — gives both magnitude and phase."""
    F_ = np.fft.fftshift(np.fft.fft2(img_np))
    return np.abs(F_), np.angle(F_)


def masked_amplitude_descriptor(img_np, alpha, scale="linear"):
    """
    The exact quantity shared in the Style Bank / used for clustering:
    low-frequency-cropped amplitude, flattened. Delegates to
    clustering.py::crop_lowfreq so this audit tests literally what
    gets transmitted, not a re-derived approximation of it.
    """
    amp, _ = full_fft(img_np)
    return crop_lowfreq(amp, alpha=alpha, scale=scale)


def masked_amplitude_image(img_np, alpha):
    """
    2D (non-flattened) masked amplitude, used for Exp 2 reconstruction:
    everything OUTSIDE the low-freq crop is zeroed, everything inside
    is kept at full precision — this is the most an attacker holding
    only the Style Bank crop could plausibly reconstruct amplitude-wise.
    """
    amp, _ = full_fft(img_np)
    H, W = amp.shape
    cy, cx = H // 2, W // 2
    ly, lx = int(alpha * H), int(alpha * W)
    masked = np.zeros_like(amp)
    masked[cy - ly:cy + ly + 1, cx - lx:cx + lx + 1] = \
        amp[cy - ly:cy + ly + 1, cx - lx:cx + lx + 1]
    return masked


# ══════════════════════════════════════════════════════════════
# SAMPLE COLLECTION — build (path, identity_label, client_id) list
# ══════════════════════════════════════════════════════════════
def collect_all_samples(client_data):
    """
    Flattens client_data into a single list of (path, global_identity,
    client_id) tuples. Uses GLOBAL identity = (client_id, local_label)
    since local labels are only unique within a client, not across —
    identity classification/verification here is scoped per-client,
    matching how the paper's own closed-set clients are structured.
    """
    samples = []
    for ci, cd in enumerate(client_data):
        for path, local_label in cd["train_samples"]:
            samples.append((path, local_label, ci))
    return samples


# ══════════════════════════════════════════════════════════════
# SMALL PROBE MODELS — deliberately simple (architecture-agnostic
# leakage test: a weak probe finding signal is the stronger finding)
# ══════════════════════════════════════════════════════════════
class MLPProbe(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def embed(self, x):
        for layer in self.net[:-1]:
            x = layer(x)
        return F.normalize(x, p=2, dim=1)


class DescriptorDataset(Dataset):
    def __init__(self, descriptors, labels):
        self.x = torch.tensor(np.stack(descriptors), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_probe(model, loader, device, epochs=30, lr=1e-3):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"      [probe] epoch {ep+1:3d}/{epochs} "
                  f"train_acc={100.0*correct/max(total,1):.2f}%")
    return model


@torch.no_grad()
def eval_probe_verification(model, loader, device):
    """1:1 verification EER from probe embeddings (mirrors utils.evaluate_split)."""
    model.eval()
    feats, labels = [], []
    for xb, yb in loader:
        emb = model.embed(xb.to(device)).cpu().numpy()
        feats.append(emb)
        labels.append(yb.numpy())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    sim = feats @ feats.T
    scores, lbls = [], []
    n = len(feats)
    rng = np.random.default_rng(0)
    idx_pairs = rng.choice(n, size=(min(20000, n * n), 2))
    for a, b in idx_pairs:
        if a == b:
            continue
        scores.append(float(sim[a, b]))
        lbls.append(1 if labels[a] == labels[b] else -1)
    return compute_eer(np.column_stack([scores, lbls]))


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Identity classification / verification from
# masked amplitude alone
# ══════════════════════════════════════════════════════════════
def experiment_1_identity_leakage(samples, img_side, alpha, scale, device):
    print(f"\n{'─'*70}")
    print(f" Experiment 1 — Identity leakage from masked amplitude "
          f"(alpha={alpha}, scale={scale})")
    print(f"{'─'*70}")

    # Restrict to a single client for a clean closed-set identity task
    # (local labels are per-client; using client 0 keeps label space simple).
    client0_samples = [(p, l) for p, l, c in samples if c == 0]
    n_classes = len(set(l for _, l in client0_samples))
    print(f" Using client-0 samples: {len(client0_samples)} images, "
          f"{n_classes} identities")

    descriptors, labels = [], []
    for path, label in client0_samples:
        img_np = load_gray(path, img_side)
        descriptors.append(masked_amplitude_descriptor(img_np, alpha, scale))
        labels.append(label)

    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.3, stratify=labels, random_state=42)

    train_ds = DescriptorDataset([descriptors[i] for i in train_idx],
                                  [labels[i] for i in train_idx])
    test_ds = DescriptorDataset([descriptors[i] for i in test_idx],
                                 [labels[i] for i in test_idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    in_dim = descriptors[0].shape[0]
    model = MLPProbe(in_dim, n_classes)
    print(f" Probe input dim (r) = {in_dim}")
    model = train_probe(model, train_loader, device, epochs=30)

    # Closed-set classification accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    test_acc = 100.0 * correct / max(total, 1)
    chance_acc = 100.0 / n_classes

    # Open-set-style verification EER on the probe's embedding space
    full_loader = DataLoader(DescriptorDataset(descriptors, labels),
                              batch_size=64, shuffle=False)
    eer = eval_probe_verification(model, full_loader, device)

    result = {
        "alpha": alpha, "scale": scale,
        "n_identities": n_classes, "n_samples": len(labels),
        "descriptor_dim": in_dim,
        "test_accuracy_pct": test_acc,
        "chance_accuracy_pct": chance_acc,
        "verification_eer_pct": eer * 100,
    }
    print(f"\n Result: test_acc={test_acc:.2f}% "
          f"(chance={chance_acc:.2f}%)  verif_EER={eer*100:.2f}%")
    return result


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Simplest reconstruction attack: attacker has only
# the shared masked amplitude + a phase THEY hold (mean or random
# phase across the dataset), inverse-FFTs, then re-identifies via
# a verifier trained on real full images.
# ══════════════════════════════════════════════════════════════
def compute_mean_phase(sample_paths, img_side, n_ref=200):
    """Attacker-side generic phase: average phase over a public/
    reference sample of images (NOT the target's own phase)."""
    rng = np.random.default_rng(0)
    ref_paths = rng.choice(sample_paths, size=min(n_ref, len(sample_paths)),
                            replace=False)
    phases = []
    for p in ref_paths:
        img_np = load_gray(p, img_side)
        _, phase = full_fft(img_np)
        phases.append(phase)
    return np.mean(np.stack(phases), axis=0)


def reconstruct_from_masked_amplitude(masked_amp, phase):
    """Inverse FFT: masked amplitude (attacker-visible) + phase
    (attacker's best guess, NOT the true phase)."""
    complex_spec = masked_amp * np.exp(1j * phase)
    complex_spec = np.fft.ifftshift(complex_spec)
    img_recon = np.fft.ifft2(complex_spec).real
    return np.clip(img_recon, 0.0, 1.0)


@torch.no_grad()
def embed_with_verifier(model, img_np_list, device):
    model.eval()
    feats = []
    for img_np in img_np_list:
        t = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t = (t - t.mean()) / (t.std() + 1e-6)
        _, fe, _ = model(t.to(device), None, None)
        feats.append(fe.cpu().numpy().squeeze(0))
    return np.stack(feats)


def experiment_2_reconstruction_attack(samples, img_side, alpha, device,
                                        n_probe=200, verifier=None):
    print(f"\n{'─'*70}")
    print(f" Experiment 2 — Reconstruction attack (alpha={alpha})")
    print(f"{'─'*70}")

    client0_samples = [(p, l) for p, l, c in samples if c == 0]
    all_paths = [p for p, _ in client0_samples]
    mean_phase = compute_mean_phase(all_paths, img_side)

    rng = np.random.default_rng(1)
    probe_subset = list(rng.choice(len(client0_samples),
                                    size=min(n_probe, len(client0_samples)),
                                    replace=False))

    gallery_imgs, gallery_labels = [], []
    recon_imgs, recon_labels = [], []
    for i in probe_subset:
        path, label = client0_samples[i]
        img_np = load_gray(path, img_side)
        masked_amp = masked_amplitude_image(img_np, alpha)
        recon = reconstruct_from_masked_amplitude(masked_amp, mean_phase)

        gallery_imgs.append(img_np)
        gallery_labels.append(label)
        recon_imgs.append(recon)
        recon_labels.append(label)

    # Reuse the SAME held-out verifier (trained on real full images) to
    # score whether reconstructions re-identify the correct individual.
    if verifier is None:
        n_classes = len(set(l for _, l in client0_samples))
        verifier = compnet_fedpalm(num_classes=n_classes).to(device)
        verifier.eval()
        print(" [warn] No pretrained verifier passed in — using a "
              "randomly initialized network. Pass --verifier_ckpt for "
              "a meaningful re-ID number; random-network results below "
              "only sanity-check the pipeline, not real leakage.")

    gal_feats = embed_with_verifier(verifier, gallery_imgs, device)
    recon_feats = embed_with_verifier(verifier, recon_imgs, device)

    sim = recon_feats @ gal_feats.T
    nn_idx = np.argmax(sim, axis=1)
    correct = sum(gallery_labels[nn_idx[i]] == recon_labels[i]
                   for i in range(len(recon_labels)))
    reid_rate = 100.0 * correct / max(len(recon_labels), 1)

    result = {
        "alpha": alpha, "n_probe": len(probe_subset),
        "reconstruction_reid_rate_pct": reid_rate,
    }
    print(f"\n Result: reconstruction→re-ID rate = {reid_rate:.2f}% "
          f"(chance ≈ {100.0/max(len(set(gallery_labels)),1):.2f}%)")
    return result


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Membership inference: can an attacker tell if a
# given amplitude descriptor was IN a client's training set?
# Simplified shadow-model protocol appropriate for small N.
# ══════════════════════════════════════════════════════════════
def experiment_3_membership_inference(samples, img_side, alpha, scale, device):
    print(f"\n{'─'*70}")
    print(f" Experiment 3 — Membership inference (alpha={alpha}, scale={scale})")
    print(f"{'─'*70}")

    client0_samples = [(p, l) for p, l, c in samples if c == 0]
    n_classes = len(set(l for _, l in client0_samples))

    rng = np.random.default_rng(2)
    idx = np.arange(len(client0_samples))
    rng.shuffle(idx)
    half = len(idx) // 2
    member_idx, nonmember_idx = idx[:half], idx[half:]

    # "Shadow model": a probe trained ONLY on member_idx, mimicking a
    # client's local statistics computed only from its true training set.
    member_desc, member_lab = [], []
    for i in member_idx:
        path, label = client0_samples[i]
        img_np = load_gray(path, img_side)
        member_desc.append(masked_amplitude_descriptor(img_np, alpha, scale))
        member_lab.append(label)

    shadow_ds = DescriptorDataset(member_desc, member_lab)
    shadow_loader = DataLoader(shadow_ds, batch_size=64, shuffle=True)
    in_dim = member_desc[0].shape[0]
    shadow_model = MLPProbe(in_dim, n_classes)
    shadow_model = train_probe(shadow_model, shadow_loader, device, epochs=30)
    shadow_model.eval()

    # Attack feature: per-sample max softmax confidence (standard MI signal)
    def confidence_scores(sample_idxs):
        descs = []
        for i in sample_idxs:
            path, _ = client0_samples[i]
            img_np = load_gray(path, img_side)
            descs.append(masked_amplitude_descriptor(img_np, alpha, scale))
        x = torch.tensor(np.stack(descs), dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = F.softmax(shadow_model(x), dim=1)
        return probs.max(dim=1).values.cpu().numpy()

    member_conf = confidence_scores(member_idx)
    nonmember_conf = confidence_scores(nonmember_idx)

    mi_scores = np.concatenate([member_conf, nonmember_conf])
    mi_labels = np.concatenate([np.ones_like(member_conf),
                                 np.zeros_like(nonmember_conf)])
    try:
        auc = roc_auc_score(mi_labels, mi_scores)
    except ValueError:
        auc = float("nan")

    result = {
        "alpha": alpha, "scale": scale,
        "n_members": len(member_idx), "n_nonmembers": len(nonmember_idx),
        "membership_inference_auc": auc,
    }
    print(f"\n Result: membership-inference AUC = {auc:.4f} "
          f"(0.5 = no leakage, 1.0 = perfect attack)")
    return result


# ══════════════════════════════════════════════════════════════
# MAIN — runs all 3 experiments across alpha values / datasets
# ══════════════════════════════════════════════════════════════
def run_for_dataset(dataset_name, alphas, scale, device, args):
    print(f"\n{'='*70}")
    print(f" PRIVACY AUDIT — {dataset_name.upper()}")
    print(f"{'='*70}")

    cfg = get_config("proposed")
    cfg["dataset"] = dataset_name
    if dataset_name == "xjtu":
        cfg["n_ids"] = args.xjtu_n_ids

    client_data, _, _, _, spectra = get_federated_splits(cfg, seed=cfg["random_seed"])
    samples = collect_all_samples(client_data)
    print(f" Loaded {len(client_data)} clients, {len(samples)} total samples")

    verifier = None
    if args.verifier_ckpt and os.path.exists(args.verifier_ckpt):
        n_classes = client_data[0]["num_classes"]
        verifier = compnet_fedpalm(num_classes=n_classes).to(device)
        verifier.load_state_dict(torch.load(args.verifier_ckpt, map_location=device))
        verifier.eval()
        print(f" Loaded pretrained verifier from {args.verifier_ckpt}")

    all_rows = []
    for alpha in alphas:
        row = {"dataset": dataset_name, "alpha": alpha, "scale": scale}
        row["exp1"] = experiment_1_identity_leakage(
            samples, cfg["img_side"], alpha, scale, device)
        row["exp2"] = experiment_2_reconstruction_attack(
            samples, cfg["img_side"], alpha, device, verifier=verifier)
        row["exp3"] = experiment_3_membership_inference(
            samples, cfg["img_side"], alpha, scale, device)
        all_rows.append(row)

    return all_rows


def print_summary_table(all_results):
    print(f"\n{'='*100}")
    print(" SUMMARY — Privacy Audit Across Datasets and Alpha")
    print(f"{'='*100}")
    header = (f"{'Dataset':<10}{'alpha':<8}{'scale':<8}"
              f"{'Cls.Acc%':<10}{'Chance%':<10}{'Verif.EER%':<12}"
              f"{'Recon.ReID%':<13}{'MI.AUC':<10}")
    print(header)
    print("─" * len(header))
    for row in all_results:
        e1, e2, e3 = row["exp1"], row["exp2"], row["exp3"]
        print(f"{row['dataset']:<10}{row['alpha']:<8}{row['scale']:<8}"
              f"{e1['test_accuracy_pct']:<10.2f}{e1['chance_accuracy_pct']:<10.2f}"
              f"{e1['verification_eer_pct']:<12.2f}"
              f"{e2['reconstruction_reid_rate_pct']:<13.2f}"
              f"{e3['membership_inference_auc']:<10.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Privacy audit of masked amplitude")
    p.add_argument("--datasets", nargs="+", default=["casiams", "xjtu", "xpalm"],
                    choices=["casiams", "xjtu", "xpalm"])
    p.add_argument("--alpha", type=float, default=None,
                    help="Single alpha to test.")
    p.add_argument("--alpha_sweep", nargs="+", type=float,
                    default=[0.05, 0.10, 0.15, 0.25, 0.40],
                    help="Sweep of alpha values (used if --alpha not set).")
    p.add_argument("--scale", choices=["linear", "log"], default="linear")
    p.add_argument("--xjtu_n_ids", type=int, default=192)
    p.add_argument("--verifier_ckpt", default=None,
                    help="Path to a pretrained compnet_fedpalm state_dict "
                         "for Exp 2's re-ID scoring. Without this, Exp 2 "
                         "uses a random-init network (pipeline check only).")
    p.add_argument("--out", default="privacy_audit_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas = [args.alpha] if args.alpha is not None else args.alpha_sweep

    all_results = []
    for dataset_name in args.datasets:
        all_results.extend(
            run_for_dataset(dataset_name, alphas, args.scale, device, args))

    print_summary_table(all_results)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
