# ==============================================================
#  utils.py — FFT utilities, evaluation, and training functions
# ==============================================================
#
#  FFT utilities:
#    gaussian_mask, extract_style_template, apply_style_template
#
#  Evaluation:
#    extract_features, compute_eer, evaluate_model
#
#  Training (one epoch):
#    train_compnet_epoch  — CE + ArcFace  (single image per sample)
#    train_ccnet_epoch    — CE + SupConLoss (paired images per sample)
# ==============================================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import SupConLoss


# ══════════════════════════════════════════════════════════════
#  FFT STYLE AUGMENTATION HELPERS
# ══════════════════════════════════════════════════════════════

def gaussian_mask(H, W, beta):
    """
    Soft Gaussian low-frequency mask centred at DC.
    Values close to 1 at centre (low freq), close to 0 at edges (high freq).
    """
    sigma  = min(H, W) * beta
    cy, cx = H // 2, W // 2
    ys = np.arange(H) - cy
    xs = np.arange(W) - cx
    xs, ys = np.meshgrid(xs, ys)
    return np.exp(-(xs**2 + ys**2) / (2 * sigma**2)).astype(np.float32)


def extract_style_template(img_np):
    """
    Extract the full fftshifted amplitude spectrum of a grayscale image.
    Shareable style descriptor — captures global illumination and spectral
    tone but no fine-grained texture or identity information.
    The full spectrum is always extracted; the Gaussian mask is applied
    only during augmentation in apply_style_template().

    Parameters
    ----------
    img_np : np.ndarray  (H, W)  float32 in [0, 1]

    Returns
    -------
    amp_shifted : np.ndarray  (H, W)  fftshifted amplitude
    """
    amp = np.abs(np.fft.fft2(img_np))
    return np.fft.fftshift(amp).astype(np.float32)


def apply_style_template(img_np, amp_template, beta):
    """
    Swap the low-frequency amplitude of img_np with amp_template using a
    Gaussian soft mask. Phase (identity-specific texture) is preserved.

    The foreground mask (img_np > 0) is saved before FFT and re-applied
    after reconstruction so that NormSingleROI always operates on the
    correct palm ROI pixels — inverse FFT leaves small non-zero residuals
    in the background which would corrupt the per-ROI statistics.

    Parameters
    ----------
    img_np       : np.ndarray  (H, W)  float32 in [0, 1]
    amp_template : np.ndarray  (H, W)  fftshifted amplitude from another client
    beta         : float       Gaussian sigma as fraction of min(H, W)

    Returns
    -------
    img_syn : np.ndarray  (H, W)  float32 in [0, 1]
    """
    H, W  = img_np.shape[:2]
    mask  = gaussian_mask(H, W, beta)

    # preserve foreground mask before FFT — background stays zero after synthesis
    fg_mask = img_np > 0

    fft   = np.fft.fft2(img_np)
    amp_s = np.fft.fftshift(np.abs(fft))
    pha   = np.angle(fft)

    # soft blend: low-freq from template, high-freq from original
    amp_syn = (1.0 - mask) * amp_s + mask * amp_template
    amp_syn = np.fft.ifftshift(amp_syn)

    img_syn = np.fft.ifft2(amp_syn * np.exp(1j * pha)).real
    img_syn = np.clip(img_syn, 0.0, 1.0).astype(np.float32)

    # re-apply foreground mask so NormSingleROI gets correct per-ROI statistics
    img_syn[~fg_mask] = 0.0
    return img_syn


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    """
    Extract L2-normalised embeddings from a DataLoader using model.get_embedding().
    Compatible with both CompNet and CCNet (both expose get_embedding).

    Returns
    -------
    feats  : np.ndarray  [N, embed_dim]
    labels : np.ndarray  [N]
    """
    model.eval()
    feats, labels = [], []
    for imgs, labs in loader:
        feats.append(model.get_embedding(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def compute_eer(scores_array):
    """
    EER from an Nx2 array of [cosine_score, label(+1/-1)].
    Higher cosine score = more similar (genuine pairs score near +1).
    Returns EER as a float in [0, 1].
    """
    ins  = scores_array[scores_array[:, 1] ==  1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0:
        return 1.0
    y = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s = np.concatenate([ins, outs])
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))


def evaluate_model(model, gallery_loader, probe_loader, device):
    """
    Cosine-similarity evaluation on pre-split gallery and probe loaders.
    Uses model.get_embedding() — compatible with both CompNet and CCNet.

    Returns
    -------
    eer   : float  EER in [0, 1]
    rank1 : float  Rank-1 accuracy in [0, 100]
    """
    gal_feats, gal_labels = extract_features(model, gallery_loader, device)
    prb_feats, prb_labels = extract_features(model, probe_loader,   device)

    # cosine similarity matrix (embeddings are L2-normalised)
    sim_matrix  = prb_feats @ gal_feats.T

    scores_list, labels_list = [], []
    for i in range(len(prb_feats)):
        for j in range(len(gal_feats)):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if prb_labels[i] == gal_labels[j] else -1)

    eer = compute_eer(np.column_stack([scores_list, labels_list]))

    nn_idx  = np.argmax(sim_matrix, axis=1)
    correct = sum(prb_labels[i] == gal_labels[nn_idx[i]]
                  for i in range(len(prb_feats)))
    rank1   = 100.0 * correct / max(len(prb_feats), 1)

    return eer, rank1


# ══════════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS  (one epoch per call)
# ══════════════════════════════════════════════════════════════

def train_compnet_epoch(model, loader, criterion, optimizer, device):
    """
    Train CompNet for one epoch.

    Loss      : CrossEntropyLoss on ArcFace logits.
    Dataset   : returns (img, label) — single image per sample.
    Forward   : model(img, label) → logits (ArcFace applied during training).

    Parameters
    ----------
    model     : CompNet
    loader    : DataLoader yielding (img_tensor, label_tensor)
    criterion : nn.CrossEntropyLoss
    optimizer : torch.optim.Optimizer
    device    : torch.device

    Returns
    -------
    avg_loss : float
    accuracy : float  (classification accuracy in %)
    """
    model.train()
    running_loss = 0.0; correct = 0; total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs, labels)          # ArcFace forward with labels
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct      += out.argmax(1).eq(labels).sum().item()
        total        += imgs.size(0)

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train_ccnet_epoch(model, loader, criterion, optimizer, device,
                      ce_weight=0.8, con_weight=0.2, temperature=0.07):
    """
    Train CCNet for one epoch.

    Loss      : ce_weight * CrossEntropy  +  con_weight * SupConLoss.
    Dataset   : PairedDataset returns ([img1, img2], label).
                img2 is a different sample of the same identity —
                required to form the two views for SupConLoss.
    Forward   : model(img, label) → (logits, norm_features).
                Both images are forwarded independently; their feature
                vectors are stacked into a [batch, 2, feat_dim] tensor
                and passed to SupConLoss.

    Parameters
    ----------
    model       : CCNet
    loader      : DataLoader yielding ([img1, img2], label)
    criterion   : nn.CrossEntropyLoss
    optimizer   : torch.optim.Optimizer
    device      : torch.device
    ce_weight   : float  weight for CrossEntropy loss (default 0.8)
    con_weight  : float  weight for SupConLoss (default 0.2)
    temperature : float  SupConLoss temperature (default 0.07)

    Returns
    -------
    avg_loss : float
    accuracy : float  (classification accuracy on first view in %)
    """
    con_criterion = SupConLoss(temperature=temperature,
                               base_temperature=temperature)
    model.train()
    running_loss = 0.0; correct = 0; total = 0

    for datas, labels in loader:
        img1 = datas[0].to(device)
        img2 = datas[1].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits1, fe1 = model(img1, labels)
        logits2, fe2 = model(img2, labels)

        # stack views for SupConLoss: [batch, 2, feat_dim]
        fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

        ce_loss  = criterion(logits1, labels)
        con_loss = con_criterion(fe, labels)
        loss     = ce_weight * ce_loss + con_weight * con_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img1.size(0)
        correct      += logits1.argmax(1).eq(labels).sum().item()
        total        += img1.size(0)

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)
