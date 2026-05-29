# ==============================================================
#  utils.py — FFT utilities, evaluation, and training functions
# ==============================================================
#
#  FFT utilities:
#    gaussian_mask, extract_style_template, extract_radial_template,
#    apply_style_template
#
#  Evaluation:
#    extract_features, compute_eer, whiten_features, evaluate_model
#
#  Training (one epoch):
#    train_compnet_epoch  — CE + ArcFace (+ style/supcon/grl/center)
#                           Handles plain CompNet, DINOv2, and
#                           MultiExpertCompNet (base+domain logit routing).
#    train_ccnet_epoch    — CE + SupConLoss (paired images per sample)
# ==============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import SupConLoss, GRL


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
    Extract fftshifted amplitude spectrum from a grayscale or RGB image.

    Grayscale (H, W)    → returns (H, W)      amplitude array
    RGB       (H, W, 3) → returns (H, W, 3)   per-channel amplitude arrays
    """
    if img_np.ndim == 2:
        return np.fft.fftshift(
            np.abs(np.fft.fft2(img_np))).astype(np.float32)
    return np.stack([
        np.fft.fftshift(np.abs(np.fft.fft2(img_np[..., c])))
        for c in range(img_np.shape[2])
    ], axis=-1).astype(np.float32)


def extract_radial_template(img_np, n_bins=64):
    """
    Extract a radial spectral profile from a grayscale image.
    Returns a (n_bins,) float32 array — the mean log-amplitude per
    concentric frequency ring.  Grayscale only.
    """
    assert img_np.ndim == 2, "extract_radial_template expects grayscale (H,W)"
    H, W   = img_np.shape
    cy, cx = H // 2, W // 2
    amp    = np.fft.fftshift(np.abs(np.fft.fft2(img_np)))
    amp_log = np.log1p(amp)
    Y, X   = np.ogrid[:H, :W]
    R      = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r_max  = min(H, W) / 2.0
    bins   = np.zeros(n_bins, dtype=np.float32)
    edges  = np.linspace(0, r_max, n_bins + 1)
    for b in range(n_bins):
        ring = (R >= edges[b]) & (R < edges[b + 1])
        if ring.any():
            bins[b] = amp_log[ring].mean()
    return bins


def apply_style_template(img_np, amp_template, beta, method="amplitude"):
    """
    Apply FFT-based domain style transfer to img_np using amp_template.

    method="amplitude" : swap low-frequency amplitude via Gaussian soft mask.
    method="radial"    : scale each frequency ring to match donor's radial
                         log-amplitude profile.
    """
    if method == "radial":
        return _apply_radial_template(img_np, amp_template)

    H, W = img_np.shape[:2]
    mask = gaussian_mask(H, W, beta)

    def _swap_channel(ch, amp_tpl_ch):
        fft     = np.fft.fft2(ch)
        amp_s   = np.fft.fftshift(np.abs(fft))
        pha     = np.angle(fft)
        amp_syn = (1.0 - mask) * amp_s + mask * amp_tpl_ch
        amp_syn = np.fft.ifftshift(amp_syn)
        return np.clip(
            np.fft.ifft2(amp_syn * np.exp(1j * pha)).real,
            0.0, 1.0).astype(np.float32)

    if img_np.ndim == 2:
        fg_mask = img_np > 0
        result  = _swap_channel(img_np, amp_template)
        result[~fg_mask] = 0.0
        return result

    return np.stack([
        _swap_channel(img_np[..., c], amp_template[..., c])
        for c in range(img_np.shape[2])
    ], axis=-1).astype(np.float32)


def _apply_radial_template(img_np, radial_profile_donor):
    """
    Radial profile matching — scale each frequency ring of img_np so its
    mean log-amplitude matches the donor's radial profile.  Grayscale only.
    """
    assert img_np.ndim == 2, "_apply_radial_template expects grayscale (H,W)"
    H, W   = img_np.shape
    n_bins = len(radial_profile_donor)
    cy, cx = H // 2, W // 2

    fft    = np.fft.fft2(img_np)
    amp    = np.fft.fftshift(np.abs(fft))
    pha    = np.angle(fft)
    amp_log = np.log1p(amp)

    Y, X = np.ogrid[:H, :W]
    R    = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r_max = min(H, W) / 2.0
    edges = np.linspace(0, r_max, n_bins + 1)

    amp_syn = amp.copy()
    for b in range(n_bins):
        ring = (R >= edges[b]) & (R < edges[b + 1])
        if not ring.any():
            continue
        target_mean = amp_log[ring].mean()
        donor_mean  = radial_profile_donor[b]
        scale = np.exp(donor_mean - target_mean)
        amp_syn[ring] = amp[ring] * scale

    amp_syn = np.fft.ifftshift(amp_syn)
    img_syn = np.clip(
        np.fft.ifft2(amp_syn * np.exp(1j * pha)).real,
        0.0, 1.0).astype(np.float32)

    fg_mask = img_np > 0
    img_syn[~fg_mask] = 0.0
    return img_syn


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    """
    Extract L2-normalised embeddings using model.get_embedding().
    Compatible with CompNet, CCNet, DINOv2.  For MultiExpertCompNet the
    domain-aware path is in main.evaluate_model_with_domain (this plain
    version uses base expert only via get_embedding(x) without domain_ids).
    NaN guard: any NaN row replaced with zero vector.
    """
    model.eval()
    feats, labels = [], []
    for batch in loader:
        # support 2-tuple (img, label) and 3-tuple (img, label, domain_id)
        if len(batch) == 3:
            imgs, labs, _ = batch
        else:
            imgs, labs = batch
        batch_np = model.get_embedding(imgs.to(device)).cpu().numpy()
        nan_rows = np.isnan(batch_np).any(axis=1)
        if nan_rows.any():
            batch_np[nan_rows] = 0.0
        feats.append(batch_np)
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def compute_eer(scores_array):
    """
    EER from an Nx2 array of [cosine_score, label(+1/-1)].
    Returns EER in [0, 1]. Returns 1.0 on error or NaN.
    """
    ins  = scores_array[scores_array[:, 1] ==  1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0:
        return 1.0
    y = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s = np.concatenate([ins, outs])
    if not np.isfinite(s).all():
        return 1.0
    try:
        fpr, tpr, _ = roc_curve(y, s, pos_label=1)
        return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))
    except Exception:
        return 1.0


def whiten_features(gal_feats, prb_feats, eps=1e-4, n_components=None):
    """
    PCA whitening — projects to principal components, whitens, L2-normalises.
    Estimated from gallery, applied to both gallery and probe.
    """
    N, D = gal_feats.shape
    if n_components is None:
        n_components = min(N - 1, D, 256)

    global_mean = gal_feats.mean(axis=0)
    centred_gal = gal_feats - global_mean
    _, s, Vt = np.linalg.svd(centred_gal, full_matrices=False)

    Vt_k = Vt[:n_components]
    s_k  = s[:n_components]
    eigvals = (s_k ** 2) / N
    scale   = 1.0 / np.sqrt(eigvals + eps)

    gal_proj = (gal_feats - global_mean) @ Vt_k.T
    prb_proj = (prb_feats - global_mean) @ Vt_k.T
    gal_w = gal_proj * scale
    prb_w = prb_proj * scale

    gal_w = gal_w / (np.linalg.norm(gal_w, axis=1, keepdims=True) + 1e-8)
    prb_w = prb_w / (np.linalg.norm(prb_w, axis=1, keepdims=True) + 1e-8)
    return gal_w, prb_w


def evaluate_model(model, gallery_loader, probe_loader, device,
                   use_whitening=False):
    """
    Cosine-similarity evaluation on gallery and probe sets.
    For MultiExpertCompNet, prefer main.evaluate_model_with_domain which
    passes domain_ids; this function uses base-expert embeddings only.
    """
    gal_feats, gal_labels = extract_features(model, gallery_loader, device)
    prb_feats, prb_labels = extract_features(model, probe_loader,   device)

    if use_whitening:
        gal_feats, prb_feats = whiten_features(gal_feats, prb_feats)

    sim_matrix = np.nan_to_num(prb_feats @ gal_feats.T, nan=0.0)

    scores_list, labels_list = [], []
    for i in range(len(prb_feats)):
        for j in range(len(gal_feats)):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if prb_labels[i] == gal_labels[j] else -1)

    eer     = compute_eer(np.column_stack([scores_list, labels_list]))
    nn_idx  = np.argmax(sim_matrix, axis=1)
    correct = sum(prb_labels[i] == gal_labels[nn_idx[i]]
                  for i in range(len(prb_feats)))
    rank1   = 100.0 * correct / max(len(prb_feats), 1)
    return eer, rank1


# ══════════════════════════════════════════════════════════════
#  CENTER LOSS
# ══════════════════════════════════════════════════════════════

class CenterLoss(nn.Module):
    """
    Center Loss — minimises distance between embeddings and class centres.
    Per-client, never shared, carried over across rounds.
    """
    def __init__(self, num_classes, embed_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.centres = nn.Parameter(
            torch.zeros(num_classes, embed_dim, device=device),
            requires_grad=True)

    def forward(self, embeddings, labels):
        centres_batch = self.centres[labels]
        return ((embeddings - centres_batch) ** 2).sum(dim=1).mean()


# ══════════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS  (one epoch per call)
# ══════════════════════════════════════════════════════════════

def train_compnet_epoch(model, loader, criterion, optimizer, device,
                        center_loss=None, center_optimizer=None,
                        lambda_center=0.0, lambda_style=0.0,
                        lambda_grl=0.0, lambda_load_balance=0.0,
                        lambda_supcon=0.0, temperature=0.07):
    """
    Train CompNet / DINOv2 / MultiExpertCompNet for one epoch.

    Loss : CE(aggregated logits)
         + lambda_style   × StyleConsistencyLoss   (optional)
         + lambda_supcon  × SupConLoss              (optional)
         + lambda_grl     × DomainAdversarialLoss   (optional, GRL)
         + lambda_center  × CenterLoss              (optional)

    MultiExpertCompNet routing
    ──────────────────────────
    For the classification logits the full model.forward(orig, labels,
    domain_ids) is used — it internally averages base + domain expert
    logits (0.5/0.5).  For embedding-level losses (style, supcon, center,
    grl) the BASE expert's backbone embedding is used, since those losses
    enforce representation properties rather than routing behaviour.
    """
    from models import MultiExpertCompNet
    is_multi_expert = isinstance(model, MultiExpertCompNet)

    model.train()
    if center_loss is not None:
        center_loss.train()

    use_center  = (center_loss is not None and
                   center_optimizer is not None and
                   lambda_center > 0.0)
    use_style   = lambda_style  > 0.0
    use_supcon  = lambda_supcon > 0.0
    # GRL only supported on plain CompNet (single domain_classifier head)
    use_grl     = (lambda_grl > 0.0 and not is_multi_expert and
                   hasattr(model, "use_grl") and model.use_grl and
                   getattr(model, "domain_classifier", None) is not None)

    supcon_criterion = SupConLoss(temperature=temperature,
                                  base_temperature=temperature) \
                       if use_supcon else None

    def _get_emb(imgs, domain_ids=None):
        # MultiExpertCompNet: embedding-level losses use the base expert.
        if is_multi_expert:
            return model.experts[0]._backbone(imgs)
        if hasattr(model, "_backbone"):
            try:
                return model._backbone(imgs, domain_ids)
            except TypeError:
                return model._backbone(imgs)
        return model.backbone(imgs)

    def _get_logits(orig, emb, labels, domain_ids=None):
        # MultiExpertCompNet aggregates base+domain logits in forward().
        if is_multi_expert:
            return model(orig, labels, domain_ids)
        if hasattr(model, "drop"):
            return model.arc(model.drop(emb), labels)
        return model.arc(emb, labels)

    running_loss = 0.0; correct = 0; total = 0

    for batch in loader:
        if len(batch) == 3:
            data, labels, domain_ids = batch
            domain_ids = domain_ids.to(device)
        else:
            data, labels = batch
            domain_ids   = None

        labels = labels.to(device)

        if isinstance(data, (list, tuple)):
            orig    = data[0].to(device)
            aug     = data[1].to(device)
            has_aug = True
        else:
            orig    = data.to(device)
            aug     = None
            has_aug = False

        optimizer.zero_grad()
        if use_center:
            center_optimizer.zero_grad()

        # ── logits (routed) and CE loss ───────────────────────────────────────
        emb_orig = _get_emb(orig, domain_ids)
        logits   = _get_logits(orig, emb_orig, labels, domain_ids)
        loss     = criterion(logits, labels)

        # ── auxiliary embeddings for aug-based losses ─────────────────────────
        if use_style and has_aug:
            emb_aug = _get_emb(aug, domain_ids)
            sim     = F.cosine_similarity(
                F.normalize(emb_orig, p=2, dim=1),
                F.normalize(emb_aug,  p=2, dim=1), dim=1)
            loss = loss + lambda_style * (1.0 - sim.mean())
        elif has_aug and (use_supcon or use_grl):
            emb_aug = _get_emb(aug, domain_ids)

        if use_supcon and has_aug:
            emb_o = F.normalize(emb_orig, p=2, dim=1)
            emb_a = F.normalize(emb_aug,  p=2, dim=1)
            feats = torch.stack([emb_o, emb_a], dim=1)   # [B, 2, D]
            loss  = loss + lambda_supcon * supcon_criterion(feats, labels)

        if use_grl and domain_ids is not None:
            emb_grl    = GRL.apply(emb_orig, lambda_grl)
            dom_logits = model.domain_classifier(emb_grl)
            loss_dom   = F.cross_entropy(dom_logits, domain_ids)
            loss       = loss + lambda_grl * loss_dom

        if use_center:
            loss = loss + lambda_center * center_loss(emb_orig.detach(), labels)

        loss.backward()
        optimizer.step()

        if use_center:
            for p in center_loss.parameters():
                if p.grad is not None:
                    p.grad.data *= 1.0 / (lambda_center + 1e-8)
            center_optimizer.step()

        running_loss += loss.item() * orig.size(0)
        correct      += logits.argmax(1).eq(labels).sum().item()
        total        += orig.size(0)

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train_ccnet_epoch(model, loader, criterion, optimizer, device,
                      ce_weight=0.8, con_weight=0.2, temperature=0.07,
                      center_loss=None, center_optimizer=None,
                      lambda_center=0.0):
    """
    Train CCNet for one epoch.
    Loss : ce_weight × CE + con_weight × SupConLoss + lambda_center × CenterLoss
    """
    con_criterion = SupConLoss(temperature=temperature,
                               base_temperature=temperature)
    model.train()
    if center_loss is not None:
        center_loss.train()

    use_center = (center_loss is not None and
                  center_optimizer is not None and
                  lambda_center > 0.0)

    running_loss = 0.0; correct = 0; total = 0

    for datas, labels in loader:
        img1   = datas[0].to(device)
        img2   = datas[1].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if use_center:
            center_optimizer.zero_grad()

        logits1, fe1 = model(img1, labels)
        logits2, fe2 = model(img2, labels)

        fe       = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        ce_loss  = criterion(logits1, labels)
        con_loss = con_criterion(fe, labels)
        loss     = ce_weight * ce_loss + con_weight * con_loss

        if use_center:
            with torch.no_grad():
                emb1 = model._extract(img1)
                emb1 = model.fc1(model.fc(emb1))
            cl_loss = center_loss(emb1.detach(), labels)
            loss    = loss + lambda_center * cl_loss

        loss.backward()
        optimizer.step()

        if use_center:
            for p in center_loss.parameters():
                if p.grad is not None:
                    p.grad.data *= 1.0 / (lambda_center + 1e-8)
            center_optimizer.step()

        running_loss += loss.item() * img1.size(0)
        correct      += logits1.argmax(1).eq(labels).sum().item()
        total        += img1.size(0)

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)
