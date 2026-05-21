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
import torch.nn.functional as F
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
    Extract fftshifted amplitude spectrum from a grayscale or RGB image.

    Grayscale (H, W)   → returns (H, W)      amplitude array
    RGB       (H, W, 3) → returns (H, W, 3)   per-channel amplitude arrays

    For CompNet/CCNet the input is always (H, W) grayscale.
    For DINOv2 the input is (H, W, 3) RGB — each channel is processed
    independently so the style template preserves per-channel illumination.
    """
    if img_np.ndim == 2:
        return np.fft.fftshift(
            np.abs(np.fft.fft2(img_np))).astype(np.float32)
    # RGB: process each channel independently
    return np.stack([
        np.fft.fftshift(np.abs(np.fft.fft2(img_np[..., c])))
        for c in range(img_np.shape[2])
    ], axis=-1).astype(np.float32)


def apply_style_template(img_np, amp_template, beta):
    """
    Swap low-frequency amplitude of img_np with amp_template using a
    Gaussian soft mask. Phase (identity-specific texture) is preserved.

    Grayscale (H, W):
      Foreground mask (> 0) is saved and restored after synthesis so
      NormSingleROI always operates on palm ROI pixels only.

    RGB (H, W, 3):
      Each channel is processed independently. No foreground mask —
      RGB palmprint images (e.g. XJTU smartphone) have no zero-padded
      background so masking is not needed.

    Parameters
    ----------
    img_np       : np.ndarray  (H, W) or (H, W, 3)  float32 in [0, 1]
    amp_template : np.ndarray  same shape as img_np  fftshifted amplitude
    beta         : float       Gaussian sigma as fraction of min(H, W)

    Returns
    -------
    img_syn : np.ndarray  same shape as img_np  float32 in [0, 1]
    """
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
        # grayscale — restore foreground mask so NormSingleROI is correct
        fg_mask = img_np > 0
        result  = _swap_channel(img_np, amp_template)
        result[~fg_mask] = 0.0
        return result

    # RGB — process per channel, no foreground mask
    return np.stack([
        _swap_channel(img_np[..., c], amp_template[..., c])
        for c in range(img_np.shape[2])
    ], axis=-1).astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    """
    Extract L2-normalised embeddings from a DataLoader using model.get_embedding().
    Compatible with CompNet, CCNet, and DINOv2 (all expose get_embedding).

    NaN guard: a randomly-initialised model can produce near-zero embedding
    vectors; after L2 normalisation these become 0/0 = NaN. Any NaN row is
    replaced with a zero vector so downstream evaluation remains numerically
    stable (zero vectors produce a cosine similarity of 0 with all gallery
    entries, which is a safe neutral score).

    Returns
    -------
    feats  : np.ndarray  [N, embed_dim]  — NaN-free
    labels : np.ndarray  [N]
    """
    model.eval()
    feats, labels = [], []
    for imgs, labs in loader:
        batch_feats = model.get_embedding(imgs.to(device)).cpu().numpy()
        # replace any NaN row (zero-norm embedding after L2 norm) with zeros
        nan_rows = np.isnan(batch_feats).any(axis=1)
        if nan_rows.any():
            batch_feats[nan_rows] = 0.0
        feats.append(batch_feats)
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def compute_eer(scores_array):
    """
    EER from an Nx2 array of [cosine_score, label(+1/-1)].
    Higher cosine score = more similar (genuine pairs score near +1).
    Returns EER as a float in [0, 1].
    Returns 1.0 (worst case) if scores contain NaN or classes are missing.
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


def evaluate_model(model, gallery_loader, probe_loader, device):
    """
    Cosine-similarity evaluation on pre-split gallery and probe loaders.
    Uses model.get_embedding() — compatible with all models.

    Returns
    -------
    eer   : float  EER in [0, 1]
    rank1 : float  Rank-1 accuracy in [0, 100]
    """
    gal_feats, gal_labels = extract_features(model, gallery_loader, device)
    prb_feats, prb_labels = extract_features(model, probe_loader,   device)

    # cosine similarity matrix (embeddings are L2-normalised)
    sim_matrix = prb_feats @ gal_feats.T
    # guard against any residual NaN in similarity matrix
    sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)

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
#  DOMAIN PROTOTYPE EXTRACTION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_domain_prototypes(model, loader, device, K=20):
    """
    Extract K cluster centres from the embeddings of all local training
    samples, computed using the current (global) backbone.

    Called BEFORE local training so that all clients' prototypes live in
    the same feature space — the shared global backbone with no ArcFace
    specialisation. Identity variation averages out across the K clusters
    when K << N_samples, leaving the dominant domain structure.

    Parameters
    ----------
    model  : backbone model — must expose get_embedding()
    loader : DataLoader over the client's training samples
    device : torch.device
    K      : int  number of cluster centres to extract

    Returns
    -------
    centres : np.ndarray  [K, embed_dim]  L2-normalised cluster centres
    """
    from sklearn.cluster import KMeans

    model.eval()
    embs = []
    for imgs, _ in loader:
        e = model.get_embedding(imgs.to(device)).cpu().numpy()
        embs.append(e)
    embs = np.concatenate(embs, axis=0)   # [N, D]

    K_eff = min(K, len(embs))             # never more clusters than samples
    km    = KMeans(n_clusters=K_eff, n_init=10, random_state=42)
    km.fit(embs)

    centres = km.cluster_centers_.astype(np.float32)  # [K, D]
    # L2-normalise so cosine distance is used consistently
    norms   = np.linalg.norm(centres, axis=1, keepdims=True) + 1e-8
    return centres / norms

class CenterLoss(nn.Module):
    """
    Center Loss — minimises the distance between each embedding and
    its class centre, enforcing intra-class compactness.

    Reference: Wen et al., "A Discriminative Feature Learning Approach
    for Deep Face Recognition", ECCV 2016.

    In the FL setting, each client maintains its own set of centres — they
    encode per-client identity prototypes and are never shared with the
    server. Centres are carried over across rounds so they accumulate
    knowledge of where the backbone places each identity, even as the
    backbone itself is reset each round from the global model.

    Parameters
    ----------
    num_classes : int    number of training identities for this client
    embed_dim   : int    embedding dimensionality (512 for CompNet, 384 for DINOv2)
    device      : torch.device

    Usage
    -----
    center_loss = CenterLoss(num_classes, embed_dim, device)
    center_opt  = torch.optim.SGD(center_loss.parameters(), lr=center_lr)

    # inside training loop:
    emb  = model._backbone(imgs)          # raw embedding before ArcFace
    loss = ce_loss + lambda_c * center_loss(emb, labels)
    loss.backward()
    optimizer.step()
    # scale centre gradients then step (standard trick from the paper)
    for p in center_loss.parameters():
        p.grad.data *= 1.0 / (lambda_c + 1e-8)
    center_opt.step()
    """
    def __init__(self, num_classes, embed_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        # centres initialised to zero — pulled toward actual embeddings on
        # the first forward pass
        self.centres = nn.Parameter(
            torch.zeros(num_classes, embed_dim, device=device),
            requires_grad=True,
        )

    def forward(self, embeddings, labels):
        """
        Parameters
        ----------
        embeddings : Tensor [B, embed_dim]  raw (un-normalised) embeddings
        labels     : Tensor [B]             integer class indices

        Returns
        -------
        loss : scalar  mean squared distance to class centres
        """
        centres_batch = self.centres[labels]                     # [B, embed_dim]
        return ((embeddings - centres_batch) ** 2).sum(dim=1).mean()


# ══════════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS  (one epoch per call)

def train_compnet_epoch(model, loader, criterion, optimizer, device,
                        center_loss=None, center_optimizer=None,
                        lambda_center=0.0, lambda_style=0.0,
                        other_protos=None, proto_beta=0.3,
                        lambda_proto=0.0):
    """
    Train CompNet (or DINOv2) for one epoch.

    Loss : CrossEntropy(ArcFace, emb_orig)
         + lambda_style  × StyleConsistencyLoss       (optional)
         + lambda_center × CenterLoss(emb_orig)       (optional)
         + lambda_proto  × ProtoConsistencyLoss        (optional)

    StyleConsistencyLoss:
      1 - cosine_sim(emb_orig, emb_aug)
      Enforces that FFT/spatially-augmented embeddings stay close to
      their original counterparts — input-level domain invariance.

    ProtoConsistencyLoss:
      1 - cosine_sim(emb_orig, emb_mixed)
      where emb_mixed = (1-β)×emb_orig + β×nearest_other_proto
      For each sample, finds the nearest prototype from other clients
      (shared domain feature cluster centres extracted from the global
      backbone), blends the embedding toward it, and enforces that the
      backbone's output is close to the domain-shifted version. ArcFace
      never sees emb_mixed — no identity-label mismatch.
      Enforces feature-level domain invariance.

    Dataset returns ([orig, aug], label).
    ArcFace and CenterLoss use emb_orig only.

    Parameters
    ----------
    model            : CompNet or DINOv2Model
    loader           : DataLoader yielding ([orig, aug], label)
    criterion        : nn.CrossEntropyLoss
    optimizer        : torch.optim.Optimizer
    device           : torch.device
    center_loss      : CenterLoss | None
    center_optimizer : torch.optim.SGD | None
    lambda_center    : float  CenterLoss weight              (0 → disabled)
    lambda_style     : float  StyleConsistencyLoss weight    (0 → disabled)
    other_protos     : Tensor [N_other, D] | None  other clients' prototypes
    proto_beta       : float  blend weight toward donor prototype
    lambda_proto     : float  ProtoConsistencyLoss weight    (0 → disabled)

    Returns
    -------
    avg_loss : float
    accuracy : float  (classification accuracy on orig in %)
    """
    model.train()
    if center_loss is not None:
        center_loss.train()

    use_center = (center_loss is not None and
                  center_optimizer is not None and
                  lambda_center > 0.0)
    use_style  = lambda_style > 0.0
    use_proto  = (lambda_proto > 0.0 and
                  other_protos is not None and
                  len(other_protos) > 0)

    # pre-move prototypes to device once, L2-normalised
    if use_proto:
        protos_t = F.normalize(
            torch.tensor(other_protos, dtype=torch.float32, device=device),
            p=2, dim=1)                             # [N_other, D]

    def _get_emb(imgs):
        return model._backbone(imgs) if hasattr(model, "_backbone") \
               else model.backbone(imgs)

    def _get_logits(emb, labels):
        if hasattr(model, "drop"):
            return model.arc(model.drop(emb), labels)
        return model.arc(emb, labels)

    running_loss = 0.0; correct = 0; total = 0

    for data, labels in loader:
        orig   = data[0].to(device)
        aug    = data[1].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if use_center:
            center_optimizer.zero_grad()

        emb_orig = _get_emb(orig)
        logits   = _get_logits(emb_orig, labels)
        loss     = criterion(logits, labels)

        if use_style:
            emb_aug = _get_emb(aug)
            sim     = F.cosine_similarity(
                F.normalize(emb_orig, p=2, dim=1),
                F.normalize(emb_aug,  p=2, dim=1), dim=1)
            loss = loss + lambda_style * (1.0 - sim.mean())

        if use_proto:
            # find nearest other-client prototype for each sample
            emb_n      = F.normalize(emb_orig, p=2, dim=1)    # [B, D]
            dists      = torch.cdist(emb_n, protos_t)          # [B, N_other]
            nearest    = protos_t[dists.argmin(dim=1)]         # [B, D]

            # blend toward nearest donor prototype
            emb_mixed  = (1.0 - proto_beta) * emb_n + proto_beta * nearest
            emb_mixed  = F.normalize(emb_mixed, p=2, dim=1)

            # consistency loss — ArcFace never sees emb_mixed
            sim_proto  = F.cosine_similarity(emb_n, emb_mixed, dim=1)
            loss       = loss + lambda_proto * (1.0 - sim_proto.mean())

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

    Loss : ce_weight × CrossEntropy
         + con_weight × SupConLoss
         + lambda_center × CenterLoss  (optional)

    Parameters
    ----------
    model            : CCNet
    loader           : DataLoader yielding ([img1, img2], label)
    criterion        : nn.CrossEntropyLoss
    optimizer        : torch.optim.Optimizer
    device           : torch.device
    ce_weight        : float  CrossEntropy weight
    con_weight       : float  SupConLoss weight
    temperature      : float  SupConLoss temperature
    center_loss      : CenterLoss | None
    center_optimizer : torch.optim.SGD | None
    lambda_center    : float  center loss weight (0 → disabled)

    Returns
    -------
    avg_loss : float
    accuracy : float  (classification accuracy on first view in %)
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
            # use fc1 output (2048-d) as embedding for centre loss
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
