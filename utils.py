# ==============================================================
#  utils.py — FFT utilities, evaluation, and training functions
# ==============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import SupConLoss, GRL


# ══════════════════════════════════════════════════════════════
#  FFT STYLE AUGMENTATION HELPERS
# ══════════════════════════════════════════════════════════════

def gaussian_mask(H, W, beta):
    sigma  = min(H, W) * beta
    cy, cx = H // 2, W // 2
    ys = np.arange(H) - cy
    xs = np.arange(W) - cx
    xs, ys = np.meshgrid(xs, ys)
    return np.exp(-(xs**2 + ys**2) / (2 * sigma**2)).astype(np.float32)


def extract_style_template(img_np):
    if img_np.ndim == 2:
        return np.fft.fftshift(
            np.abs(np.fft.fft2(img_np))).astype(np.float32)
    return np.stack([
        np.fft.fftshift(np.abs(np.fft.fft2(img_np[..., c])))
        for c in range(img_np.shape[2])
    ], axis=-1).astype(np.float32)


def apply_style_template(img_np, amp_template, beta):
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


# ══════════════════════════════════════════════════════════════
#  EVALUATION  (domain-aware)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device, domain_ids_list=None):
    """
    Extract L2-normalised embeddings using model.get_embedding().

    domain_ids_list : list[int] | None
      If provided (length == len(loader.dataset)), each sample is embedded
      with its own domain expert: model.get_embedding(x, domain_id=d).
      If None, domain_id=None (base FC only).

    Returns
    -------
    feats  : np.ndarray [N, D]
    labels : np.ndarray [N]
    """
    model.eval()
    feats, labels_out = [], []
    sample_idx = 0

    for imgs, labs in loader:
        B = imgs.size(0)
        if domain_ids_list is not None:
            # group by domain_id to call get_embedding once per domain group
            batch_ids = domain_ids_list[sample_idx: sample_idx + B]
            batch_np  = torch.zeros(B, 512)   # placeholder — filled below

            unique_domains = set(batch_ids)
            for d in unique_domains:
                mask = [i for i, did in enumerate(batch_ids) if did == d]
                idx  = torch.tensor(mask, dtype=torch.long)
                emb  = model.get_embedding(imgs[idx].to(device), domain_id=d)
                # handle variable embedding dim
                if batch_np.shape[1] != emb.shape[1]:
                    batch_np = torch.zeros(B, emb.shape[1])
                batch_np[idx] = emb.cpu()

            batch_np = batch_np.numpy()
        else:
            batch_np = model.get_embedding(imgs.to(device)).cpu().numpy()

        nan_rows = np.isnan(batch_np).any(axis=1)
        if nan_rows.any():
            batch_np[nan_rows] = 0.0
        feats.append(batch_np)
        labels_out.append(labs.numpy())
        sample_idx += B


    return np.concatenate(feats), np.concatenate(labels_out)


@torch.no_grad()
def extract_features_dual(base_model, gallery_loader, probe_loader, device,
                           gallery_domain_ids, probe_domain_ids,
                           domain_expert_registry, use_whitening=False):
    """
    GlobalFull evaluation using each client's domain_gabor branch.

    For each sample with domain_id=k:
      fused = (1-w)*base_gabor(img) + w*domain_gabor_k(img)
      emb   = fc(fused)   L2-normalised

    Falls back to base-only for unknown domain_ids.
    """
    if not domain_expert_registry:
        return None, None

    from models import CompetitiveBlock

    # Materialise cb1d/cb2d/cb3d per registered client on eval device
    domain_gabors = {}    # cid -> (cb1d, cb2d, cb3d)
    for cid, state in domain_expert_registry.items():
        cb1d = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00).to(device)
        cb2d = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50).to(device)
        cb3d = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25).to(device)
        cb1d.load_state_dict({k: v.to(device) for k,v in state["cb1d_state"].items()})
        cb2d.load_state_dict({k: v.to(device) for k,v in state["cb2d_state"].items()})
        cb3d.load_state_dict({k: v.to(device) for k,v in state["cb3d_state"].items()})
        cb1d.eval(); cb2d.eval(); cb3d.eval()
        domain_gabors[cid] = (cb1d, cb2d, cb3d)

    def _extract(loader, domain_ids_list):
        base_model.eval()
        feats_out = []; labels_out = []; idx = 0
        for imgs, labs in loader:
            B   = imgs.size(0)
            img = imgs.to(device)
            # Start with GlobalBase embeddings (L2-normalised)
            emb = base_model.get_embedding(img, domain_id=None)   # [B, 512]
            if domain_ids_list is not None:
                batch_ids = domain_ids_list[idx: idx + B]
                for d in set(batch_ids):
                    if d not in domain_gabors:
                        continue
                    mask = [i for i, did in enumerate(batch_ids) if did == d]
                    msk  = torch.tensor(mask, dtype=torch.long, device=device)
                    cb1d, cb2d, cb3d = domain_gabors[d]
                    emb[msk] = base_model.get_embedding_with_external_domain(
                        img[msk], cb1d, cb2d, cb3d)   # [M, 512] L2-normed
            emb_np = emb.cpu().numpy()   # already L2-normalised
            nan_rows = np.isnan(emb_np).any(axis=1)
            if nan_rows.any(): emb_np[nan_rows] = 0.0
            feats_out.append(emb_np)
            labels_out.append(labs.numpy())
            idx += B
        return np.concatenate(feats_out), np.concatenate(labels_out)

    gal_feats, gal_labels = _extract(gallery_loader, gallery_domain_ids)
    prb_feats, prb_labels = _extract(probe_loader,   probe_domain_ids)

    if use_whitening:
        gal_feats, prb_feats = whiten_features(gal_feats, prb_feats)

    sim_matrix = np.nan_to_num(prb_feats @ gal_feats.T, nan=0.0)
    scores_list = []; labels_list = []
    for i in range(len(prb_feats)):
        for j in range(len(gal_feats)):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if prb_labels[i] == gal_labels[j] else -1)

    eer    = compute_eer(np.column_stack([scores_list, labels_list]))
    nn_idx = np.argmax(sim_matrix, axis=1)
    correct = sum(prb_labels[i] == gal_labels[nn_idx[i]]
                  for i in range(len(prb_feats)))
    rank1  = 100.0 * correct / max(len(prb_feats), 1)
    return eer, rank1

def compute_eer(scores_array):
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
    N, D = gal_feats.shape
    if n_components is None:
        n_components = min(N - 1, D, 256)
    global_mean = gal_feats.mean(axis=0)
    centred_gal = gal_feats - global_mean
    _, s, Vt    = np.linalg.svd(centred_gal, full_matrices=False)
    Vt_k  = Vt[:n_components]
    s_k   = s[:n_components]
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
                   use_whitening=False,
                   gallery_domain_ids=None, probe_domain_ids=None):
    """
    Cosine-similarity evaluation.

    gallery_domain_ids / probe_domain_ids : list[int] | None
      When provided (same length as the respective dataset), each sample is
      embedded using its domain-specific expert (base + expert[d]).
      When None, only the base FC is used.

    Returns  eer [0,1], rank1 [0,100]
    """
    gal_feats, gal_labels = extract_features(
        model, gallery_loader, device, gallery_domain_ids)
    prb_feats, prb_labels = extract_features(
        model, probe_loader,   device, probe_domain_ids)

    if use_whitening:
        gal_feats, prb_feats = whiten_features(gal_feats, prb_feats)

    sim_matrix = np.nan_to_num(prb_feats @ gal_feats.T, nan=0.0)

    scores_list, labels_list = [], []
    for i in range(len(prb_feats)):
        for j in range(len(gal_feats)):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if prb_labels[i] == gal_labels[j] else -1)

    eer    = compute_eer(np.column_stack([scores_list, labels_list]))
    nn_idx = np.argmax(sim_matrix, axis=1)
    correct = sum(prb_labels[i] == gal_labels[nn_idx[i]]
                  for i in range(len(prb_feats)))
    rank1  = 100.0 * correct / max(len(prb_feats), 1)
    return eer, rank1


# ══════════════════════════════════════════════════════════════
#  CENTER LOSS
# ══════════════════════════════════════════════════════════════

class CenterLoss(nn.Module):
    """
    Center Loss — minimises distance between embeddings and per-client class centres.
    Centres are kept local and never shared with the server.
    """
    def __init__(self, num_classes, embed_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.centres = nn.Parameter(
            torch.zeros(num_classes, embed_dim, device=device),
            requires_grad=True,
        )

    def forward(self, embeddings, labels):
        centres_batch = self.centres[labels]
        return ((embeddings - centres_batch) ** 2).sum(dim=1).mean()


# ══════════════════════════════════════════════════════════════
#  TRAINING  (one epoch)
# ══════════════════════════════════════════════════════════════

def train_compnet_epoch(model, loader, criterion, optimizer, device,
                        center_loss=None, center_optimizer=None,
                        lambda_center=0.0, lambda_style=0.0,
                        lambda_grl=0.0, lambda_load_balance=0.0,
                        lambda_supcon=0.0, temperature=0.07,
                        collect_grad_norms=False,
                        lambda_domain_recon=0.0):
    """
    Train CompNet (or DINOv2) for one epoch.

    Asymmetric MoE routing is automatic: domain_ids from the batch are
    passed directly to model._backbone() / MoEFC.forward(), which calls
    only expert[domain_id] per sample. No other change needed here.

    Gradient norms are collected on the LAST batch of the epoch when
    collect_grad_norms=True (avoids redundant per-batch overhead).

    Returns
    -------
    avg_loss        : float
    accuracy        : float  (%)
    last_grad_norms : dict | None   — from model.get_grad_norms()
    avg_recon_loss  : float         — avg domain reconstruction loss (0 if disabled)
    """
    model.train()
    if center_loss is not None:
        center_loss.train()

    use_center = (center_loss is not None and center_optimizer is not None
                  and lambda_center > 0.0)
    use_style  = lambda_style  > 0.0
    use_supcon = lambda_supcon > 0.0
    use_grl    = (lambda_grl > 0.0 and hasattr(model, "use_grl")
                  and model.use_grl and model.domain_classifier is not None)

    supcon_criterion = (SupConLoss(temperature=temperature,
                                   base_temperature=temperature)
                        if use_supcon else None)

    def _get_emb(imgs, domain_ids=None):
        if hasattr(model, "_backbone"):
            try:
                r = model._backbone(imgs, domain_ids)
                return r[0] if isinstance(r, tuple) else r
            except TypeError:
                r = model._backbone(imgs)
                return r[0] if isinstance(r, tuple) else r
        return model.backbone(imgs)

    def _get_backbone_full(imgs, domain_ids=None):
        """Returns (emb, base_feat, domain_feat, real_mask)."""
        if hasattr(model, "_backbone"):
            try:
                r = model._backbone(imgs, domain_ids)
                if isinstance(r, tuple) and len(r) == 4:
                    return r   # (emb, base_feat, domain_feat, real_mask)
                if isinstance(r, tuple):
                    return r[0], None, None, None
                return r, None, None, None
            except TypeError:
                r = model._backbone(imgs)
                return (r[0],None,None,None) if isinstance(r,tuple) else (r,None,None,None)
        return model.backbone(imgs), None, None, None

    def _get_logits(emb, labels):
        if hasattr(model, "drop"):
            return model.arc(model.drop(emb), labels)
        return model.arc(emb, labels)

    running_loss = 0.0; correct = 0; total = 0
    running_recon = 0.0
    use_recon = (lambda_domain_recon > 0.0
                 and hasattr(model, 'compute_domain_recon_loss')
                 and hasattr(model, 'use_moe') and getattr(model, 'use_moe', False)
                 and not getattr(model, 'moe_is_warming_up', True))
    last_grad_norms = None
    batches = list(loader)
    n_batches = len(batches)

    for batch_idx, batch in enumerate(batches):
        is_last = (batch_idx == n_batches - 1)

        if len(batch) == 3:
            data, labels, domain_ids = batch
            domain_ids = domain_ids.to(device)
        else:
            data, labels = batch
            domain_ids   = None

        labels = labels.to(device)

        if isinstance(data, (list, tuple)):
            orig = data[0].to(device); aug = data[1].to(device); has_aug = True
        else:
            orig = data.to(device); aug = None; has_aug = False

        optimizer.zero_grad()
        if use_center:
            center_optimizer.zero_grad()

        if use_recon and domain_ids is not None:
            emb_orig, base_feat, domain_feat, real_mask = \
                _get_backbone_full(orig, domain_ids)
        else:
            base_feat = domain_feat = real_mask = None
            emb_orig  = _get_emb(orig, domain_ids)

        logits   = _get_logits(emb_orig, labels)
        loss     = criterion(logits, labels)

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
            feats = torch.stack([emb_o, emb_a], dim=1)
            loss  = loss + lambda_supcon * supcon_criterion(feats, labels)

        if use_grl and domain_ids is not None:
            # Filter out sentinel samples (domain_id == -1, FFT-only, no expert)
            # GRL domain classification only makes sense for real domain samples.
            real_mask = (domain_ids >= 0)
            if real_mask.any():
                emb_grl    = GRL.apply(emb_orig[real_mask], lambda_grl)
                dom_logits = model.domain_classifier(emb_grl)
                loss_dom   = F.cross_entropy(dom_logits, domain_ids[real_mask])
                loss       = loss + lambda_grl * loss_dom

        # Domain reconstruction loss
        if use_recon and base_feat is not None and domain_feat is not None:
            l_recon = model.compute_domain_recon_loss(
                base_feat.detach(), domain_feat, real_mask)
            loss           = loss + lambda_domain_recon * l_recon
            running_recon += float(l_recon) * orig.size(0)

        if use_center:
            loss = loss + lambda_center * center_loss(emb_orig.detach(), labels)

        loss.backward()

        # collect grad norms on last batch (after backward, before step)
        if collect_grad_norms and is_last and hasattr(model, "get_grad_norms"):
            last_grad_norms = model.get_grad_norms()

        optimizer.step()

        if use_center:
            for p in center_loss.parameters():
                if p.grad is not None:
                    p.grad.data *= 1.0 / (lambda_center + 1e-8)
            center_optimizer.step()

        running_loss += loss.item() * orig.size(0)
        correct      += logits.argmax(1).eq(labels).sum().item()
        total        += orig.size(0)

    return (running_loss / max(total, 1),
            100.0 * correct / max(total, 1),
            last_grad_norms,
            running_recon / max(total, 1))


def train_ccnet_epoch(model, loader, criterion, optimizer, device,
                      ce_weight=0.8, con_weight=0.2, temperature=0.07,
                      center_loss=None, center_optimizer=None,
                      lambda_center=0.0):
    """Train CCNet for one epoch. Returns (avg_loss, accuracy, None)."""
    con_criterion = SupConLoss(temperature=temperature,
                               base_temperature=temperature)
    model.train()
    if center_loss is not None:
        center_loss.train()
    use_center = (center_loss is not None and center_optimizer is not None
                  and lambda_center > 0.0)

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
                emb1 = model.fc1(model.fc(model._extract(img1)))
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

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1), None, 0.0



# ══════════════════════════════════════════════════════════════
#  DOMAIN PREDICTOR UTILITIES
# ══════════════════════════════════════════════════════════════

def extract_dp_features(img_np, pool_size=16, mode="style"):
    """
    Extract FFT-based features for domain predictor.
    mode="full"  → full FFT amplitude
    mode="style" → low-frequency only (Gaussian-masked)
    Both pooled to pool_size×pool_size.
    """
    if img_np.ndim == 3:
        img_np = img_np.mean(axis=-1)
    fft_amp = np.abs(np.fft.fftshift(np.fft.fft2(img_np)))
    if mode == "style":
        H, W = fft_amp.shape
        mask = gaussian_mask(H, W, beta=0.3)
        fft_amp = fft_amp * mask
    fft_amp = np.log1p(fft_amp)
    from PIL import Image as PILImage
    fft_pil = PILImage.fromarray(fft_amp.astype(np.float32))
    fft_pil = fft_pil.resize((pool_size, pool_size), PILImage.BILINEAR)
    features = np.array(fft_pil, dtype=np.float32)
    features = (features - features.mean()) / (features.std() + 1e-6)
    return features


def build_dp_dataset(style_bank, client_ids, pool_size=16, mode="style"):
    """Build training dataset for domain predictor from style bank."""
    features, labels = [], []
    for label_idx, cid in enumerate(client_ids):
        templates = style_bank[cid]
        for tmpl in templates:
            if mode == "style":
                H, W = tmpl.shape[:2]
                mask = gaussian_mask(H, W, beta=0.3)
                feat = tmpl * mask if tmpl.ndim == 2 else tmpl[..., 0] * mask
            else:
                feat = tmpl if tmpl.ndim == 2 else tmpl[..., 0]
            feat = np.log1p(feat)
            from PIL import Image as PILImage
            feat_pil = PILImage.fromarray(feat.astype(np.float32))
            feat_pil = feat_pil.resize((pool_size, pool_size), PILImage.BILINEAR)
            feat_arr = np.array(feat_pil, dtype=np.float32)
            feat_arr = (feat_arr - feat_arr.mean()) / (feat_arr.std() + 1e-6)
            features.append(feat_arr.flatten())
            labels.append(label_idx)
    return np.array(features), np.array(labels)


def train_domain_predictor(predictor, features, labels, cfg, device="cuda"):
    """Train domain predictor. Returns: predictor, accuracy."""
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.tensor(features, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=cfg["dp_batch_size"], shuffle=True)

    predictor = predictor.to(device)
    opt = torch.optim.Adam(predictor.parameters(), lr=cfg["dp_lr"])
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(cfg["dp_epochs"]):
        predictor.train()
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = predictor(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        acc = 100.0 * correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
        if (epoch + 1) % 20 == 0 or epoch == cfg["dp_epochs"] - 1:
            print(f"    [DP] ep {epoch+1:3d}/{cfg['dp_epochs']}  acc={acc:.1f}%")
    print(f"    [DP] Best accuracy: {best_acc:.1f}%")
    predictor.eval()
    return predictor, best_acc


# ══════════════════════════════════════════════════════════════
#  PER-SAMPLE DOMAIN PREDICTION + SOFT ROUTING
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_alpha_per_sample(domain_predictor, images, n_clients,
                              pool_size=16, mode="style", device="cuda"):
    """
    Predict per-sample domain probabilities.
    Returns: probs [B, K] softmax over K domains.
    """
    domain_predictor.eval()
    features = []
    for i in range(images.size(0)):
        img_np = images[i].cpu().numpy()
        if img_np.ndim == 3:
            img_np = img_np.mean(axis=0) if img_np.shape[0] <= 4 else img_np.mean(axis=-1)
        feat = extract_dp_features(img_np, pool_size, mode)
        features.append(feat.flatten())
    X = torch.tensor(np.array(features), dtype=torch.float32).to(device)
    logits = domain_predictor(X)
    return torch.softmax(logits, dim=1)  # [B, K]


@torch.no_grad()
def embed_with_routing(local_models, global_model, images, domain_ids,
                        domain_predictor, cfg, device="cuda"):
    """
    Per-sample soft routing between local and global models.

    For each sample i:
      - domain_ids[i] = true spectrum index (from dataset metadata)
      - If dp_mode == "ideal": use true domain_id directly
      - If dp_mode == "predicted": use domain predictor

      α_i = P(domain == domain_ids[i] | sample_i)
        → high α: sample is from a known client domain → use that client's local model
        → low α: sample is from unknown domain → use global model

      emb_i = α_i · local_models[predicted_domain].get_embedding(x_i)
            + (1-α_i) · global_model.get_embedding(x_i)

    Returns: embeddings [B, D]
    """
    B = images.size(0)
    n_clients = len(local_models)
    dp_mode = cfg.get("dp_mode", "ideal")

    # Get global embeddings for all samples
    emb_global = global_model.get_embedding(images)  # [B, D]
    D = emb_global.size(1)

    if dp_mode == "ideal":
        # Oracle: use true domain_id
        # α = 1.0 for the matching client, route fully to local model
        emb_fused = torch.zeros_like(emb_global)
        for i in range(B):
            did = domain_ids[i].item() if torch.is_tensor(domain_ids[i]) else domain_ids[i]
            if 0 <= did < n_clients:
                local_emb = local_models[did].get_embedding(
                    images[i:i+1])  # [1, D]
                emb_fused[i] = local_emb.squeeze(0)
            else:
                emb_fused[i] = emb_global[i]

    elif dp_mode == "predicted":
        # Use domain predictor for per-sample α
        probs = predict_alpha_per_sample(
            domain_predictor, images, n_clients,
            pool_size=cfg["dp_pool_size"],
            mode=cfg["dp_input"], device=device)  # [B, K]

        # For each sample, find best matching client
        best_domain = probs.argmax(dim=1)  # [B]
        alpha = probs.max(dim=1).values    # [B] confidence

        emb_fused = torch.zeros_like(emb_global)
        for i in range(B):
            did = best_domain[i].item()
            a = alpha[i].item()
            local_emb = local_models[did].get_embedding(
                images[i:i+1]).squeeze(0)
            emb_fused[i] = a * local_emb + (1.0 - a) * emb_global[i]
    else:
        emb_fused = emb_global

    return F.normalize(emb_fused, p=2, dim=1)


@torch.no_grad()
def extract_embeddings_routed(local_models, global_model, loader,
                               domain_predictor, cfg, device="cuda",
                               mode="moe"):
    """
    Extract embeddings from a loader using specified mode.

    mode="global": global model only
    mode="local":  per-sample local model (using true domain_id)
    mode="moe":    soft routing between local and global

    Returns: feats [N, D], labels [N]
    """
    feats_list, labels_list = [], []
    n_clients = len(local_models)

    for batch in loader:
        if len(batch) == 3:
            imgs, labels, domain_ids = batch
        elif len(batch) == 2:
            imgs, labels = batch
            domain_ids = torch.zeros(imgs.size(0), dtype=torch.long)
        imgs = imgs.to(device)

        if mode == "global":
            emb = global_model.get_embedding(imgs)
            emb = F.normalize(emb, p=2, dim=1)

        elif mode == "local":
            # Each sample uses its own domain's local model
            emb = torch.zeros(imgs.size(0), 512, device=device)
            for i in range(imgs.size(0)):
                did = domain_ids[i].item() if torch.is_tensor(domain_ids[i]) else domain_ids[i]
                if 0 <= did < n_clients:
                    emb[i] = local_models[did].get_embedding(
                        imgs[i:i+1]).squeeze(0)
                else:
                    emb[i] = global_model.get_embedding(
                        imgs[i:i+1]).squeeze(0)
            emb = F.normalize(emb, p=2, dim=1)

        elif mode == "moe":
            emb = embed_with_routing(
                local_models, global_model, imgs, domain_ids,
                domain_predictor, cfg, device)

        feats_list.append(emb.cpu())
        labels_list.append(labels)

    return torch.cat(feats_list), torch.cat(labels_list)


def evaluate_all_modes(local_models, global_model, domain_predictor,
                        gallery_loader, probe_loader, cfg, device="cuda"):
    """
    Evaluate all three modes: global, local, moe.
    Returns dict with results for each mode.
    """
    results = {}

    for mode in ["global", "local", "moe"]:
        gal_feats, gal_labels = extract_embeddings_routed(
            local_models, global_model, gallery_loader,
            domain_predictor, cfg, device, mode=mode)
        prb_feats, prb_labels = extract_embeddings_routed(
            local_models, global_model, probe_loader,
            domain_predictor, cfg, device, mode=mode)

        # Rank-1
        sim = prb_feats @ gal_feats.T
        top_idx = sim.argmax(dim=1)
        predicted = gal_labels[top_idx]
        rank1 = (predicted == prb_labels).float().mean().item() * 100

        # EER
        genuine, impostor = [], []
        for i in range(len(prb_labels)):
            pid = prb_labels[i].item()
            sims = sim[i].numpy()
            glabs = gal_labels.numpy()
            gen_mask = glabs == pid
            imp_mask = glabs != pid
            if gen_mask.any():
                genuine.extend(sims[gen_mask].tolist())
            if imp_mask.any():
                impostor.extend(sims[imp_mask].tolist())

        genuine = np.array(genuine); impostor = np.array(impostor)
        all_labels = np.concatenate([np.ones(len(genuine)),
                                      np.zeros(len(impostor))])
        all_scores = np.concatenate([genuine, impostor])
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2) * 100

        results[mode] = {
            "rank1": rank1, "eer": eer,
            "n_gallery": len(gal_labels), "n_probe": len(prb_labels),
        }

    return results
