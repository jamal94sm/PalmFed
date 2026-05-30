# ==============================================================
#  models.py — CompNet and CCNet architectures
# ==============================================================
#
#  DualExpertFC redesign:
#
#  domain_expert now operates in 9708-d Gabor FEATURE SPACE (not embedding space).
#  It outputs a 9708-d residual that corrects the Gabor features before
#  the base_expert projects them to 512-d. Combination uses a fixed scalar weight:
#
#    corrected_feat = gabor_feat + expert_weight * domain_expert(gabor_feat)
#    emb            = base_expert(corrected_feat)
#
#  Benefits over the previous embedding-space residual:
#    1. Domain signal lives in Gabor feature space (spectral/lighting differences
#       are directly encoded there). The correction is geometrically meaningful.
#    2. GlobalFull mismatch eliminated: the domain correction is always composed
#       with whatever base_expert is active — local or FedAvg'd global.
#    3. The domain reconstruction loss has a natural target in this space.
#
#  Domain reconstruction loss (computed in utils.train_compnet_epoch):
#    For real own-domain samples in a batch:
#      overall_mean  = mean(gabor_feat over all samples in batch)
#      domain_signal = mean(gabor_feat[real_mask])
#      target        = domain_signal - overall_mean   [domain-specific offset]
#      L_recon = MSE(domain_expert(real_feats), target.expand_as(real_feats))
#
#    This gives domain_expert an exclusive self-supervised task: predict the
#    average domain-specific deviation of the Gabor features from the global
#    batch mean. The base_expert cannot optimise this (it sees all domains and
#    its gradient averages out the domain signal). The domain_expert is rewarded
#    for being domain-homogeneous and identity-invariant — the right inductive bias.
#
# ==============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ══════════════════════════════════════════════════════════════
#  COMPNET BUILDING BLOCKS
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
        y  = y0.view(1, -1).repeat(c_out, c_in, ksz, 1)
        x  = x0.view(-1, 1).repeat(c_out, c_in, 1, ksz)
        x  = x.to(sigma.device); y = y.to(sigma.device)
        xt =  x * torch.cos(theta.view(-1,1,1,1)) + y * torch.sin(theta.view(-1,1,1,1))
        yt = -x * torch.sin(theta.view(-1,1,1,1)) + y * torch.cos(theta.view(-1,1,1,1))
        gb = -torch.exp(
            -0.5 * ((gamma * xt)**2 + yt**2) / (8 * sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2 * math.pi * f.view(-1,1,1,1) * xt + psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2, 3], keepdim=True)

    def forward(self, x):
        self.kernel = self._gen_bank(
            self.kernel_size, self.channel_in, self.channel_out,
            self.sigma, self.gamma, self.theta, self.f, self.psi)
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
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

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
            return self.s * ((one_hot * phi) + ((1.0 - one_hot) * cosine))
        return self.s * cosine


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.clone()
    @staticmethod
    def backward(ctx, grad): return -ctx.lam * grad, None


class DomainClassifier(nn.Module):
    def __init__(self, in_features, n_domains):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(),
            nn.Linear(256, n_domains))
    def forward(self, x): return self.net(x)


class _FeatureResidualExpert(nn.Module):
    """
    Low-rank residual adapter operating in 9708-d Gabor FEATURE space.

    Architecture: A(9708→rank) → ReLU → B(rank→9708)
    Output is a 9708-d correction to the Gabor feature vector.

    Unlike the previous embedding-space residual (9708→rank→512), this
    module outputs in the same space as its input, making it interpretable
    as a domain-specific feature correction.

    B initialised to zero → zero correction at round 0 (safe start).
    A initialised Kaiming → gradient flows from the first batch.
    """
    def __init__(self, in_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, in_features, bias=False)
        nn.init.kaiming_normal_(self.A.weight, nonlinearity="relu")
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(F.relu(self.A(x)))


# ══════════════════════════════════════════════════════════════
#  DUAL-EXPERT FC BOTTLENECK  (feature-space domain correction)
# ══════════════════════════════════════════════════════════════

class DualExpertFC(nn.Module):
    """
    Two-expert FC bottleneck with feature-space domain correction.

    base_expert   : nn.Linear(9708→512)
      • Receives gradient from ALL samples (real + FFT sentinel).
      • FedAvg'd — learns the domain-invariant projection.

    domain_expert : _FeatureResidualExpert(9708→rank→9708)
      • Receives gradient ONLY from real own-domain samples
        (domain_id >= 0, NOT the FFT sentinel -1).
      • NEVER FedAvg'd — permanently local.
      • Learns a domain-specific FEATURE-SPACE correction.

    expert_weight : fixed scalar w in [0, 1]   (from config, not learned)
      • Controls how strongly the domain correction is applied.
      • Fixed rather than learned because a learned gate had no gradient
        pressure to open (the ArcFace loss does not specifically reward
        the domain_expert's contribution).

    Forward (training):
      gabor_feat  [B, 9708]
      ├─ real samples (domain_id >= 0):
      │    corrected = gabor_feat + expert_weight * domain_expert(gabor_feat)
      │    emb       = base_expert(corrected)
      └─ sentinel samples (domain_id == -1):
           emb = base_expert(gabor_feat)     [domain_expert skipped]

    Domain reconstruction loss (computed externally in train_compnet_epoch):
      On real samples in each batch:
        overall_mean  = mean(gabor_feat, dim=0)         [9708] all samples
        domain_mean   = mean(gabor_feat[real], dim=0)   [9708] domain centroid
        target        = domain_mean - overall_mean       [9708] domain-specific offset
        prediction    = domain_expert(gabor_feat[real])  [R, 9708]
        L_recon = MSE(prediction, target.unsqueeze(0).expand_as(prediction))

      The domain_expert is rewarded for capturing what makes domain k
      different from the global average — a purely domain-specific signal.

    Inference:
      embed(feat, domain_id=int) → base(feat + w * domain_expert(feat))
      embed(feat, domain_id=None) → base(feat)               [GlobalBase]

    GlobalFull inference (external expert from another client's local model):
      embed_with_external_expert(feat, ext_expert, ext_weight)
      → base(feat + ext_weight * ext_expert(feat))
      The mismatch of the previous design is eliminated: domain correction
      is in feature space, so it composes correctly with any base_expert.
    """

    SENTINEL = -1

    def __init__(self, in_features=9708, out_features=512, rank=64,
                 expert_weight=0.5):
        super().__init__()
        self.in_features   = in_features
        self.out_features  = out_features
        self.expert_weight = expert_weight   # fixed, not learned

        # ── shared (FedAvg'd) ──────────────────────────────────────────────
        self.base_expert = nn.Linear(in_features, out_features)

        # ── local-only (never FedAvg'd) ───────────────────────────────────
        self.domain_expert = _FeatureResidualExpert(in_features, rank)

        self._warmup = False
        self.register_buffer("_real_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_skip_count", torch.tensor(0, dtype=torch.long))

    # ── warmup ──────────────────────────────────────────────────────────────

    def set_warmup(self, warmup: bool):
        self._warmup = warmup
        for p in self.domain_expert.parameters():
            p.requires_grad = not warmup

    @property
    def is_warming_up(self): return self._warmup

    # ── routing stats ────────────────────────────────────────────────────────

    def reset_routing_stats(self): self._real_count.zero_(); self._skip_count.zero_()

    def get_routing_stats(self):
        return {"real": int(self._real_count), "skip": int(self._skip_count)}

    # ── forward (training) ───────────────────────────────────────────────────

    def forward(self, x, domain_ids=None):
        """
        x          : Tensor [B, 9708]  — Gabor features
        domain_ids : Tensor [B] int | None
          >= 0  → real own-domain sample → feature corrected by domain_expert
          == -1 → FFT sentinel           → base_expert only
          None  → warmup                 → base_expert only

        Returns emb [B, 512] — base_expert applied to (optionally corrected) features.
        Also returns domain_expert_outputs [B, 9708] (only for real samples, else None)
        for use in the domain reconstruction loss.
        """
        if self._warmup or domain_ids is None:
            return self.base_expert(x), None

        real_mask = (domain_ids >= 0)
        skip_mask = ~real_mask

        # start with a copy of x; real samples will be corrected in place
        corrected = x.clone()
        domain_expert_out = None   # will hold domain_expert predictions for real

        if real_mask.any():
            raw_correction          = self.domain_expert(x[real_mask])   # [R, 9708]
            corrected[real_mask]    = x[real_mask] + self.expert_weight * raw_correction
            domain_expert_out       = raw_correction  # returned for recon loss
            if self.training:
                self._real_count += real_mask.sum().item()

        if skip_mask.any() and self.training:
            self._skip_count += skip_mask.sum().item()

        emb = self.base_expert(corrected)
        return emb, domain_expert_out   # domain_expert_out is None if no real samples

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed(self, feat, domain_id=None):
        """Single-domain inference."""
        if domain_id is None or self._warmup:
            return self.base_expert(feat)
        correction = self.domain_expert(feat)
        return self.base_expert(feat + self.expert_weight * correction)

    @torch.no_grad()
    def embed_with_external_expert(self, feat, ext_domain_expert, ext_weight):
        """
        GlobalFull inference with an external client's domain_expert.
        Feature-space correction composes correctly with any base_expert
        because it operates before the base projection.
        """
        correction = ext_domain_expert(feat)
        return self.base_expert(feat + ext_weight * correction)

    # ── gradient norms ───────────────────────────────────────────────────────

    def get_grad_norms(self):
        def _n(p): return round(float(p.grad.norm()), 4) if p.grad is not None else None
        return {
            "base_grad_norm"  : _n(self.base_expert.weight),
            "domain_grad_norm": _n(self.domain_expert.B.weight),
            "gate_grad_norm"  : None,   # no learned gate in this design
        }

    # ── weight-space diagnostics ─────────────────────────────────────────────

    @torch.no_grad()
    def get_weight_diagnostics(self):
        return {
            "base_weight_norm"  : round(float(self.base_expert.weight.norm()), 4),
            "domain_weight_norm": round(float(self.domain_expert.B.weight.norm()), 4),
            "gate_value"        : self.expert_weight,   # fixed, for display
        }

    # ── activation-space diagnostics ─────────────────────────────────────────

    @torch.no_grad()
    def get_activation_diagnostics(self, feat):
        """
        feat : Tensor [N, 9708]  — fixed probe Gabor features.

        Returns per-sample stats measuring how much the domain_expert
        shifts the features and how much the base_expert output changes.
        """
        feat      = feat.float()
        correction = self.domain_expert(feat)              # [N, 9708]
        corrected  = feat + self.expert_weight * correction
        base_only  = self.base_expert(feat)                # [N, 512]
        base_full  = self.base_expert(corrected)           # [N, 512]

        # how large is the feature-space correction relative to the features
        feat_norm   = float(feat.norm(dim=1).mean())
        corr_norm   = float((self.expert_weight * correction).norm(dim=1).mean())
        feat_ratio  = corr_norm / max(feat_norm, 1e-8)

        # how much does the embedding direction change
        cos_sim = float(F.cosine_similarity(
            F.normalize(base_only, dim=1),
            F.normalize(base_full, dim=1)).mean())

        # how large is the final embedding difference
        base_emb_norm = float(base_only.norm(dim=1).mean())
        diff_norm     = float((base_full - base_only).norm(dim=1).mean())
        emb_ratio     = diff_norm / max(base_emb_norm, 1e-8)

        return {
            "base_norm"           : round(float(base_only.norm(dim=1).mean()), 4),
            "feat_correction_norm": round(corr_norm, 4),
            "feat_correction_ratio": round(feat_ratio, 4),  # corr / feat magnitude
            "emb_diff_norm"       : round(diff_norm, 4),
            "gated_base_ratio"    : round(emb_ratio, 4),    # embedding shift ratio
            "base_full_cos_sim"   : round(cos_sim, 4),
            "gate_value"          : self.expert_weight,
            "domain_gated_norm"   : round(corr_norm, 4),   # alias for printer compat
        }

    # ── domain reconstruction loss ───────────────────────────────────────────

    def compute_domain_recon_loss(self, gabor_feat, domain_ids):
        """
        Compute the domain reconstruction loss for real samples in the batch.

        Theory
        ──────
        The domain_expert should predict the domain-specific component of
        the Gabor features — i.e. what makes domain k different from all
        other domains in feature space.

        We approximate this as:
          target = domain_mean - overall_mean

        where:
          overall_mean = mean of all gabor_feat in this batch [9708]
                         ≈ domain-invariant features (multiple domains averaged)
          domain_mean  = mean of real own-domain features [9708]
                         ≈ domain k centroid (identity variation averages out)
          target       = domain-specific offset [9708]

        The domain_expert's output for any individual sample of domain k
        should approximate this target. This is identity-invariant by
        construction (the target is a per-batch constant shared by all
        real samples) and domain-specific (it captures what domain k looks
        like relative to the global average).

        Why the base_expert cannot optimise this:
          The base_expert sees all domains and its gradient is pulled in
          multiple directions simultaneously. Its loss (ArcFace classification)
          rewards identity discriminability, not domain homogeneity.
          The reconstruction loss explicitly rewards the domain_expert for
          being homogeneous across identities within a domain — the exact
          opposite of the identity-discriminative ArcFace task.

        Parameters
        ──────────
        gabor_feat : Tensor [B, 9708]  — raw Gabor features, before correction
        domain_ids : Tensor [B]        — -1 for sentinel, >=0 for real

        Returns
        ───────
        loss : scalar Tensor  — MSE reconstruction loss (0.0 if no real samples)
        target_norm : float   — norm of the target (for diagnostics)
        """
        real_mask = (domain_ids >= 0)
        if not real_mask.any():
            return torch.tensor(0.0, device=gabor_feat.device, requires_grad=False), 0.0

        real_feats   = gabor_feat[real_mask]               # [R, 9708]
        overall_mean = gabor_feat.mean(dim=0)              # [9708]  all samples
        domain_mean  = real_feats.mean(dim=0)              # [9708]  domain centroid
        target       = (domain_mean - overall_mean).detach()  # [9708] stop-grad on target

        prediction   = self.domain_expert(real_feats)     # [R, 9708]
        target_exp   = target.unsqueeze(0).expand_as(prediction)

        loss = F.mse_loss(prediction, target_exp)
        return loss, float(target.norm())


# ══════════════════════════════════════════════════════════════
#  COMPNET  (with DualExpertFC)
# ══════════════════════════════════════════════════════════════

class CompNet(nn.Module):
    """
    CompNet = CB1//CB2//CB3 + DualExpertFC(9708→[9708]→512) + Dropout + ArcFace.

    The domain_expert now corrects in Gabor feature space (9708-d) before
    the base_expert projects to 512-d. Fixed expert_weight controls the blend.

    Weight management
    ─────────────────
    FedAvg'd  : fc.base_expert.* + all Gabor/CB parameters
    Local-only : arc.*, fc.domain_expert.*
    (No gate parameters — expert_weight is a fixed config scalar.)
    """
    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25,
                 use_moe=False, lora_rank=64,
                 use_grl=False, n_domains=6,
                 expert_weight=0.5,
                 # legacy args kept for compat — ignored
                 gate_mode=None, gate_init=None, gate_scale=None, n_experts=None):
        super().__init__()
        self.cb1 = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2 = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3 = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)
        self.use_moe = use_moe
        if use_moe:
            self.fc = DualExpertFC(9708, embedding_dim, lora_rank,
                                   expert_weight=expert_weight)
        else:
            self.fc = nn.Linear(9708, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.arc  = ArcMarginProduct(embedding_dim, num_classes,
                                     s=arcface_s, m=arcface_m)
        self.use_grl = use_grl
        self.domain_classifier = (DomainClassifier(embedding_dim, n_domains)
                                   if use_grl else None)

    def local_only_keys(self):
        """Keys never sent to server: local ArcFace + local domain_expert."""
        return ("arc.", "fc.domain_expert.")

    def set_moe_warmup(self, warmup: bool):
        if self.use_moe: self.fc.set_warmup(warmup)

    @property
    def moe_is_warming_up(self):
        return self.use_moe and self.fc.is_warming_up

    def _gabor_feat(self, x):
        return torch.cat([self.cb1(x).flatten(1),
                          self.cb2(x).flatten(1),
                          self.cb3(x).flatten(1)], dim=1)   # [B, 9708]

    def _backbone(self, x, domain_ids=None):
        """
        Returns (emb, domain_expert_out).
        domain_expert_out is None when not MoE or warmup or no real samples.
        Used by train_compnet_epoch to compute the reconstruction loss.
        """
        feat = self._gabor_feat(x)
        if self.use_moe:
            emb, de_out = self.fc(feat, domain_ids)
            return emb, de_out
        return self.fc(feat), None

    def forward(self, x, y=None, domain_ids=None):
        emb, _ = self._backbone(x, domain_ids)
        return self.arc(self.drop(emb), y)

    @torch.no_grad()
    def get_embedding(self, x, domain_id=None):
        feat = self._gabor_feat(x)
        if self.use_moe:
            return F.normalize(self.fc.embed(feat, domain_id), p=2, dim=1)
        return F.normalize(self.fc(feat), p=2, dim=1)

    @torch.no_grad()
    def get_embedding_with_external_expert(self, x, ext_domain_expert, ext_weight):
        feat = self._gabor_feat(x)
        raw  = self.fc.embed_with_external_expert(feat, ext_domain_expert, ext_weight)
        return F.normalize(raw, p=2, dim=1)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def reset_routing_stats(self):
        if self.use_moe: self.fc.reset_routing_stats()
    def get_routing_stats(self):
        return self.fc.get_routing_stats() if self.use_moe else None
    def get_grad_norms(self):
        return self.fc.get_grad_norms() if self.use_moe else None
    def get_weight_diagnostics(self):
        return self.fc.get_weight_diagnostics() if self.use_moe else None
    def get_activation_diagnostics(self, probe_feat):
        return self.fc.get_activation_diagnostics(probe_feat) if self.use_moe else None
    def compute_domain_recon_loss(self, gabor_feat, domain_ids):
        if not self.use_moe: return torch.tensor(0.0), 0.0
        return self.fc.compute_domain_recon_loss(gabor_feat, domain_ids)


# ══════════════════════════════════════════════════════════════
#  CCNET
# ══════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature; self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 3: raise ValueError('features needs [bsz, n_views, ...]')
        if len(features.shape) > 3: features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None: raise ValueError('Cannot define both')
        elif labels is None and mask is None: mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask   = torch.eq(labels, labels.T).float().to(device)
        else: mask = mask.float().to(device)
        contrast_count   = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one': anchor_feature = features[:, 0]; anchor_count = 1
        elif self.contrast_mode == 'all': anchor_feature = contrast_feature; anchor_count = contrast_count
        else: raise ValueError(f'Unknown mode: {self.contrast_mode}')
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob   = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -(self.temperature / self.base_temperature) * mean_log_prob_pos.view(anchor_count, batch_size).mean()


class CCGaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in=channel_in; self.channel_out=channel_out; self.kernel_size=kernel_size
        self.stride=stride; self.padding=padding; self.init_ratio=init_ratio if init_ratio>0 else 1.0; self.kernel=0
        _S=9.2*self.init_ratio; _F=0.057/self.init_ratio; _G=2.0
        self.gamma=nn.Parameter(torch.FloatTensor([_G]),requires_grad=True)
        self.sigma=nn.Parameter(torch.FloatTensor([_S]),requires_grad=True)
        self.theta=nn.Parameter(torch.arange(0,channel_out).float()*math.pi/channel_out,requires_grad=False)
        self.f=nn.Parameter(torch.FloatTensor([_F]),requires_grad=True)
        self.psi=nn.Parameter(torch.FloatTensor([0]),requires_grad=False)
    def genGaborBank(self,kernel_size,channel_in,channel_out,sigma,gamma,theta,f,psi):
        xmax=kernel_size//2; xmin=-xmax; ksize=xmax-xmin+1
        y_0=torch.arange(xmin,xmax+1).float(); x_0=torch.arange(xmin,xmax+1).float()
        y=y_0.view(1,-1).repeat(channel_out,channel_in,ksize,1); x=x_0.view(-1,1).repeat(channel_out,channel_in,1,ksize)
        x=x.to(sigma.device); y=y.to(sigma.device)
        xt=x*torch.cos(theta.view(-1,1,1,1))+y*torch.sin(theta.view(-1,1,1,1))
        yt=-x*torch.sin(theta.view(-1,1,1,1))+y*torch.cos(theta.view(-1,1,1,1))
        gb=-torch.exp(-0.5*((gamma*xt)**2+yt**2)/(8*sigma.view(-1,1,1,1)**2))*torch.cos(2*math.pi*f.view(-1,1,1,1)*xt+psi.view(-1,1,1,1))
        return gb-gb.mean(dim=[2,3],keepdim=True)
    def forward(self,x):
        k=self.genGaborBank(self.kernel_size,self.channel_in,self.channel_out,self.sigma,self.gamma,self.theta,self.f,self.psi)
        self.kernel=k; return F.conv2d(x,k,stride=self.stride,padding=self.padding)


class SELayer(nn.Module):
    def __init__(self,channel,reduction=1):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(channel,channel//reduction,bias=False),nn.ReLU(inplace=True),nn.Linear(channel//reduction,channel,bias=False),nn.Sigmoid())
    def forward(self,x):
        b,c,_,_=x.size(); y=self.avg_pool(x).view(b,c); return x*self.fc(y).view(b,c,1,1).expand_as(x)


class CompetitiveBlock_Mul_Ord_Comp(nn.Module):
    def __init__(self,channel_in,n_competitor,ksize,stride,padding,weight,init_ratio=1,o1=32,o2=12):
        super().__init__()
        self.gabor1=CCGaborConv2d(channel_in,n_competitor,ksize,stride=2,padding=ksize//2,init_ratio=init_ratio)
        self.gabor2=CCGaborConv2d(n_competitor,n_competitor,ksize,stride=2,padding=ksize//2,init_ratio=init_ratio)
        self.argmax=nn.Softmax(dim=1); self.argmax_x=nn.Softmax(dim=2); self.argmax_y=nn.Softmax(dim=3)
        self.conv1=nn.Conv2d(n_competitor,o1//2,5,2,0); self.conv2=nn.Conv2d(n_competitor,o1//2,5,2,0)
        self.pool=nn.MaxPool2d(2,2); self.se1=SELayer(n_competitor); self.se2=SELayer(n_competitor)
        self.wc=weight; self.ws=(1-weight)/2
    def forward(self,x):
        x=self.gabor1(x); x1=self.wc*self.argmax(x)+self.ws*(self.argmax_x(x)+self.argmax_y(x)); x1=self.pool(self.conv1(self.se1(x1)))
        x=self.gabor2(x); x2=self.wc*self.argmax(x)+self.ws*(self.argmax_x(x)+self.argmax_y(x)); x2=self.pool(self.conv2(self.se2(x2)))
        return torch.cat((x1.flatten(1),x2.flatten(1)),dim=1)


class CCNet(nn.Module):
    def __init__(self,num_classes,comp_weight=0.8,dropout=0.5,arcface_s=30.0,arcface_m=0.50):
        super().__init__()
        self.cb1=CompetitiveBlock_Mul_Ord_Comp(1,9,35,stride=3,padding=17,init_ratio=1,weight=comp_weight)
        self.cb2=CompetitiveBlock_Mul_Ord_Comp(1,36,17,stride=3,padding=8,init_ratio=0.5,weight=comp_weight,o2=24)
        self.cb3=CompetitiveBlock_Mul_Ord_Comp(1,9,7,stride=3,padding=3,init_ratio=0.25,weight=comp_weight)
        self.fc=nn.Linear(13152,4096); self.fc1=nn.Linear(4096,2048); self.drop=nn.Dropout(p=dropout)
        self.arclayer=ArcMarginProduct(2048,num_classes,s=arcface_s,m=arcface_m)
    def _extract(self,x): return torch.cat((self.cb1(x),self.cb2(x),self.cb3(x)),dim=1)
    def forward(self,x,y=None):
        x=self._extract(x); x1=self.fc(x); x=self.fc1(x1)
        fe=F.normalize(torch.cat((x1,x),dim=1),dim=-1); return self.arclayer(self.drop(x),y),fe
    @torch.no_grad()
    def get_embedding(self,x,domain_id=None): x=self._extract(x); return F.normalize(self.fc1(self.fc(x)),p=2,dim=1)
    def local_only_keys(self): return ("arc.",)


# ══════════════════════════════════════════════════════════════
#  DINOV2
# ══════════════════════════════════════════════════════════════

class DINOBackbone(nn.Module):
    EMBED_DIM = 384
    def __init__(self):
        super().__init__()
        backbone=torch.hub.load("facebookresearch/dinov2","dinov2_vits14",verbose=False)
        for name,p in backbone.named_parameters():
            p.requires_grad=False
            if "blocks.10" in name or "blocks.11" in name: p.requires_grad=True
        self.backbone=backbone
    def forward(self,x): return F.normalize(self.backbone.forward_features(x)["x_norm_clstoken"],p=2,dim=1)

class LoRAExpert(nn.Module): pass

class DINOv2Model(nn.Module):
    def __init__(self,num_classes,arcface_s=16.0,arcface_m=0.30):
        super().__init__()
        self.backbone=DINOBackbone()
        self.arc=ArcMarginProduct(DINOBackbone.EMBED_DIM,num_classes,s=arcface_s,m=arcface_m)
    def forward(self,x,y=None): return self.arc(self.backbone(x),y)
    @torch.no_grad()
    def get_embedding(self,x,domain_id=None): return self.backbone(x)
    def local_only_keys(self): return ("arc.",)


# ══════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ══════════════════════════════════════════════════════════════

def build_model(cfg, num_classes):
    name = cfg["model"].strip().lower()
    if name == "compnet":
        return CompNet(
            num_classes   = num_classes,
            embedding_dim = cfg["embedding_dim"],
            arcface_s     = cfg["arcface_s"],
            arcface_m     = cfg["arcface_m"],
            dropout       = cfg["dropout"],
            use_moe       = cfg.get("use_moe",          False),
            lora_rank     = cfg.get("lora_rank",         64),
            use_grl       = cfg.get("use_grl",           False),
            n_domains     = cfg.get("n_domains",         6),
            expert_weight = cfg.get("moe_expert_weight", 0.5),
        )
    elif name == "ccnet":
        return CCNet(
            num_classes = num_classes,
            comp_weight = cfg.get("comp_weight", 0.8),
            dropout     = cfg.get("dropout", 0.5),
            arcface_s   = cfg["arcface_s"],
            arcface_m   = cfg["arcface_m"],
        )
    elif name == "dinov2":
        return DINOv2Model(
            num_classes = num_classes,
            arcface_s   = cfg.get("dino_scale",  16.0),
            arcface_m   = cfg.get("dino_margin",  0.30),
        )
    else:
        raise ValueError(f"Unknown model: '{cfg['model']}'.")
