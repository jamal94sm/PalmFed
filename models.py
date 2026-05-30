# ==============================================================
#  models.py — CompNet and CCNet architectures
# ==============================================================
#
#  MoE design (CompNet):
#
#  Each client owns TWO experts inside DualExpertFC:
#    base_expert    : nn.Linear(9708→512)
#                     Updated by ALL samples (real + FFT augmented).
#                     Shared across clients via FedAvg.
#                     Learns domain-invariant projection.
#
#    domain_expert  : _ResidualExpert(9708→rank→512)
#                     Updated ONLY by real own-domain samples
#                     (domain_id == client_id, i.e. aug_idx==0).
#                     NEVER FedAvg'd — stays permanently local.
#                     Learns the domain-specific correction on top
#                     of the shared base projection.
#
#  Combination:
#    Training (real sample)  : emb = base_expert(x) + gate * domain_expert(x)
#    Training (FFT sentinel) : emb = base_expert(x)          (domain_expert skipped)
#    Inference (domain known): emb = base_expert(x) + gate_k * domain_expert_k(x)
#    Inference (domain unknown): emb = base_expert(x)
#
#  Two global models (assembled by FLServer each round):
#    GlobalBase : shared base_expert only. Used when domain label is unknown.
#    GlobalFull : shared base_expert + all n_clients domain_experts stacked.
#                 At inference for sample with known domain_id=k:
#                   emb = base(x) + gate_k * domain_experts[k](x)
#                 Routing is deterministic by domain label — no learned router.
#
#  Weight management:
#    local_base_keys()   → "fc.base_expert.*", "fc.gate.*"  (FedAvg'd)
#    local_only_keys()   → "arc.*", "fc.domain_expert.*"    (never shared)
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


class _ResidualExpert(nn.Module):
    """Low-rank residual: A(in→rank)→ReLU→B(rank→out).  B init=0."""
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_normal_(self.A.weight, nonlinearity="relu")
        nn.init.zeros_(self.B.weight)
    def forward(self, x):
        return self.B(F.relu(self.A(x)))


# ══════════════════════════════════════════════════════════════
#  DUAL-EXPERT FC BOTTLENECK
# ══════════════════════════════════════════════════════════════

class DualExpertFC(nn.Module):
    """
    Two-expert FC bottleneck replacing the single nn.Linear(9708→512).

    base_expert   : nn.Linear(9708→512)
      • Receives gradient from ALL samples (real + FFT augmented).
      • Shared via FedAvg — learns the domain-invariant projection.

    domain_expert : _ResidualExpert(9708→rank→512)
      • Receives gradient ONLY from real own-domain samples
        (domain_id >= 0, i.e. NOT the FFT sentinel -1).
      • NEVER FedAvg'd — permanently local to owning client.
      • Learns a domain-specific correction on top of base_expert.

    gate : learnable scalar in (0, gate_scale)
      • Scales the domain_expert residual.
      • Local (not FedAvg'd). Initialised to gate_init.
      • Allows the model to learn how much domain correction to apply.

    Forward logic
    ─────────────
    Training:
      domain_id >= 0  →  base(x) + gate * domain_expert(x)   [real sample]
      domain_id == -1 →  base(x)                              [FFT sentinel]
      domain_id is None → base(x)                             [warmup]

    Inference (via embed()):
      domain_id = int  →  base(x) + gate * domain_expert(x)
      domain_id = None →  base(x)

    GlobalFull inference (via embed_with_external_expert()):
      Accepts an external domain_expert module (from a different client's
      local model) and its gate scalar, enabling the server to assemble
      a full specialist model without owning any client data.
    """

    SENTINEL = -1   # FFT-augmented samples: base only, domain_expert skipped

    def __init__(self, in_features=9708, out_features=512, rank=64,
                 gate_mode="scalar", gate_init=1.0, gate_scale=2.0):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.gate_scale   = gate_scale

        # ── shared (FedAvg'd) ──────────────────────────────────────────────
        self.base_expert = nn.Linear(in_features, out_features)

        # ── local-only (never FedAvg'd) ───────────────────────────────────
        self.domain_expert = _ResidualExpert(in_features, out_features, rank)

        # gate: raw scalar; effective = sigmoid(raw) * scale
        raw_init = math.log(
            max(gate_init, 1e-6) / max(gate_scale - gate_init, 1e-6))
        self.gate_raw = nn.Parameter(torch.tensor(raw_init))

        self._warmup = False
        # routing counters
        self.register_buffer("_real_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_skip_count", torch.tensor(0, dtype=torch.long))

    # ── gate helpers ─────────────────────────────────────────────────────────

    def gate_value(self):
        """Current effective gate value (scalar float)."""
        return float(torch.sigmoid(self.gate_raw) * self.gate_scale)

    # ── warmup control ───────────────────────────────────────────────────────

    def set_warmup(self, warmup: bool):
        """Freeze/unfreeze domain_expert and gate."""
        self._warmup = warmup
        for p in self.domain_expert.parameters():
            p.requires_grad = not warmup
        self.gate_raw.requires_grad = not warmup

    @property
    def is_warming_up(self): return self._warmup

    # ── routing stats ────────────────────────────────────────────────────────

    def reset_routing_stats(self):
        self._real_count.zero_(); self._skip_count.zero_()

    def get_routing_stats(self):
        return {
            "real" : int(self._real_count),
            "skip" : int(self._skip_count),
        }

    # ── forward (training) ───────────────────────────────────────────────────

    def forward(self, x, domain_ids=None):
        """
        x          : Tensor [B, in_features]
        domain_ids : Tensor [B] of int | None
          >= 0  → real own-domain sample → base + gate*domain_expert
          == -1 → FFT sentinel           → base only
          None  → warmup / no routing info → base only
        """
        base_out = self.base_expert(x)

        if self._warmup or domain_ids is None:
            return base_out

        # partition into real (>=0) and sentinel (-1)
        real_mask = (domain_ids >= 0)
        skip_mask = ~real_mask

        out = base_out.clone()

        if real_mask.any():
            g        = torch.sigmoid(self.gate_raw) * self.gate_scale
            residual = self.domain_expert(x[real_mask])
            out[real_mask] = base_out[real_mask] + g * residual
            if self.training:
                self._real_count += real_mask.sum().item()

        if skip_mask.any() and self.training:
            self._skip_count += skip_mask.sum().item()

        return out

    # ── inference: own domain_expert ─────────────────────────────────────────

    def embed(self, feat, domain_id=None):
        """
        Single-domain inference.
        domain_id : int → base + gate * domain_expert  (domain-aware)
                    None → base only                   (domain-unknown)
        """
        base_out = self.base_expert(feat)
        if domain_id is None or self._warmup:
            return base_out
        g = torch.sigmoid(self.gate_raw) * self.gate_scale
        return base_out + g * self.domain_expert(feat)

    # ── inference: external domain_expert (GlobalFull) ───────────────────────

    @torch.no_grad()
    def embed_with_external_expert(self, feat, ext_domain_expert, ext_gate_raw):
        """
        GlobalFull inference: use this module's base_expert but an external
        client's domain_expert and gate_raw.

        Called by FLServer.global_full_model for gallery/probe samples whose
        domain_id maps to a specific client's specialist expert.

        feat             : Tensor [B, in_features]
        ext_domain_expert: _ResidualExpert  (from client k's local model)
        ext_gate_raw     : Tensor scalar    (gate_raw from client k)
        """
        base_out = self.base_expert(feat)
        g        = torch.sigmoid(ext_gate_raw) * self.gate_scale
        return base_out + g * ext_domain_expert(feat)

    # ── gradient norms ───────────────────────────────────────────────────────

    def get_grad_norms(self):
        def _n(p): return round(float(p.grad.norm()), 4) if p.grad is not None else None
        return {
            "base_grad_norm"  : _n(self.base_expert.weight),
            "domain_grad_norm": _n(self.domain_expert.B.weight),
            "gate_grad_norm"  : _n(self.gate_raw),
        }

    # ── weight-space diagnostics ─────────────────────────────────────────────

    @torch.no_grad()
    def get_weight_diagnostics(self):
        b_norm = float(self.base_expert.weight.norm())
        d_norm = float(self.domain_expert.B.weight.norm())
        gate   = self.gate_value()
        return {
            "base_weight_norm"  : round(b_norm, 4),
            "domain_weight_norm": round(d_norm, 4),
            "gate_value"        : round(gate,   4),
        }

    # ── activation-space diagnostics ─────────────────────────────────────────

    @torch.no_grad()
    def get_activation_diagnostics(self, feat):
        """
        Run fixed probe feat [N, in_features] through both experts.
        Returns norms, gated ratio, and cosine similarity between
        base and (base + gated residual) to measure how much the
        domain_expert actually shifts the embedding direction.
        """
        feat     = feat.float()
        base_out = self.base_expert(feat)              # [N, 512]
        raw_res  = self.domain_expert(feat)            # [N, 512]
        g        = torch.sigmoid(self.gate_raw) * self.gate_scale
        gated    = g * raw_res                         # [N, 512]
        full_out = base_out + gated                    # [N, 512]

        base_norm  = float(base_out.norm(dim=1).mean())
        raw_norm   = float(raw_res.norm(dim=1).mean())
        gated_norm = float(gated.norm(dim=1).mean())

        # cosine similarity between base embedding and full embedding
        # closer to 1 → domain_expert barely shifts direction
        # closer to 0 → strong directional correction
        cos_sim = float(F.cosine_similarity(
            F.normalize(base_out, dim=1),
            F.normalize(full_out, dim=1)).mean())

        return {
            "base_norm"        : round(base_norm, 4),
            "domain_raw_norm"  : round(raw_norm, 4),
            "domain_gated_norm": round(gated_norm, 4),
            "gated_base_ratio" : round(gated_norm / max(base_norm, 1e-8), 4),
            "base_full_cos_sim": round(cos_sim, 4),   # 1=no shift, 0=90° shift
            "gate_value"       : round(float(g), 4),
        }


# ══════════════════════════════════════════════════════════════
#  COMPNET  (with DualExpertFC)
# ══════════════════════════════════════════════════════════════

class CompNet(nn.Module):
    """
    CompNet = CB1//CB2//CB3 + DualExpertFC(9708→512) + Dropout + ArcFace.

    Weight management
    ─────────────────
    FedAvg'd (base keys):  fc.base_expert.*,  fc.gate_raw (gate is shared)
    Local-only:            arc.*,  fc.domain_expert.*,  fc.gate_raw*

    Wait — gate is local (it scales a local expert) but whether to FedAvg
    it is a design choice:
      • FedAvg gate: all clients converge to similar gate magnitude.
        Simpler. Works if domain correction magnitude is similar everywhere.
      • Local gate: each client learns how much its domain needs correction.
        Better if some domains are very different from others.
    We keep gate LOCAL (not FedAvg'd) because it directly scales a
    local expert — averaging it across clients would be meaningless.

    local_only_keys():  ("arc.", "fc.domain_expert.", "fc.gate_raw")
    fedavg_keys():      everything else (fc.base_expert.*, Gabor CB params)

    Two-model inference
    ───────────────────
    GlobalBase  → get_embedding(x, domain_id=None)      uses base only
    GlobalFull  → get_embedding(x, domain_id=k)         uses base + domain_expert_k
                  where domain_expert_k comes from client k's local model
                  (assembled by FLServer after each round)
    """

    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25,
                 use_moe=False, lora_rank=64,
                 use_grl=False, n_domains=6,
                 gate_mode="scalar", gate_init=1.0, gate_scale=2.0):
        super().__init__()
        self.cb1 = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2 = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3 = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)
        self.use_moe = use_moe
        if use_moe:
            self.fc = DualExpertFC(9708, embedding_dim, lora_rank,
                                   gate_mode=gate_mode,
                                   gate_init=gate_init,
                                   gate_scale=gate_scale)
        else:
            self.fc = nn.Linear(9708, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.arc  = ArcMarginProduct(embedding_dim, num_classes,
                                     s=arcface_s, m=arcface_m)
        self.use_grl = use_grl
        self.domain_classifier = (DomainClassifier(embedding_dim, n_domains)
                                   if use_grl else None)

    # ── key classification ───────────────────────────────────────────────────

    def local_only_keys(self):
        """
        Keys that NEVER leave the client.
        arc.*              : local ArcFace prototypes
        fc.domain_expert.* : local domain-specific residual adapter
        fc.gate_raw        : local gate scalar (scales local expert)
        """
        return ("arc.", "fc.domain_expert.", "fc.gate_raw")

    def fedavg_keys(self):
        """
        Keys that ARE aggregated via FedAvg.
        fc.base_expert.*   : shared domain-invariant FC projection
        cb*.*/conv*/...    : shared Gabor feature extractor
        """
        excl = self.local_only_keys()
        return [k for k in self.state_dict()
                if not any(k.startswith(p) for p in excl)]

    # ── MoE warmup ───────────────────────────────────────────────────────────

    def set_moe_warmup(self, warmup: bool):
        if self.use_moe: self.fc.set_warmup(warmup)

    @property
    def moe_is_warming_up(self):
        return self.use_moe and self.fc.is_warming_up

    # ── feature extraction ───────────────────────────────────────────────────

    def _gabor_feat(self, x):
        return torch.cat([self.cb1(x).flatten(1),
                          self.cb2(x).flatten(1),
                          self.cb3(x).flatten(1)], dim=1)   # [B, 9708]

    def _backbone(self, x, domain_ids=None):
        feat = self._gabor_feat(x)
        if self.use_moe:
            return self.fc(feat, domain_ids)
        return self.fc(feat)

    # ── training forward ─────────────────────────────────────────────────────

    def forward(self, x, y=None, domain_ids=None):
        return self.arc(self.drop(self._backbone(x, domain_ids)), y)

    # ── inference: own expert (local model or GlobalFull by domain) ──────────

    @torch.no_grad()
    def get_embedding(self, x, domain_id=None):
        """
        domain_id : int  → base_expert + gate * domain_expert  (GlobalFull)
                    None → base_expert only                     (GlobalBase)
        """
        feat = self._gabor_feat(x)
        if self.use_moe:
            return F.normalize(self.fc.embed(feat, domain_id), p=2, dim=1)
        return F.normalize(self.fc(feat), p=2, dim=1)

    # ── inference: external expert (server assembles GlobalFull) ────────────

    @torch.no_grad()
    def get_embedding_with_external_expert(self, x, ext_domain_expert,
                                            ext_gate_raw):
        """
        Used by FLServer.global_full_model to evaluate GlobalFull:
        applies this model's base_expert + Gabor backbone, but uses an
        external client's domain_expert and gate_raw.

        x                : Tensor [B, C, H, W]
        ext_domain_expert: _ResidualExpert from client k
        ext_gate_raw     : Tensor scalar from client k
        """
        feat    = self._gabor_feat(x)
        raw_emb = self.fc.embed_with_external_expert(
            feat, ext_domain_expert, ext_gate_raw)
        return F.normalize(raw_emb, p=2, dim=1)

    # ── diagnostics delegation ───────────────────────────────────────────────

    def reset_routing_stats(self):
        if self.use_moe: self.fc.reset_routing_stats()

    def get_routing_stats(self):
        return self.fc.get_routing_stats() if self.use_moe else None

    def get_grad_norms(self):
        return self.fc.get_grad_norms() if self.use_moe else None

    def get_weight_diagnostics(self):
        return self.fc.get_weight_diagnostics() if self.use_moe else None

    def get_activation_diagnostics(self, probe_feat):
        return self.fc.get_activation_diagnostics(probe_feat) \
               if self.use_moe else None


# ══════════════════════════════════════════════════════════════
#  CCNET
# ══════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature      = temperature
        self.contrast_mode    = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('features needs [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both labels and mask')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask   = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count   = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]; anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature; anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits         = anchor_dot_contrast - logits_max.detach()
        mask           = mask.repeat(anchor_count, contrast_count)
        logits_mask    = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask           = mask * logits_mask
        exp_logits     = torch.exp(logits) * logits_mask
        log_prob       = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


class CCGaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in = channel_in; self.channel_out = channel_out
        self.kernel_size = kernel_size; self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio if init_ratio > 0 else 1.0; self.kernel = 0
        _S = 9.2*self.init_ratio; _F = 0.057/self.init_ratio; _G = 2.0
        self.gamma = nn.Parameter(torch.FloatTensor([_G]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([_S]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([_F]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out,
                     sigma, gamma, theta, f, psi):
        xmax = kernel_size//2; xmin = -xmax; ksize = xmax-xmin+1
        y_0 = torch.arange(xmin, xmax+1).float()
        x_0 = torch.arange(xmin, xmax+1).float()
        y = y_0.view(1,-1).repeat(channel_out,channel_in,ksize,1)
        x = x_0.view(-1,1).repeat(channel_out,channel_in,1,ksize)
        x = x.to(sigma.device); y = y.to(sigma.device)
        xt =  x*torch.cos(theta.view(-1,1,1,1)) + y*torch.sin(theta.view(-1,1,1,1))
        yt = -x*torch.sin(theta.view(-1,1,1,1)) + y*torch.cos(theta.view(-1,1,1,1))
        gb = -torch.exp(
            -0.5*((gamma*xt)**2+yt**2)/(8*sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2*math.pi*f.view(-1,1,1,1)*xt+psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        k = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out,
                              self.sigma, self.gamma, self.theta, self.f, self.psi)
        self.kernel = k
        return F.conv2d(x, k, stride=self.stride, padding=self.padding)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b,c,_,_ = x.size(); y = self.avg_pool(x).view(b,c)
        return x * self.fc(y).view(b,c,1,1).expand_as(x)


class CompetitiveBlock_Mul_Ord_Comp(nn.Module):
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 weight, init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor1  = CCGaborConv2d(channel_in, n_competitor, ksize,
                                     stride=2, padding=ksize//2, init_ratio=init_ratio)
        self.gabor2  = CCGaborConv2d(n_competitor, n_competitor, ksize,
                                     stride=2, padding=ksize//2, init_ratio=init_ratio)
        self.argmax  = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        self.conv1   = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.conv2   = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.pool    = nn.MaxPool2d(2, 2)
        self.se1     = SELayer(n_competitor)
        self.se2     = SELayer(n_competitor)
        self.wc = weight; self.ws = (1-weight)/2

    def forward(self, x):
        x   = self.gabor1(x)
        x1  = self.wc*self.argmax(x) + self.ws*(self.argmax_x(x)+self.argmax_y(x))
        x1  = self.pool(self.conv1(self.se1(x1)))
        x   = self.gabor2(x)
        x2  = self.wc*self.argmax(x) + self.ws*(self.argmax_x(x)+self.argmax_y(x))
        x2  = self.pool(self.conv2(self.se2(x2)))
        return torch.cat((x1.flatten(1), x2.flatten(1)), dim=1)


class CCNet(nn.Module):
    def __init__(self, num_classes, comp_weight=0.8, dropout=0.5,
                 arcface_s=30.0, arcface_m=0.50):
        super().__init__()
        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(
            1,  9, 35, stride=3, padding=17, init_ratio=1,    weight=comp_weight)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(
            1, 36, 17, stride=3, padding=8,  init_ratio=0.5,  weight=comp_weight, o2=24)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(
            1,  9,  7, stride=3, padding=3,  init_ratio=0.25, weight=comp_weight)
        self.fc       = nn.Linear(13152, 4096)
        self.fc1      = nn.Linear(4096, 2048)
        self.drop     = nn.Dropout(p=dropout)
        self.arclayer = ArcMarginProduct(2048, num_classes,
                                         s=arcface_s, m=arcface_m)
    def _extract(self, x):
        return torch.cat((self.cb1(x), self.cb2(x), self.cb3(x)), dim=1)
    def forward(self, x, y=None):
        x = self._extract(x); x1 = self.fc(x); x = self.fc1(x1)
        fe = F.normalize(torch.cat((x1, x), dim=1), dim=-1)
        return self.arclayer(self.drop(x), y), fe
    @torch.no_grad()
    def get_embedding(self, x, domain_id=None):
        x = self._extract(x)
        return F.normalize(self.fc1(self.fc(x)), p=2, dim=1)
    def local_only_keys(self): return ("arc.",)


# ══════════════════════════════════════════════════════════════
#  DINOV2
# ══════════════════════════════════════════════════════════════

class DINOBackbone(nn.Module):
    EMBED_DIM = 384
    def __init__(self):
        super().__init__()
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                  verbose=False)
        for name, p in backbone.named_parameters():
            p.requires_grad = False
            if "blocks.10" in name or "blocks.11" in name:
                p.requires_grad = True
        self.backbone = backbone
    def forward(self, x):
        return F.normalize(
            self.backbone.forward_features(x)["x_norm_clstoken"], p=2, dim=1)

class LoRAExpert(nn.Module): pass  # placeholder

class DINOv2Model(nn.Module):
    def __init__(self, num_classes, arcface_s=16.0, arcface_m=0.30):
        super().__init__()
        self.backbone = DINOBackbone()
        self.arc = ArcMarginProduct(DINOBackbone.EMBED_DIM, num_classes,
                                    s=arcface_s, m=arcface_m)
    def forward(self, x, y=None): return self.arc(self.backbone(x), y)
    @torch.no_grad()
    def get_embedding(self, x, domain_id=None): return self.backbone(x)
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
            use_moe       = cfg.get("use_moe",       False),
            lora_rank     = cfg.get("lora_rank",      64),
            use_grl       = cfg.get("use_grl",        False),
            n_domains     = cfg.get("n_domains",      6),
            gate_mode     = cfg.get("moe_gate_mode",  "scalar"),
            gate_init     = cfg.get("moe_gate_init",  1.0),
            gate_scale    = cfg.get("moe_gate_scale", 2.0),
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
        raise ValueError(
            f"Unknown model: '{cfg['model']}'. Choose 'compnet', 'ccnet', or 'dinov2'.")
