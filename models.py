# ==============================================================
#  models.py — CompNet, CCNet, DINOv2 architectures
# ==============================================================
#
#  MoE design — MultiExpertCompNet
#  ────────────────────────────────
#  N full CompNet models (one per domain + one shared base).
#
#  Forward (train & inference):
#    logits = 0.5 × base_expert(x) + 0.5 × domain_expert[d](x)
#
#  Embedding (gallery/probe evaluation):
#    emb = 0.5 × base_expert.get_embedding(x)
#        + 0.5 × domain_expert[d].get_embedding(x)
#    then L2-normalised
#
#  Warmup (moe_warmup_round > 0):
#    Phase 1: only base_expert is trained (plain CompNet via FedAvg).
#    Phase 2: activate_moe() copies base_expert weights into every
#             domain_expert as a warm start, then full training begins.
#
#  FedAvg:
#    All CompNet backbone weights shared (base + all domain experts).
#    arc.* heads kept local on every expert.
# ==============================================================

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ══════════════════════════════════════════════════════════════
#  COMPNET BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    """Learnable Gabor Convolution layer — CompNet version."""
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
        y  = y0.view(1,-1).repeat(c_out, c_in, ksz, 1)
        x  = x0.view(-1,1).repeat(c_out, c_in, 1, ksz)
        x  = x.to(sigma.device); y = y.to(sigma.device)
        xt =  x*torch.cos(theta.view(-1,1,1,1)) + y*torch.sin(theta.view(-1,1,1,1))
        yt = -x*torch.sin(theta.view(-1,1,1,1)) + y*torch.cos(theta.view(-1,1,1,1))
        gb = -torch.exp(-0.5*((gamma*xt)**2+yt**2)/(8*sigma.view(-1,1,1,1)**2)
            ) * torch.cos(2*math.pi*f.view(-1,1,1,1)*xt+psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        self.kernel = self._gen_bank(self.kernel_size, self.channel_in,
                                     self.channel_out, self.sigma, self.gamma,
                                     self.theta, self.f, self.psi)
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)


class CompetitiveBlock(nn.Module):
    """CB = LGC + soft-argmax + PPU  (CompNet)."""
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
    """ArcFace angular margin product layer."""
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.s = s; self.m = m
        self.weight      = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

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
            return self.s * ((one_hot * phi) + ((1 - one_hot) * cosine))
        return self.s * cosine


class GRL(torch.autograd.Function):
    """Gradient Reversal Layer."""
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class DomainClassifier(nn.Module):
    """Domain classification head used with GRL."""
    def __init__(self, in_features, n_domains):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, n_domains),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════
#  SINGLE CompNet  (one expert)
# ══════════════════════════════════════════════════════════════

class CompNet(nn.Module):
    """
    Single CompNet expert: CB1//CB2//CB3 → FC(9708→512) → LN → Dropout → ArcFace.

    Used standalone (baseline) or as one expert inside MultiExpertCompNet.
    get_embedding() returns the L2-normalised 512-d embedding before ArcFace.
    """
    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25,
                 use_grl=False, n_domains=6):
        super().__init__()
        self.cb1      = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2      = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3      = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)
        self.fc       = nn.Linear(9708, embedding_dim)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.drop     = nn.Dropout(p=dropout)
        self.arc      = ArcMarginProduct(embedding_dim, num_classes,
                                         s=arcface_s, m=arcface_m)
        self.use_grl  = use_grl
        self.domain_classifier = (DomainClassifier(embedding_dim, n_domains)
                                   if use_grl else None)

    def _backbone(self, x):
        feat = torch.cat([self.cb1(x).flatten(1),
                          self.cb2(x).flatten(1),
                          self.cb3(x).flatten(1)], dim=1)   # (B, 9708)
        return self.emb_norm(self.fc(feat))                 # (B, 512)

    def forward(self, x, y=None, domain_ids=None):
        return self.arc(self.drop(self._backbone(x)), y)

    @torch.no_grad()
    def get_embedding(self, x):
        return F.normalize(self._backbone(x), p=2, dim=1)

    def get_backbone_state(self):
        """Return all parameters except arc.* for FedAvg."""
        return {k: v.cpu().clone()
                for k, v in self.state_dict().items()
                if not k.startswith("arc.")}

    def set_backbone_state(self, state_dict):
        """Load backbone weights; arc.* preserved unchanged."""
        local = self.state_dict()
        for k, v in state_dict.items():
            if k in local and local[k].shape == v.shape:
                local[k] = v.clone()
        self.load_state_dict(local)


# ══════════════════════════════════════════════════════════════
#  MultiExpertCompNet  (MoE over full CompNet models)
# ══════════════════════════════════════════════════════════════

class MultiExpertCompNet(nn.Module):
    """
    MoE wrapper holding N+1 full CompNet models:
      • experts[0]          — shared base expert (always active)
      • experts[1..N]       — one domain-specific expert per domain

    Forward (train & eval):
      logits = 0.5 × base_expert(x, y) + 0.5 × domain_expert[d](x, y)

    Embedding (gallery/probe):
      emb = L2_norm(0.5 × base_emb + 0.5 × domain_emb)

    domain_ids tensor selects which domain expert to pair with the base.
    At inference domain_ids comes from the test sample's domain label
    (stored in gallery/probe tuples as the third element).

    Warmup (moe_warmup_round > 0)
    ──────────────────────────────
    During warmup use_moe=False — only experts[0] (the base) is trained.
    activate_moe() copies base weights into every domain expert and sets
    use_moe=True so subsequent rounds use both experts.

    FedAvg
    ──────
    get_weights() returns backbone params of ALL experts (base + domain).
    arc.* heads of every expert are excluded and stay local.
    """

    def __init__(self, num_classes, n_domains, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25,
                 use_moe=False, use_grl=False):
        super().__init__()
        self.n_domains     = n_domains
        self.embedding_dim = embedding_dim
        self.use_moe       = use_moe

        # experts[0] = shared base; experts[1..n_domains] = domain experts
        n_experts = 1 + n_domains
        self.experts = nn.ModuleList([
            CompNet(num_classes, embedding_dim, arcface_s, arcface_m,
                    dropout, use_grl=use_grl, n_domains=n_domains)
            for _ in range(n_experts)
        ])

        self.use_grl = use_grl

    # ── MoE activation ────────────────────────────────────────────────────────

    def activate_moe(self):
        """
        Copy base expert weights into every domain expert as a warm start,
        then enable dual-expert routing.

        Called once by the server after warmup aggregation.  Each domain
        expert starts with identical weights to the base — they diverge
        only by the domain-specific gradient signal they receive in
        subsequent rounds.
        """
        if self.use_moe:
            print("  [activate_moe] already active — skipping.")
            return

        base_state = copy.deepcopy(self.experts[0].state_dict())
        for d in range(1, len(self.experts)):
            self.experts[d].load_state_dict(base_state)

        self.use_moe = True
        n_domain_experts = len(self.experts) - 1
        print(f"  [activate_moe] {n_domain_experts} domain experts "
              f"warm-started from base expert.")

    # ── forward helpers ───────────────────────────────────────────────────────

    def _base_expert(self):
        return self.experts[0]

    def _domain_expert(self, d):
        # experts[1..n_domains] → domain index d maps to experts[d+1]
        return self.experts[d + 1]

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x, y=None, domain_ids=None):
        """
        Training forward — returns aggregated logits.

        If use_moe=False (warmup): only base expert runs.
        If use_moe=True: base + domain expert logits averaged (0.5/0.5).

        domain_ids : LongTensor (B,) — domain index per sample (0..n_domains-1)
        """
        base_logits = self._base_expert()(x, y)

        if not self.use_moe or domain_ids is None:
            return base_logits

        # per-sample domain expert selection — vectorised over unique domains
        domain_logits = torch.zeros_like(base_logits)
        for d in range(self.n_domains):
            mask = (domain_ids == d)
            if mask.any():
                domain_logits[mask] = self._domain_expert(d)(x[mask], y[mask])

        return 0.5 * base_logits + 0.5 * domain_logits

    @torch.no_grad()
    def get_embedding(self, x, domain_ids=None):
        """
        L2-normalised embedding for gallery/probe evaluation.

        If use_moe=False or domain_ids=None: base expert only.
        If use_moe=True: average of base + domain expert embeddings,
        then L2-normalised.

        domain_ids : LongTensor (B,) or None
        """
        base_emb = self._base_expert().get_embedding(x)   # (B, 512), L2-normed

        if not self.use_moe or domain_ids is None:
            return base_emb

        domain_emb = torch.zeros_like(base_emb)
        for d in range(self.n_domains):
            mask = (domain_ids == d)
            if mask.any():
                domain_emb[mask] = self._domain_expert(d).get_embedding(x[mask])

        return F.normalize(0.5 * base_emb + 0.5 * domain_emb, p=2, dim=1)

    # ── weight management for FedAvg ──────────────────────────────────────────

    def get_weights(self):
        """
        Return all backbone parameters across all experts for FedAvg.
        Keys are prefixed with 'experts.{i}.' to avoid collisions.
        arc.* excluded from every expert.
        """
        weights = {}
        for i, expert in enumerate(self.experts):
            for k, v in expert.get_backbone_state().items():
                weights[f"experts.{i}.{k}"] = v
        return weights

    def set_weights(self, weights):
        """
        Load FedAvg-aggregated backbone weights back into all experts.
        arc.* keys absent from weights dict — preserved unchanged.
        """
        for i, expert in enumerate(self.experts):
            prefix = f"experts.{i}."
            expert_state = {k[len(prefix):]: v
                            for k, v in weights.items()
                            if k.startswith(prefix)}
            if expert_state:
                expert.set_backbone_state(expert_state)

    def get_moe_lb_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)


# ══════════════════════════════════════════════════════════════
#  CCNET
# ══════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature      = temperature
        self.contrast_mode    = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count   = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]; anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature; anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask         = mask * logits_mask
        exp_logits   = torch.exp(logits) * logits_mask
        log_prob     = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


class CCGaborConv2d(nn.Module):
    """Learnable Gabor Convolution layer — CCNet version."""
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.init_ratio  = init_ratio if init_ratio > 0 else 1.0
        self.kernel      = 0
        self.gamma = nn.Parameter(torch.FloatTensor([2.0]),      requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([9.2*init_ratio]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([0.057/init_ratio]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out,
                     sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2; xmin = -xmax; ksize = xmax - xmin + 1
        y_0  = torch.arange(xmin, xmax + 1).float()
        x_0  = torch.arange(xmin, xmax + 1).float()
        y = y_0.view(1,-1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1,1).repeat(channel_out, channel_in, 1, ksize)
        x = x.float().to(sigma.device); y = y.float().to(sigma.device)
        xt =  x*torch.cos(theta.view(-1,1,1,1)) + y*torch.sin(theta.view(-1,1,1,1))
        yt = -x*torch.sin(theta.view(-1,1,1,1)) + y*torch.cos(theta.view(-1,1,1,1))
        gb = -torch.exp(
            -0.5*((gamma*xt)**2 + yt**2) / (8*sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2*math.pi*f.view(-1,1,1,1)*xt + psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in,
                                   self.channel_out, self.sigma, self.gamma,
                                   self.theta, self.f, self.psi)
        self.kernel = kernel
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CompetitiveBlock_Mul_Ord_Comp(nn.Module):
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 weight, init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor_conv2d  = CCGaborConv2d(channel_in, n_competitor, ksize,
                                           stride=2, padding=ksize//2,
                                           init_ratio=init_ratio)
        self.gabor_conv2d2 = CCGaborConv2d(n_competitor, n_competitor, ksize,
                                           stride=2, padding=ksize//2,
                                           init_ratio=init_ratio)
        self.argmax   = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        self.conv1_1  = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.conv2_1  = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.maxpool  = nn.MaxPool2d(2, 2)
        self.se1      = SELayer(n_competitor)
        self.se2      = SELayer(n_competitor)
        self.weight_chan = weight
        self.weight_spa  = (1 - weight) / 2

    def forward(self, x):
        x = self.gabor_conv2d(x)
        x_1 = self.weight_chan * self.argmax(x) + self.weight_spa * (
              self.argmax_x(x) + self.argmax_y(x))
        x_1 = self.se1(x_1)
        x_1 = self.maxpool(self.conv1_1(x_1))
        x = self.gabor_conv2d2(x)
        x_2 = self.weight_chan * self.argmax(x) + self.weight_spa * (
              self.argmax_x(x) + self.argmax_y(x))
        x_2 = self.se2(x_2)
        x_2 = self.maxpool(self.conv2_1(x_2))
        return torch.cat((x_1.view(x_1.shape[0], -1),
                          x_2.view(x_2.shape[0], -1)), dim=1)


class CCNet(nn.Module):
    def __init__(self, num_classes, comp_weight=0.8, dropout=0.5,
                 arcface_s=30.0, arcface_m=0.50):
        super().__init__()
        self.num_classes = num_classes
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
        x  = self._extract(x)
        x1 = self.fc(x)
        x  = self.fc1(x1)
        fe = F.normalize(torch.cat((x1, x), dim=1), dim=-1)
        return self.arclayer(self.drop(x), y), fe

    @torch.no_grad()
    def get_embedding(self, x):
        x = self._extract(x)
        return F.normalize(self.fc1(self.fc(x)), p=2, dim=1)


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
        out = self.backbone.forward_features(x)
        return F.normalize(out["x_norm_clstoken"], p=2, dim=1)


class DINOv2Model(nn.Module):
    def __init__(self, num_classes, arcface_s=16.0, arcface_m=0.30):
        super().__init__()
        self.backbone = DINOBackbone()
        self.arc      = ArcMarginProduct(DINOBackbone.EMBED_DIM, num_classes,
                                         s=arcface_s, m=arcface_m)

    def forward(self, x, y=None):
        return self.arc(self.backbone(x), y)

    @torch.no_grad()
    def get_embedding(self, x):
        return self.backbone(x)


# ══════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ══════════════════════════════════════════════════════════════

def build_model(cfg, num_classes):
    """
    Instantiate the model specified in cfg["model"].

    For compnet with use_moe=True: returns MultiExpertCompNet.
      - If moe_warmup_round > 0, use_moe is forced False on construction
        so the model starts in warmup mode (base expert only).
        activate_moe() is called by the server at the right round.
      - If moe_warmup_round == 0, full MoE is active from round 1.

    For compnet with use_moe=False: returns plain CompNet.
    """
    name = cfg["model"].strip().lower()

    if name == "compnet":
        if cfg.get("use_moe", False):
            warmup = cfg.get("moe_warmup_round", 0)
            # start in warmup mode when a warmup period is configured
            effective_use_moe = (warmup == 0)
            n_domains = cfg.get("n_domains", 6)
            return MultiExpertCompNet(
                num_classes   = num_classes,
                n_domains     = n_domains,
                embedding_dim = cfg["embedding_dim"],
                arcface_s     = cfg["arcface_s"],
                arcface_m     = cfg["arcface_m"],
                dropout       = cfg["dropout"],
                use_moe       = effective_use_moe,
                use_grl       = cfg.get("use_grl", False),
            )
        else:
            return CompNet(
                num_classes   = num_classes,
                embedding_dim = cfg["embedding_dim"],
                arcface_s     = cfg["arcface_s"],
                arcface_m     = cfg["arcface_m"],
                dropout       = cfg["dropout"],
                use_grl       = cfg.get("use_grl", False),
                n_domains     = cfg.get("n_domains", 6),
            )

    elif name == "ccnet":
        return CCNet(
            num_classes  = num_classes,
            comp_weight  = cfg.get("comp_weight", 0.8),
            dropout      = cfg.get("dropout", 0.5),
            arcface_s    = cfg["arcface_s"],
            arcface_m    = cfg["arcface_m"],
        )

    elif name == "dinov2":
        return DINOv2Model(
            num_classes = num_classes,
            arcface_s   = cfg.get("dino_scale",  16.0),
            arcface_m   = cfg.get("dino_margin",  0.30),
        )

    else:
        raise ValueError(f"Unknown model: '{cfg['model']}'. "
                         f"Choose 'compnet', 'ccnet', or 'dinov2'.")
