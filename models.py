# ==============================================================
#  models.py — CompNet and CCNet architectures
# ==============================================================
#  CompNet : GaborConv2d → CompetitiveBlock × 3 → FC → ArcFace
#  CCNet   : CCGaborConv2d → CompetitiveBlock_Mul_Ord_Comp × 3
#            → FC(13152→4096→2048) → ArcFace  +  SupConLoss
# ==============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ══════════════════════════════════════════════════════════════
#  COMPNET
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
    """ArcFace angular margin product layer (shared by CompNet and CCNet)."""
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


class MoEFC(nn.Module):
    """
    Base FC + residual Mixture-of-Experts replacing CompNet's FC(9708→512).

    Architecture:
      output = base_FC(x) + Σ_k gates_k × residual_expert_k(x)

    The shared base_FC is identical to the original nn.Linear(9708→512).
    Residual experts are low-rank adapters (9708→rank→512) initialised to
    output zero (B matrix set to zeros). This guarantees that at round 0
    the model behaves exactly like the original CompNet — no degradation
    from initialisation. Experts learn domain-specific corrections on top
    of the shared base as training progresses.

    Top-K gating: only top-k experts receive non-zero weight, forcing
    specialisation. Load balancing uses full gate weights so all experts
    receive gradient to maintain utilisation.

    Parameters
    ----------
    in_features  : int  9708
    out_features : int  512
    n_experts    : int  number of residual experts
    rank         : int  expert bottleneck rank
    top_k        : int  active experts per sample
    """
    def __init__(self, in_features=9708, out_features=512,
                 n_experts=6, rank=64, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = min(top_k, n_experts)
        self._lb_loss  = None

        # shared base — same as original FC, same initialisation
        self.base = nn.Linear(in_features, out_features)

        # gating network
        self.gate = nn.Linear(in_features, n_experts)

        # residual experts — B initialised to zero → zero initial output
        self.experts = nn.ModuleList([
            _ResidualExpert(in_features, out_features, rank)
            for _ in range(n_experts)
        ])

    def forward(self, x):
        """
        x   : [B, 9708]
        out : [B, 512]   base + weighted residual corrections
        """
        base_out   = self.base(x)                              # [B, 512]

        gates_full = torch.softmax(self.gate(x), dim=-1)      # [B, n_experts]

        # top-k mask
        if self.top_k < self.n_experts:
            topk_vals, topk_idx = gates_full.topk(self.top_k, dim=-1)
            mask  = torch.zeros_like(gates_full)
            mask.scatter_(1, topk_idx, topk_vals)
            gates = mask / (mask.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            gates = gates_full

        # residual expert outputs: [B, n_experts, 512]
        expert_outs = torch.stack(
            [exp(x) for exp in self.experts], dim=1)

        # weighted sum of residuals: [B, 512]
        residual = (gates.unsqueeze(-1) * expert_outs).sum(dim=1)

        # load balancing on full (unmasked) gates
        mean_gates    = gates_full.mean(dim=0)
        self._lb_loss = self.n_experts * (mean_gates ** 2).sum()

        return base_out + residual


class _ResidualExpert(nn.Module):
    """Low-rank residual adapter: Linear(in→rank) → ReLU → Linear(rank→out).
    B initialised to zero so initial contribution is exactly zero."""
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_normal_(self.A.weight, nonlinearity="relu")
        nn.init.zeros_(self.B.weight)    # ← zero init: no contribution at start

    def forward(self, x):
        return self.B(F.relu(self.A(x)))


class CompNet(nn.Module):
    """
    CompNet = CB1 // CB2 // CB3 + FC(9708→emb_dim) + Dropout + ArcFace.

    Without MoE (use_moe=False):
      Three parallel CompetitiveBlocks → concat [B, 9708]
      → FC(9708→512) → Dropout → ArcFace

    With MoE (use_moe=True):
      Three parallel CompetitiveBlocks → concat [B, 9708]
      → MoEFC(9708→512, K experts, top-k routing) → Dropout → ArcFace

    The MoEFC replaces the single projection with K low-rank experts.
    The gate reads the 9708-d multi-scale Gabor feature vector and routes
    each sample to its top-k experts — learning domain-specific projections
    from the concatenated texture fingerprint.

    get_moe_lb_loss(): returns load balancing loss after _backbone()
    is called. Returns zero when use_moe=False.
    """
    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25,
                 use_moe=False, n_experts=6, lora_rank=64, moe_top_k=2):
        super().__init__()
        self.cb1 = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2 = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3 = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)
        self.use_moe = use_moe
        if use_moe:
            self.fc = MoEFC(9708, embedding_dim, n_experts, lora_rank, moe_top_k)
        else:
            self.fc = nn.Linear(9708, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.arc  = ArcMarginProduct(embedding_dim, num_classes,
                                     s=arcface_s, m=arcface_m)

    def _backbone(self, x):
        x1 = self.cb1(x).flatten(1)
        x2 = self.cb2(x).flatten(1)
        x3 = self.cb3(x).flatten(1)
        return self.fc(torch.cat([x1, x2, x3], dim=1))

    def forward(self, x, y=None):
        return self.arc(self.drop(self._backbone(x)), y)

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised embedding for matching — no ArcFace head."""
        return F.normalize(self._backbone(x), p=2, dim=1)

    def get_moe_lb_loss(self):
        """Load balancing loss from MoEFC. Zero when use_moe=False."""
        if not self.use_moe or not hasattr(self.fc, "_lb_loss") \
                or self.fc._lb_loss is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.fc._lb_loss


# ══════════════════════════════════════════════════════════════
#  CCNET
# ══════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (CCNet/loss.py — exact copy).
    Expects features of shape [batch, n_views, embed_dim].
    """
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
        mask = mask * logits_mask
        exp_logits        = torch.exp(logits) * logits_mask
        log_prob          = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


class CCGaborConv2d(nn.Module):
    """Learnable Gabor Convolution layer — CCNet version (stride=2, padding)."""
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
        _SIGMA = 9.2   * self.init_ratio
        _FREQ  = 0.057 / self.init_ratio
        _GAMMA = 2.0
        self.gamma = nn.Parameter(torch.FloatTensor([_GAMMA]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([_SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([_FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]),     requires_grad=False)

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
    """Squeeze-and-Excitation layer (CCNet)."""
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
    """
    Multi-Order Comprehensive Competition Block (CCNet).
    1st order: LGC → spatial+channel competition → SE → conv → pool
    2nd order: LGC(on 1st Gabor) → competition → SE → conv → pool
    Output: concatenation of 1st and 2nd order flattened features.
    """
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 weight, init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor_conv2d  = CCGaborConv2d(channel_in, n_competitor, ksize,
                                           stride=2, padding=ksize//2,
                                           init_ratio=init_ratio)
        self.gabor_conv2d2 = CCGaborConv2d(n_competitor, n_competitor, ksize,
                                           stride=2, padding=ksize//2,
                                           init_ratio=init_ratio)
        self.argmax   = nn.Softmax(dim=1)   # channel competition
        self.argmax_x = nn.Softmax(dim=2)   # spatial-x competition
        self.argmax_y = nn.Softmax(dim=3)   # spatial-y competition
        self.conv1_1  = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.conv2_1  = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.maxpool  = nn.MaxPool2d(2, 2)
        self.se1      = SELayer(n_competitor)
        self.se2      = SELayer(n_competitor)
        self.weight_chan = weight
        self.weight_spa  = (1 - weight) / 2

    def forward(self, x):
        # 1st order
        x = self.gabor_conv2d(x)
        x_1 = self.weight_chan * self.argmax(x) + self.weight_spa * (
              self.argmax_x(x) + self.argmax_y(x))
        x_1 = self.se1(x_1)
        x_1 = self.maxpool(self.conv1_1(x_1))
        # 2nd order
        x = self.gabor_conv2d2(x)
        x_2 = self.weight_chan * self.argmax(x) + self.weight_spa * (
              self.argmax_x(x) + self.argmax_y(x))
        x_2 = self.se2(x_2)
        x_2 = self.maxpool(self.conv2_1(x_2))
        return torch.cat((x_1.view(x_1.shape[0], -1),
                          x_2.view(x_2.shape[0], -1)), dim=1)


class CCNet(nn.Module):
    """
    CCNet = CB1 // CB2 // CB3 + FC(13152→4096→2048) + Dropout + ArcFace.
    FC input size 13152 is fixed for 128×128 input images.

    forward() returns (logits, normalised_6144d_contrastive_features)
    get_embedding() returns L2-normalised 2048-d matching embedding.
    """
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
        x1 = self.cb1(x); x2 = self.cb2(x); x3 = self.cb3(x)
        return torch.cat((x1, x2, x3), dim=1)

    def forward(self, x, y=None):
        x  = self._extract(x)
        x1 = self.fc(x)
        x  = self.fc1(x1)
        # 6144-d normalised features for SupConLoss
        fe = F.normalize(torch.cat((x1, x), dim=1), dim=-1)
        logits = self.arclayer(self.drop(x), y)
        return logits, fe

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised 2048-d embedding for matching."""
        x  = self._extract(x)
        return F.normalize(self.fc1(self.fc(x)), p=2, dim=1)



# ══════════════════════════════════════════════════════════════
#  DINOV2
# ══════════════════════════════════════════════════════════════

class DINOBackbone(nn.Module):
    """
    DINOv2 ViT-S/14 backbone with selective unfreezing.
    All parameters are frozen except blocks 10 and 11 of the ViT,
    which are fine-tuned to adapt to palmprint recognition.
    forward() returns the L2-normalised CLS token (384-d).
    Requires: torch.hub access to facebookresearch/dinov2
    """
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
        """Returns L2-normalised 384-d CLS token."""
        out = self.backbone.forward_features(x)
        cls = out["x_norm_clstoken"]
        return F.normalize(cls, p=2, dim=1)


class LoRAExpert(nn.Module):
    """
    Single LoRA adapter: a low-rank decomposition A·B applied to one linear layer.
    B is initialised to zero so the initial contribution to the base layer is
    exactly zero — LoRA starts as an identity pass-through and learns from there.

    Parameters
    ----------
    in_dim  : int  input dimensionality  (384 for DINOv2 ViT-S fc1 input)
    out_dim : int  output dimensionality (1536 for DINOv2 ViT-S fc1 output)
    rank    : int  LoRA rank — controls capacity vs. parameter count
    """
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Linear(in_dim,  rank,    bias=False)
        self.B = nn.Linear(rank,    out_dim, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)          # zero init → zero initial output

    def forward(self, x):
        return self.B(self.A(x))


class LoRAMoEMLP(nn.Module):
    """
    LoRA Mixture-of-Experts wrapper for a ViT MLP block.

    Replaces the MLP inside transformer blocks 10 and 11 of DINOv2.
    Instead of adding a separate module after all blocks (which operates on
    an already-formed CLS token), this integrates domain-specific adaptation
    directly into the intermediate feature transformation, where the model
    can still modulate both patch and CLS representations.

    Architecture:
      h = fc1(x) + scale × Σ_k gate_k(CLS) × LoRA_k(x)
      h = act(h)
      h = fc2(h)

    The gating network reads the CLS token (x[:, 0, :]) and produces soft
    routing weights over all n_experts LoRA adapters. The same gates apply
    to all N tokens — domain routing is image-level, not token-level.

    Load balancing:
      After each forward pass, _lb_loss stores the load balance penalty:
        n_experts × Σ(mean_gate_k²)
      DINOv2Model.get_moe_lb_loss() collects this from both blocks 10 and 11.

    Parameters
    ----------
    original_mlp : nn.Module  the original MLP from the ViT block (kept intact)
    n_experts    : int        number of LoRA adapters
    rank         : int        LoRA rank
    embed_dim    : int        ViT embedding dim (384 for ViT-S)
    """
    def __init__(self, original_mlp, n_experts, rank, embed_dim):
        super().__init__()
        self.mlp       = original_mlp
        self.n_experts = n_experts
        fc1_out        = original_mlp.fc1.out_features   # 1536 for ViT-S

        self.lora_experts = nn.ModuleList([
            LoRAExpert(embed_dim, fc1_out, rank)
            for _ in range(n_experts)
        ])
        self.gate  = nn.Linear(embed_dim, n_experts)
        self.scale = nn.Parameter(torch.ones(1))   # learnable LoRA scaling
        self._lb_loss = None                        # populated each forward pass

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor [B, N_tokens, embed_dim]  token sequence (patch + CLS)

        Returns
        -------
        Tensor [B, N_tokens, embed_dim]  MLP output with LoRA MoE injection
        """
        # gate on CLS token (index 0) — image-level domain routing
        cls   = x[:, 0, :]                                 # [B, D]
        gates = torch.softmax(self.gate(cls), dim=-1)      # [B, n_experts]

        # all expert outputs for all tokens: stack → [B, n_experts, N, fc1_out]
        lora_out = torch.stack(
            [exp(x) for exp in self.lora_experts], dim=1)

        # weighted sum over experts: [B, N, fc1_out]
        weighted = (gates[:, :, None, None] * lora_out).sum(dim=1)

        # inject into fc1 output
        h = self.mlp.fc1(x) + self.scale * weighted
        h = self.mlp.act(h)
        h = self.mlp.drop1(h) if hasattr(self.mlp, "drop1") else h
        h = self.mlp.fc2(h)
        h = self.mlp.drop2(h) if hasattr(self.mlp, "drop2") else h

        # load balancing loss — stored for extraction by get_moe_lb_loss()
        mean_gates    = gates.mean(dim=0)                  # [n_experts]
        self._lb_loss = self.n_experts * (mean_gates ** 2).sum()

        return h


class DINOBackbone(nn.Module):
    """
    DINOv2 ViT-S/14 backbone with selective unfreezing and optional LoRA MoE.

    Without LoRA MoE (use_moe=False):
      Blocks 0–9   frozen (ImageNet features)
      Blocks 10–11 unfrozen (fine-tuned for palmprint)
      forward() → L2-normalised 384-d CLS token

    With LoRA MoE (use_moe=True):
      Blocks 0–9   frozen
      Blocks 10–11 attention layers: unfrozen
      Blocks 10–11 MLP: replaced with LoRAMoEMLP
        base MLP weights stay unfrozen (as before)
        LoRA adapters (A, B) and gating: new trainable parameters
      The LoRA adapters inject domain-specific transformations directly into
      the MLP intermediate representations, operating on all tokens rather
      than only the final CLS token.
    """
    EMBED_DIM = 384

    def __init__(self, use_moe=False, n_experts=6, lora_rank=16):
        super().__init__()
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                  verbose=False)

        # freeze all, unfreeze last two blocks
        for name, p in backbone.named_parameters():
            p.requires_grad = False
            if "blocks.10" in name or "blocks.11" in name:
                p.requires_grad = True

        # inject LoRA MoE into the MLP of blocks 10 and 11
        if use_moe:
            for blk_idx in [10, 11]:
                original_mlp = backbone.blocks[blk_idx].mlp
                backbone.blocks[blk_idx].mlp = LoRAMoEMLP(
                    original_mlp, n_experts, lora_rank, self.EMBED_DIM)

        self.backbone = backbone
        self.use_moe  = use_moe

    def forward(self, x):
        """Returns L2-normalised 384-d CLS token.
        If use_moe=True, LoRA MoE is applied inside blocks 10-11 during
        forward_features(), populating _lb_loss in each LoRAMoEMLP."""
        out = self.backbone.forward_features(x)
        cls = out["x_norm_clstoken"]
        return F.normalize(cls, p=2, dim=1)

    def get_moe_lb_loss(self):
        """
        Collect and sum load-balancing losses from both LoRA MoE blocks.
        Called after forward() — _lb_loss is populated during forward_features().
        Returns zero tensor when use_moe=False.
        """
        if not self.use_moe:
            return torch.tensor(0.0,
                                device=next(self.parameters()).device)
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for blk_idx in [10, 11]:
            mlp = self.backbone.blocks[blk_idx].mlp
            if hasattr(mlp, "_lb_loss") and mlp._lb_loss is not None:
                total = total + mlp._lb_loss
        return total


class DINOv2Model(nn.Module):
    """
    DINOv2 FL model = DINOBackbone (+ optional LoRA MoE inside blocks) + ArcFace.

    Without LoRA MoE: identical to the original — backbone CLS → ArcFace.
    With LoRA MoE:    LoRA adapters inside blocks 10-11 MLPs refine the
                      intermediate token representations domain-specifically,
                      producing a better CLS token before ArcFace.

    Weight sharing in FL:
      backbone.* (including lora_experts, gate, scale) → shared via FedAvg
      arc.*                                            → kept local

    Input: RGB 224×224 with ImageNet normalisation.
    """
    def __init__(self, num_classes, arcface_s=16.0, arcface_m=0.30,
                 use_moe=False, n_experts=6, lora_rank=16):
        super().__init__()
        self.backbone = DINOBackbone(use_moe, n_experts, lora_rank)
        self.arc      = ArcMarginProduct(DINOBackbone.EMBED_DIM, num_classes,
                                         s=arcface_s, m=arcface_m)

    def forward(self, x, y=None):
        emb = self.backbone(x)         # LoRA MoE applied inside if enabled
        return self.arc(emb, y)        # [B, num_classes] logits

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised 384-d CLS embedding — LoRA-refined when use_moe=True."""
        return self.backbone(x)

    def get_moe_lb_loss(self):
        """Sum of load-balancing losses from LoRA MoE blocks. Zero if disabled."""
        return self.backbone.get_moe_lb_loss()


# ══════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ══════════════════════════════════════════════════════════════

def build_model(cfg, num_classes):
    """
    Instantiate and return the model specified in cfg["model"].

    Parameters
    ----------
    cfg        : dict  CONFIG dictionary
    num_classes: int   number of training identities for this client

    Returns
    -------
    model : nn.Module  (CompNet | CCNet | DINOv2Model)
    """
    name = cfg["model"].strip().lower()
    if name == "compnet":
        return CompNet(
            num_classes   = num_classes,
            embedding_dim = cfg["embedding_dim"],
            arcface_s     = cfg["arcface_s"],
            arcface_m     = cfg["arcface_m"],
            dropout       = cfg["dropout"],
            use_moe       = cfg.get("use_moe",    False),
            n_experts     = cfg.get("n_experts",  6),
            lora_rank     = cfg.get("lora_rank",  64),
            moe_top_k     = cfg.get("moe_top_k",  2),
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
            use_moe     = cfg.get("use_moe",      False),
            n_experts   = cfg.get("n_experts",    6),
            lora_rank   = cfg.get("lora_rank",    16),
        )
    else:
        raise ValueError(f"Unknown model: '{cfg['model']}'. "
                         f"Choose 'compnet', 'ccnet', or 'dinov2'.")
