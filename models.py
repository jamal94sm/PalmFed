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


class _ResidualExpert(nn.Module):
    """
    Low-rank domain-specific residual adapter: Linear(in→rank) → ReLU → Linear(rank→out).
    B initialised to zero → zero contribution at round 0, learned from training.
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_normal_(self.A.weight, nonlinearity="relu")
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(F.relu(self.A(x)))


class MoEFC(nn.Module):
    """
    Mixture-of-Experts FC bottleneck for CompNet.
    Replaces the single nn.Linear(9708→512) with:
        output = base_FC(x) + expert[domain_id](x)

    base_FC  — shared across all clients via FedAvg; learns domain-invariant projection.
    expert[d] — low-rank residual (9708→rank→512); learns domain-specific correction.

    Domain routing:
      Training:  domain_ids tensor [B] selects one expert per sample (exact routing).
      Inference: domain_ids=None → base_FC only (no residual). Base FC carries the
                 full domain-invariant representation; residuals are training-time only.

    experts[k] after FedAvg:
      Client k contributes the strongest gradient to expert[k] (its own domain).
      Other clients contribute via FFT-augmented copies styled as domain k.
      Expert[k] converges toward the domain-k-specific correction, informed
      primarily by the client that owns that domain.
    """
    def __init__(self, in_features=9708, out_features=512,
                 n_experts=6, rank=64):
        super().__init__()
        self.n_experts = n_experts
        self.base      = nn.Linear(in_features, out_features)
        self.experts   = nn.ModuleList([
            _ResidualExpert(in_features, out_features, rank)
            for _ in range(n_experts)
        ])

    def forward(self, x, domain_ids=None):
        base_out = self.base(x)                              # [B, 512]
        if domain_ids is None:
            return base_out                                  # inference: base only
        residual = torch.zeros_like(base_out)
        for d in range(self.n_experts):
            mask = (domain_ids == d)
            if mask.any():
                residual[mask] = self.experts[d](x[mask])
        return base_out + residual


class CompNet(nn.Module):
    """
    CompNet = CB1 // CB2 // CB3 + FC(9708→512) + Dropout + ArcFace.

    Without MoE (use_moe=False, default):
      Three parallel Gabor competitive blocks → concat [B, 9708]
      → nn.Linear(9708→512) → Dropout → ArcFace

    With MoE (use_moe=True):
      Three parallel Gabor competitive blocks → concat [B, 9708]
      → MoEFC: base_FC(x) + expert[domain_id](x) → Dropout → ArcFace
      domain_ids passed during training; at inference domain_ids=None
      so only the base FC is used.
    """
    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25,
                 use_moe=False, n_experts=6, lora_rank=64):
        super().__init__()
        self.cb1     = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2     = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3     = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)
        self.use_moe = use_moe
        if use_moe:
            self.fc = MoEFC(9708, embedding_dim, n_experts, lora_rank)
        else:
            self.fc = nn.Linear(9708, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.arc  = ArcMarginProduct(embedding_dim, num_classes,
                                     s=arcface_s, m=arcface_m)

    def _backbone(self, x, domain_ids=None):
        feat = torch.cat([self.cb1(x).flatten(1),
                          self.cb2(x).flatten(1),
                          self.cb3(x).flatten(1)], dim=1)   # [B, 9708]
        if self.use_moe:
            return self.fc(feat, domain_ids)
        return self.fc(feat)

    def forward(self, x, y=None, domain_ids=None):
        return self.arc(self.drop(self._backbone(x, domain_ids)), y)

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised 512-d embedding — base FC only (no MoE residual)."""
        return F.normalize(self._backbone(x, domain_ids=None), p=2, dim=1)


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
    pass  # kept as placeholder — remove after confirming no external references


class DINOv2Model(nn.Module):
    """
    DINOv2 FL model = DINOBackbone + ArcFace.
    backbone(x) → L2-normalised 384-d CLS token → ArcFace → logits.
    Weight sharing: backbone.* via FedAvg; arc.* kept local.
    Input: RGB 224×224 with ImageNet normalisation.
    """
    def __init__(self, num_classes, arcface_s=16.0, arcface_m=0.30):
        super().__init__()
        self.backbone = DINOBackbone()
        self.arc      = ArcMarginProduct(DINOBackbone.EMBED_DIM, num_classes,
                                         s=arcface_s, m=arcface_m)

    def forward(self, x, y=None):
        return self.arc(self.backbone(x), y)

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised 384-d CLS embedding."""
        return self.backbone(x)


# ══════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ══════════════════════════════════════════════════════════════

def build_model(cfg, num_classes):
    """Instantiate and return the model specified in cfg["model"]."""
    name = cfg["model"].strip().lower()
    if name == "compnet":
        return CompNet(
            num_classes   = num_classes,
            embedding_dim = cfg["embedding_dim"],
            arcface_s     = cfg["arcface_s"],
            arcface_m     = cfg["arcface_m"],
            dropout       = cfg["dropout"],
            use_moe       = cfg.get("use_moe",   False),
            n_experts     = cfg.get("n_experts",  6),
            lora_rank     = cfg.get("lora_rank",  64),
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
