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


class CompNet(nn.Module):
    """
    CompNet = CB1 // CB2 // CB3 + FC(9708→emb_dim) + Dropout + ArcFace.
    FC input size 9708 is fixed for 128×128 input images.
    """
    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25):
        super().__init__()
        self.cb1 = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2 = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3 = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)
        self.fc   = nn.Linear(9708, embedding_dim)
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


class MoELayer(nn.Module):
    """
    Soft Mixture of Experts layer for domain-invariant feature learning.

    Integrated into DINOv2 between the backbone CLS token and ArcFace.
    Each expert is a two-layer MLP with a residual connection, specialising
    in a different feature transformation. The gating network produces soft
    weights so all experts receive gradients on every forward pass.

    Design rationale in the FL setting:
      n_experts matches the number of FL clients (one per domain). After
      FedAvg, the shared gating network learns universal routing — which
      input patterns benefit from which expert's transformation. Experts
      that were trained on different domains by different clients become
      complementary transformations in the shared model.

    Load balancing loss:
      Prevents expert collapse (all inputs routing to one expert) by
      penalising uneven utilisation. Standard auxiliary loss from the
      Switch Transformer paper: n_experts × sum(mean_gate² per expert).
      Returned alongside the output so training can add it to the total loss.

    Parameters
    ----------
    embed_dim  : int  input/output dimensionality (384 for DINOv2 ViT-S/14)
    n_experts  : int  number of experts — set to number of FL clients/domains
    hidden_dim : int  expert hidden layer width (bottleneck)
    """
    def __init__(self, embed_dim=384, n_experts=6, hidden_dim=128):
        super().__init__()
        self.n_experts = n_experts

        # gating: soft routing weights over all experts
        self.gate = nn.Linear(embed_dim, n_experts)

        # experts: independent 2-layer MLPs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor [B, embed_dim]  L2-normalised CLS token from backbone

        Returns
        -------
        out     : Tensor [B, embed_dim]  L2-normalised MoE output
        lb_loss : scalar Tensor          load balancing auxiliary loss
        """
        gates = torch.softmax(self.gate(x), dim=-1)    # [B, n_experts]

        # all expert outputs: [B, n_experts, D]
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=1)

        # weighted sum of expert outputs: [B, D]
        out = (gates.unsqueeze(-1) * expert_outs).sum(dim=1)

        # residual: experts refine rather than replace the backbone embedding
        out = F.normalize(x + out, p=2, dim=1)

        # load balancing loss — penalise non-uniform expert utilisation
        # target: each expert used equally (1/n_experts on average)
        mean_gates = gates.mean(dim=0)                 # [n_experts]
        lb_loss    = self.n_experts * (mean_gates ** 2).sum()

        return out, lb_loss


class DINOv2Model(nn.Module):
    """
    DINOv2 FL model = DINOBackbone + optional MoELayer + ArcFace.

    With use_moe=False (default):
      backbone(x) → CLS [B, 384] → ArcFace → logits
      get_embedding returns backbone CLS token

    With use_moe=True:
      backbone(x) → CLS [B, 384] → MoELayer → refined [B, 384] → ArcFace → logits
      get_embedding returns MoE-refined embedding

    The MoE layer is inserted between blocks 10-11 output and ArcFace —
    the last learned stage, while the frozen backbone blocks remain unchanged.

    Weight sharing in FL:
      backbone.* and moe.* → shared via FedAvg
      arc.*                → kept local (client-specific identity prototypes)

    Training: train_compnet_epoch handles the MoE load balancing loss by
    directly calling model.moe when present — no change to forward() return
    signature, maintaining compatibility with the rest of the pipeline.

    Input: RGB 224×224 with ImageNet normalisation.
    """
    def __init__(self, num_classes, arcface_s=16.0, arcface_m=0.30,
                 use_moe=False, n_experts=6, moe_hidden_dim=128):
        super().__init__()
        self.backbone = DINOBackbone()
        self.moe      = MoELayer(DINOBackbone.EMBED_DIM, n_experts, moe_hidden_dim) \
                        if use_moe else None
        self.arc      = ArcMarginProduct(DINOBackbone.EMBED_DIM, num_classes,
                                         s=arcface_s, m=arcface_m)

    def forward(self, x, y=None):
        emb = self.backbone(x)                          # [B, 384] L2-normalised
        if self.moe is not None:
            emb, _ = self.moe(emb)                     # refined [B, 384]
        return self.arc(emb, y)                         # [B, num_classes] logits

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised 384-d embedding — MoE-refined when enabled."""
        emb = self.backbone(x)
        if self.moe is not None:
            emb, _ = self.moe(emb)
        return emb


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
            num_classes    = num_classes,
            arcface_s      = cfg.get("dino_scale",      16.0),
            arcface_m      = cfg.get("dino_margin",      0.30),
            use_moe        = cfg.get("use_moe",          False),
            n_experts      = cfg.get("n_experts",        6),
            moe_hidden_dim = cfg.get("moe_hidden_dim",   128),
        )
    else:
        raise ValueError(f"Unknown model: '{cfg['model']}'. "
                         f"Choose 'compnet', 'ccnet', or 'dinov2'.")
