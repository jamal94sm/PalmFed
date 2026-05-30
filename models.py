# ==============================================================
#  models.py — CompNet and CCNet architectures
# ==============================================================
#
#  Dual-branch Gabor MoE (CompNet with use_moe=True):
#
#  Two PARALLEL Gabor extractor stacks + one shared FC + ArcFace.
#
#  base_gabor   (CB1/CB2/CB3) → base_feat   [B, 9708]
#    • Trained on ALL samples (real + FFT augmented).
#    • base_feat → shared FC(9708→512) → domain-invariant embedding.
#    • base_gabor.* and fc.* are FedAvg'd.
#
#  domain_gabor (CB1/CB2/CB3) → domain_feat [B, 9708]
#    • Trained ONLY on real own-domain samples (sentinel skips it).
#    • Domain-specific Gabor filters diverge from base_gabor over rounds.
#    • domain_gabor.* is NEVER FedAvg'd — stays local to owning client.
#
#  Fusion (feature space, before the single FC):
#    real sample  : fused = (1-w)*base_feat + w*domain_feat
#    FFT sentinel : fused = base_feat   (domain_gabor not called)
#    warmup       : fused = base_feat   (domain_gabor frozen)
#
#  emb = fc(fused)  → single shared projection for all samples.
#
#  Why feature-space fusion before one shared FC:
#    - Both branches produce 9708-d vectors in the same feature geometry
#      (same CompetitiveBlock architecture), so weighted averaging is
#      geometrically meaningful.
#    - A single FC sees domain-corrected features and learns one
#      domain-invariant projection; there is no embedding-space alignment
#      problem between two independent 512-d spaces.
#    - The FC is always FedAvg'd on the fused features, so GlobalFull
#      evaluation (using the local domain_gabor) composes correctly with
#      whatever FC version is active — local or global.
#
#  Domain reconstruction loss:
#    domain_gabor(real_feats) should predict the domain-specific component
#    of the Gabor features relative to the global batch mean:
#      target = mean(base_gabor(real)) - mean(base_gabor(all))
#    This gives domain_gabor an exclusive self-supervised task that
#    base_gabor cannot optimise, giving it a clear learning signal.
#
#  Weight management:
#    FedAvg'd  : base_gabor.*, fc.*
#    Local-only : arc.*, domain_gabor.*
#
#  Two global models:
#    GlobalBase : base_gabor + fc only  (domain_gabor not used)
#    GlobalFull : base_gabor + domain_gabor[k] + fc for domain k
#
# ==============================================================

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ══════════════════════════════════════════════════════════════
#  SHARED BUILDING BLOCKS
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
        y  = y0.view(1,-1).repeat(c_out, c_in, ksz, 1)
        x  = x0.view(-1,1).repeat(c_out, c_in, 1, ksz)
        x  = x.to(sigma.device); y = y.to(sigma.device)
        xt =  x*torch.cos(theta.view(-1,1,1,1)) + y*torch.sin(theta.view(-1,1,1,1))
        yt = -x*torch.sin(theta.view(-1,1,1,1)) + y*torch.cos(theta.view(-1,1,1,1))
        gb = -torch.exp(
            -0.5*((gamma*xt)**2+yt**2) / (8*sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2*math.pi*f.view(-1,1,1,1)*xt+psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        self.kernel = self._gen_bank(
            self.kernel_size, self.channel_in, self.channel_out,
            self.sigma, self.gamma, self.theta, self.f, self.psi)
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)


class CompetitiveBlock(nn.Module):
    """CB = LGC + soft-argmax + PPU."""
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


def make_gabor_stack(init_ratios=(1.0, 0.5, 0.25)):
    """Create three CompetitiveBlocks with given init_ratios."""
    cb1 = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=init_ratios[0])
    cb2 = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=init_ratios[1])
    cb3 = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=init_ratios[2])
    return cb1, cb2, cb3


def extract_gabor(cb1, cb2, cb3, x):
    """Run three CBs on x and concatenate outputs → [B, 9708]."""
    return torch.cat([cb1(x).flatten(1),
                      cb2(x).flatten(1),
                      cb3(x).flatten(1)], dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.s = s; self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
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


# ══════════════════════════════════════════════════════════════
#  COMPNET  —  Dual-Branch Gabor MoE
# ══════════════════════════════════════════════════════════════

class CompNet(nn.Module):
    """
    CompNet with dual parallel Gabor branches (MoE mode).

    Standard (use_moe=False):
      CB1/CB2/CB3 → feat [B,9708] → fc → drop → arc

    MoE (use_moe=True):
      base_gabor   (CB1b/CB2b/CB3b) → base_feat   [B,9708]  all samples
      domain_gabor (CB1d/CB2d/CB3d) → domain_feat [B,9708]  real only
      fused = (1-w)*base_feat + w*domain_feat               feature-space blend
      fc(fused) → drop → arc

    Sentinel (domain_id == -1, FFT samples):
      fused = base_feat   (domain_gabor branch not called)

    domain_gabor starts as a copy of base_gabor at init — same spectral
    coverage — then diverges as it trains exclusively on real samples.

    Weight management
    ─────────────────
    FedAvg'd  : base_gabor.* (cb1b/cb2b/cb3b)  +  fc.*
    Local-only : arc.*  +  domain_gabor.* (cb1d/cb2d/cb3d)
    """

    SENTINEL = -1   # domain_id value for FFT-augmented samples

    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25,
                 use_moe=False, expert_weight=0.5,
                 use_grl=False, n_domains=6,
                 # legacy kwargs silently ignored
                 lora_rank=None, gate_mode=None, gate_init=None,
                 gate_scale=None, n_experts=None):
        super().__init__()
        self.use_moe       = use_moe
        self.expert_weight = expert_weight

        if use_moe:
            # ── base branch (FedAvg'd) ────────────────────────────────────
            self.cb1b, self.cb2b, self.cb3b = make_gabor_stack()
            # ── domain branch (local only) ────────────────────────────────
            # Initialised as a deep copy of base so it starts from the same
            # spectral prior and diverges only through domain-specific data.
            self.cb1d = copy.deepcopy(self.cb1b)
            self.cb2d = copy.deepcopy(self.cb2b)
            self.cb3d = copy.deepcopy(self.cb3b)
            self._warmup = False
        else:
            # Standard single-branch CompNet
            self.cb1, self.cb2, self.cb3 = make_gabor_stack()

        self.fc   = nn.Linear(9708, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.arc  = ArcMarginProduct(embedding_dim, num_classes,
                                     s=arcface_s, m=arcface_m)
        self.use_grl = use_grl
        self.domain_classifier = (DomainClassifier(embedding_dim, n_domains)
                                   if use_grl else None)

        # routing counters
        self.register_buffer("_real_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_skip_count", torch.zeros(1, dtype=torch.long))

    # ── warmup control ───────────────────────────────────────────────────────

    def set_moe_warmup(self, warmup: bool):
        """Freeze / unfreeze domain_gabor branch."""
        if not self.use_moe:
            return
        self._warmup = warmup
        for p in list(self.cb1d.parameters()) + \
                 list(self.cb2d.parameters()) + \
                 list(self.cb3d.parameters()):
            p.requires_grad = not warmup

    @property
    def moe_is_warming_up(self):
        return self.use_moe and self._warmup

    # ── key classification ───────────────────────────────────────────────────

    def local_only_keys(self):
        """Keys that NEVER leave the client (excluded from FedAvg)."""
        if self.use_moe:
            return ("arc.", "cb1d.", "cb2d.", "cb3d.")
        return ("arc.",)

    # ── routing stats ────────────────────────────────────────────────────────

    def reset_routing_stats(self):
        self._real_count.zero_(); self._skip_count.zero_()

    def get_routing_stats(self):
        return {"real": int(self._real_count), "skip": int(self._skip_count)}

    # ── feature extraction ───────────────────────────────────────────────────

    def _base_feat(self, x):
        """Base Gabor features [B, 9708] — always computed."""
        if self.use_moe:
            return extract_gabor(self.cb1b, self.cb2b, self.cb3b, x)
        return extract_gabor(self.cb1, self.cb2, self.cb3, x)

    def _domain_feat(self, x):
        """Domain Gabor features [B, 9708] — MoE only."""
        return extract_gabor(self.cb1d, self.cb2d, self.cb3d, x)

    # ── fused feature (training + inference) ─────────────────────────────────

    def _fused_feat(self, x, domain_ids=None):
        """
        Compute weighted feature fusion for a batch.

        domain_ids : Tensor [B] | None
          >= 0  → real own-domain sample  → fused = (1-w)*base + w*domain
          == -1 → FFT sentinel            → fused = base  (domain branch skipped)
          None  → warmup / non-MoE        → fused = base

        Returns
        -------
        fused       : Tensor [B, 9708]
        base_feat   : Tensor [B, 9708]   — always returned for recon loss target
        domain_feat : Tensor [R, 9708] | None — domain branch output for real samples
        real_mask   : BoolTensor [B]    — which samples are real
        """
        base_feat = self._base_feat(x)           # [B, 9708]

        if not self.use_moe or self._warmup or domain_ids is None:
            return base_feat, base_feat, None, None

        real_mask = (domain_ids >= 0)             # FFT sentinel = False

        if not real_mask.any():
            return base_feat, base_feat, None, real_mask

        # Only run domain_gabor on real samples (saves compute + ensures
        # no gradient flows from FFT sentinel through domain branch)
        domain_feat_real = self._domain_feat(x[real_mask])   # [R, 9708]

        fused = base_feat.clone()
        fused[real_mask] = ((1.0 - self.expert_weight) * base_feat[real_mask]
                            + self.expert_weight        * domain_feat_real)

        if self.training:
            self._real_count += real_mask.sum().item()
            self._skip_count += (~real_mask).sum().item()

        return fused, base_feat, domain_feat_real, real_mask

    # ── backbone ─────────────────────────────────────────────────────────────

    def _backbone(self, x, domain_ids=None):
        """
        Full forward through Gabor + FC.
        Returns (emb [B,512], base_feat [B,9708], domain_feat [R,9708]|None, real_mask|None).
        """
        fused, base_feat, domain_feat, real_mask = self._fused_feat(x, domain_ids)
        emb = self.fc(fused)
        return emb, base_feat, domain_feat, real_mask

    # ── training forward ─────────────────────────────────────────────────────

    def forward(self, x, y=None, domain_ids=None):
        emb, _, _, _ = self._backbone(x, domain_ids)
        return self.arc(self.drop(emb), y)

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_embedding(self, x, domain_id=None):
        """
        L2-normalised 512-d embedding.
        domain_id : int  → fuse base + domain branches  (GlobalFull)
                    None → base branch only              (GlobalBase)
        """
        base_feat = self._base_feat(x)
        if self.use_moe and domain_id is not None and not self._warmup:
            domain_feat = self._domain_feat(x)
            fused = ((1.0 - self.expert_weight) * base_feat
                     + self.expert_weight        * domain_feat)
        else:
            fused = base_feat
        return F.normalize(self.fc(fused), p=2, dim=1)

    @torch.no_grad()
    def get_embedding_with_external_domain(self, x, ext_cb1d, ext_cb2d, ext_cb3d):
        """
        GlobalFull inference using an external client's domain_gabor modules.
        Server calls this after loading client k's domain_gabor state.
        """
        base_feat   = self._base_feat(x)
        domain_feat = extract_gabor(ext_cb1d, ext_cb2d, ext_cb3d, x)
        fused = ((1.0 - self.expert_weight) * base_feat
                 + self.expert_weight        * domain_feat)
        return F.normalize(self.fc(fused), p=2, dim=1)

    # ── domain reconstruction loss ────────────────────────────────────────────

    def compute_domain_recon_loss(self, base_feat, domain_feat_real, real_mask):
        """
        Domain reconstruction loss — gives domain_gabor an exclusive target.

        For real samples in a batch:
          base_all_mean  = mean(base_feat, dim=0)              [9708] all samples
          base_real_mean = mean(base_feat[real_mask], dim=0)   [9708] real centroid
          target         = (base_real_mean - base_all_mean).detach()
                           The domain-specific component of the base features:
                           what domain k looks like compared to the global average.

          prediction = domain_feat_real    [R, 9708]  (already computed in _backbone)

          L_recon = MSE(prediction, target.expand_as(prediction))

        Why this target:
          base_all_mean  ≈ domain-invariant (averaged over all 6 domains via FFT aug)
          base_real_mean = domain k centroid (identity variation averages out)
          target         = what is specific to domain k in the Gabor features
          domain_gabor is rewarded for producing this offset for EVERY sample
          → it learns to be identity-invariant and domain-homogeneous,
            the exact complement of base_gabor's identity-discriminative task.

        Note: target is computed from BASE features, not domain features.
          This means domain_gabor is anchored to the same feature geometry
          as base_gabor, preventing arbitrary divergence.

        Parameters
        ──────────
        base_feat        : Tensor [B, 9708]  detached — no grad needed
        domain_feat_real : Tensor [R, 9708]  with grad — domain_gabor output
        real_mask        : BoolTensor [B]

        Returns
        ───────
        loss : scalar Tensor
        """
        if domain_feat_real is None or not real_mask.any():
            return torch.tensor(0.0, device=base_feat.device)

        with torch.no_grad():
            base_all_mean  = base_feat.mean(dim=0)                # [9708]
            base_real_mean = base_feat[real_mask].mean(dim=0)     # [9708]
            target         = base_real_mean - base_all_mean        # [9708]

        target_exp = target.unsqueeze(0).expand_as(domain_feat_real)
        return F.mse_loss(domain_feat_real, target_exp)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def get_weight_diagnostics(self):
        if not self.use_moe:
            return None
        with torch.no_grad():
            # Cosine similarity between base and domain Gabor filter banks
            # High similarity → domain branch hasn't diverged yet (expected early)
            # Lower similarity → domain branch specialising
            def _flat_params(cb):
                # Include ALL parameters regardless of requires_grad
                # (during warmup domain_gabor params are frozen but still diagnosable)
                return torch.cat([p.data.flatten()
                                  for p in cb.parameters()])
            base_p   = torch.cat([_flat_params(self.cb1b),
                                   _flat_params(self.cb2b),
                                   _flat_params(self.cb3b)])
            domain_p = torch.cat([_flat_params(self.cb1d),
                                   _flat_params(self.cb2d),
                                   _flat_params(self.cb3d)])
            cos_sim  = float(F.cosine_similarity(
                base_p.unsqueeze(0), domain_p.unsqueeze(0)))
            divergence = 1.0 - cos_sim          # 0 = identical, 1 = orthogonal
            param_diff = float((base_p - domain_p).norm())
        return {
            "base_weight_norm"  : round(float(base_p.norm()), 3),
            "domain_weight_norm": round(float(domain_p.norm()), 3),
            "branch_cos_sim"    : round(cos_sim, 4),
            "branch_divergence" : round(divergence, 4),
            "branch_param_diff" : round(param_diff, 4),
            "gate_value"        : self.expert_weight,
        }

    def get_activation_diagnostics(self, probe_img):
        """
        probe_img : Tensor [N, 1, H, W]  — fixed probe images (not features).

        Returns feature-space and embedding-space statistics measuring
        how much the domain branch shifts the representation.
        """
        if not self.use_moe:
            return None
        with torch.no_grad():
            probe_img = probe_img.to(next(self.parameters()).device)
            base_feat   = self._base_feat(probe_img)      # [N, 9708]
            domain_feat = self._domain_feat(probe_img)    # [N, 9708]
            correction  = self.expert_weight * (domain_feat - base_feat)
            fused       = base_feat + correction

            base_emb  = self.fc(base_feat)
            fused_emb = self.fc(fused)

            feat_norm  = float(base_feat.norm(dim=1).mean())
            corr_norm  = float(correction.norm(dim=1).mean())
            feat_ratio = corr_norm / max(feat_norm, 1e-8)

            cos_sim = float(F.cosine_similarity(
                F.normalize(base_emb, dim=1),
                F.normalize(fused_emb, dim=1)).mean())

            emb_norm  = float(base_emb.norm(dim=1).mean())
            diff_norm = float((fused_emb - base_emb).norm(dim=1).mean())
            emb_ratio = diff_norm / max(emb_norm, 1e-8)

        return {
            "base_norm"            : round(feat_norm, 4),
            "feat_correction_ratio": round(feat_ratio, 4),
            "gated_base_ratio"     : round(emb_ratio, 4),
            "base_full_cos_sim"    : round(cos_sim, 4),
            "gate_value"           : self.expert_weight,
            "domain_gated_norm"    : round(float(correction.norm(dim=1).mean()), 4),
        }

    def get_grad_norms(self):
        if not self.use_moe:
            return None
        def _gnorm(modules):
            grads = [p.grad.norm() for m in modules
                     for p in m.parameters()
                     if p.grad is not None]
            return round(float(torch.stack(grads).mean()), 4) if grads else None
        return {
            "base_grad_norm"  : _gnorm([self.cb1b, self.cb2b, self.cb3b, self.fc]),
            "domain_grad_norm": _gnorm([self.cb1d, self.cb2d, self.cb3d]),
            "gate_grad_norm"  : None,   # no learned gate
        }


# ══════════════════════════════════════════════════════════════
#  CCNET
# ══════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature=temperature; self.contrast_mode=contrast_mode
        self.base_temperature=base_temperature

    def forward(self, features, labels=None, mask=None):
        device=features.device
        if len(features.shape)<3: raise ValueError('features needs [bsz,n_views,...]')
        if len(features.shape)>3: features=features.view(features.shape[0],features.shape[1],-1)
        B=features.shape[0]
        if labels is not None and mask is not None: raise ValueError('Cannot define both')
        elif labels is None and mask is None: mask=torch.eye(B,dtype=torch.float32,device=device)
        elif labels is not None:
            labels=labels.contiguous().view(-1,1)
            mask=torch.eq(labels,labels.T).float().to(device)
        else: mask=mask.float().to(device)
        nc=features.shape[1]; cf=torch.cat(torch.unbind(features,dim=1),dim=0)
        if self.contrast_mode=='one': af=features[:,0]; ac=1
        elif self.contrast_mode=='all': af=cf; ac=nc
        else: raise ValueError(f'Unknown mode: {self.contrast_mode}')
        adc=torch.div(torch.matmul(af,cf.T),self.temperature)
        lm,_=torch.max(adc,dim=1,keepdim=True); logits=adc-lm.detach()
        mask=mask.repeat(ac,nc)
        lmask=torch.scatter(torch.ones_like(mask),1,
               torch.arange(B*ac).view(-1,1).to(device),0)
        mask=mask*lmask
        exp_l=torch.exp(logits)*lmask
        lp=logits-torch.log(exp_l.sum(1,keepdim=True))
        mlpp=(mask*lp).sum(1)/mask.sum(1)
        return -(self.temperature/self.base_temperature)*mlpp.view(ac,B).mean()


class CCGaborConv2d(nn.Module):
    def __init__(self,channel_in,channel_out,kernel_size,stride=1,padding=0,init_ratio=1):
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
        x=self.gabor1(x); x1=self.wc*self.argmax(x)+self.ws*(self.argmax_x(x)+self.argmax_y(x))
        x1=self.pool(self.conv1(self.se1(x1)))
        x=self.gabor2(x); x2=self.wc*self.argmax(x)+self.ws*(self.argmax_x(x)+self.argmax_y(x))
        x2=self.pool(self.conv2(self.se2(x2)))
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
    EMBED_DIM=384
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
            expert_weight = cfg.get("moe_expert_weight", 0.5),
            use_grl       = cfg.get("use_grl",           False),
            n_domains     = cfg.get("n_domains",         6),
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
