import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
import random
from PIL import Image, ImageFilter, ImageEnhance
from pytorch_metric_learning import losses

# ----------------------------
# 1. Configuration & Toggles
# ----------------------------
batch_size = 32      
margin = 0.3
scale = 16
lr = 1e-3
weight_decay = 1e-4

pretrain_epochs = 20
epochs = 200

lamb = 0.2           # SupCon Weight
aux_weight = 0.2     # MoE Balance Weight
norm_weight = 1.0    # Scout Weight (Only used if Global Scout is ON)

# --- EXPANSION MODE ---
augmentation_expansion_mode = '4x' 

# --- VISIBILITY TOGGLE ---
use_aug_only_for_supcon = False 

# --- AUGMENTATION SETTINGS ---
use_dynamic_beta = True

# --- ARCHITECTURE TOGGLES ---
use_global_scout = True     
use_grl = True

num_experts = 3
top_k = 2

train_domains = ["460", "630"]   
test_domains  = ["940"]          

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 128

# ----------------------------
# 2. Augmentation Logic
# ----------------------------
general_aug = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.2),
    transforms.RandomApply([transforms.RandomAutocontrast()], p=0.2),
])

def label_guided_fft_mixup(x, labels):
    B, C, H, W = x.shape
    fft = torch.fft.fft2(x, dim=(-2, -1))
    amp, pha = torch.abs(fft), torch.angle(fft)
    amp_shifted = torch.fft.fftshift(amp, dim=(-2, -1))
    
    sorted_idx = torch.argsort(labels)
    target_indices = torch.roll(sorted_idx, shifts=B // 2, dims=0)
    amp_shifted_trg = amp_shifted[target_indices]
    
    if use_dynamic_beta:
        beta = np.random.uniform(0.01, 0.20)
    else:
        beta = 0.15
    
    b = int(np.floor(np.amin((H, W)) * beta))
    c_h, c_w = int(np.floor(H / 2.0)), int(np.floor(W / 2.0))
    
    if b > 0:
        amp_shifted[..., c_h-b:c_h+b, c_w-b:c_w+b] = amp_shifted_trg[..., c_h-b:c_h+b, c_w-b:c_w+b]
    
    amp_mixed = torch.fft.ifftshift(amp_shifted, dim=(-2, -1))
    x_aug = torch.fft.ifft2(amp_mixed * torch.exp(1j * pha), dim=(-2, -1)).real
    return torch.clamp(x_aug, 0, 1)

# ----------------------------
# 3. Dataset Class
# ----------------------------
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"

_all_hand_ids = set()
for root, _, files in os.walk(data_path):
    files.sort()
    for fname in files:
        if not fname.lower().endswith(".jpg"): continue
        parts = fname[:-4].split("_")
        if len(parts) != 4: continue
        subject_id, hand, spectrum, _ = parts
        if spectrum in set(train_domains):
            _all_hand_ids.add(f"{subject_id}_{hand}")
shared_label_map  = {h: i for i, h in enumerate(sorted(_all_hand_ids))}
num_total_classes = len(shared_label_map)
print(f"Shared identity space: {num_total_classes} identities (same in train & test)")

class CASIA_MS_Dataset(Dataset):
    def __init__(self, data_path, target_domains, shared_label_map, is_train=True):
        self.samples      = []
        self.hand_id_map  = shared_label_map
        self.domain_map   = {d: i for i, d in enumerate(target_domains)}
        self.is_train     = is_train
        self.to_tensor    = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.general_aug  = general_aug
        
        for root, _, files in os.walk(data_path):
            files.sort()
            for fname in files:
                if not fname.lower().endswith(".jpg"): continue
                parts = fname[:-4].split("_")
                if len(parts) != 4: continue
                subject_id, hand, spectrum, iteration = parts
                if spectrum not in target_domains: continue
                hand_id = f"{subject_id}_{hand}"
                if hand_id not in self.hand_id_map: continue
                img_path = os.path.join(root, fname)
                y_d = self.domain_map[spectrum]
                self.samples.append((img_path, self.hand_id_map[hand_id], y_d, spectrum))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, y_i, y_d, spectrum = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_orig = self.to_tensor(img)
        if self.is_train:
            img_aug_pil = self.general_aug(img)
            return img_orig, self.to_tensor(img_aug_pil), y_i, y_d
        return img_orig, y_i, y_d

# ----------------------------
# 4. Modules & Routers (Small CNN + MoE)
# ----------------------------
class GlobalDomainRouter(nn.Module):
    def __init__(self, num_domains=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 7, 4, 3), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.classifier = nn.Linear(32, num_domains)
    def forward(self, x): return self.classifier(self.features(x))

class SmallSharedTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=4, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(2, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=1)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3)) 

    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))
        x = self.pool3(self.act(self.conv4(x)))
        x = self.avgpool(x)
        return x

class DynamicMoEHead(nn.Module):
    def __init__(self, in_features, feature_dim, num_experts=3, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 1024), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 512),         nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, feature_dim),
            ) for _ in range(num_experts)
        ])

    def forward(self, x, gate, topk_probs, topk_idx):
        B = x.shape[0]
        out = torch.zeros(B, self.experts[0][-1].out_features, device=x.device)
        for i in range(self.num_experts):
            idx, k_idx = torch.where(topk_idx == i)
            if len(idx) == 0: continue
            expert_out = self.experts[i](x[idx])
            out[idx] += expert_out * topk_probs[idx, k_idx].unsqueeze(-1)
        return out

class SmallIntegratedMoEModel(nn.Module):
    def __init__(self, num_experts, top_k, feature_dim):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.trunk = SmallSharedTrunk()
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            flat_dim = self.trunk(dummy).view(1, -1).shape[1]
            
        self.moe_head = DynamicMoEHead(flat_dim, feature_dim, num_experts, top_k)
        self.scout = GlobalDomainRouter(num_experts)
        self.aux_loss = 0.0
        self.scout_logits = None
        
    def forward(self, x):
        self.scout_logits = self.scout(x)
        gate = F.softmax(self.scout_logits, dim=-1)
        topk_p, topk_i = torch.topk(gate, self.top_k, dim=-1)
        topk_p = topk_p / (topk_p.sum(-1, keepdim=True)+1e-6)
        
        frac = torch.zeros_like(gate).scatter_(1, topk_i, 1.0).mean(0)
        self.aux_loss = self.num_experts * torch.sum(frac * gate.mean(0))
        
        base_feats = self.trunk(x)
        flat_feats = base_feats.view(x.size(0), -1)
        final_features = self.moe_head(flat_feats, gate, topk_p, topk_i)
        
        return final_features

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out=128):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(True), nn.Linear(dim_in, dim_out))
    def forward(self, x): return F.normalize(self.head(x), dim=1)

class GradientReversal(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, x, alpha): ctx.alpha=alpha; return x.view_as(x)
    @staticmethod 
    def backward(ctx, grad_output): return grad_output.neg() * ctx.alpha, None

class DomainClassifier(nn.Module):
    def __init__(self, dim_in, num_domains):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, dim_in//2), nn.BatchNorm1d(dim_in//2), nn.ReLU(True), nn.Linear(dim_in//2, num_domains))
    def forward(self, x, alpha): return self.net(GradientReversal.apply(x, alpha))

# ----------------------------
# 5. Data Loading & Setup
# ----------------------------
print(f"Creating Training Dataset... (Batch Size: {batch_size})")
train_dataset = CASIA_MS_Dataset(data_path, train_domains, shared_label_map, is_train=True)
test_dataset  = CASIA_MS_Dataset(data_path, test_domains,  shared_label_map, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)

_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

class _ListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label

registration_samples = [
    (path, y_i)
    for path, y_i, y_d, spec
    in CASIA_MS_Dataset(data_path, train_domains, shared_label_map, is_train=False).samples
]
query_samples = [
    (path, y_i)
    for path, y_i, y_d, spec
    in test_dataset.samples
]

registration_loader = DataLoader(_ListDataset(registration_samples, _tf),
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)
query_loader        = DataLoader(_ListDataset(query_samples, _tf),
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)

print(f"  Source (registration): {train_domains}  —  {len(registration_samples)} images")
print(f"  Target (query)        : {test_domains[0]}  —  {len(query_samples)} images")

# --- MODEL INITIALIZATION ---
model = SmallIntegratedMoEModel(num_experts, top_k, embedding_dim).to(device)
proj_head = ProjectionHead(embedding_dim).to(device)

if use_grl:
    domain_classifier = DomainClassifier(embedding_dim, num_experts).to(device)
    criterion_domain = nn.CrossEntropyLoss().to(device)
else:
    domain_classifier = None; criterion_domain = None

# ----------------------------
# 6. Losses & Optimizer
# ----------------------------
criterion_arc    = losses.ArcFaceLoss(num_classes=num_total_classes, embedding_size=embedding_dim, margin=margin, scale=scale).to(device)
criterion_supcon = losses.SupConLoss(temperature=0.1).to(device)

all_params = list(model.parameters()) + list(criterion_arc.parameters()) + list(proj_head.parameters())
if use_grl: all_params += list(domain_classifier.parameters())

optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(pretrain_epochs + epochs), eta_min=1e-4)

# ----------------------------
# 7. Evaluation helpers
# ----------------------------
def compute_eer(scores_gen, scores_imp):
    if len(scores_gen) == 0 or len(scores_imp) == 0:
        return float("nan")
    gen  = np.array(scores_gen)
    imp  = np.array(scores_imp)
    thrs = np.linspace(min(gen.min(), imp.min()), max(gen.max(), imp.max()), 500)
    eer  = min(
        ((abs((imp >= t).mean() - (gen < t).mean()),
          ((imp >= t).mean() + (gen < t).mean()) / 2)
         for t in thrs),
        key=lambda x: x[0],
    )[1] * 100
    return eer

@torch.no_grad()
def _extract_embeddings(loader):
    feats, labels = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(device)
        emb  = model(imgs)
        feats.append(F.normalize(emb, p=2, dim=1).cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)

@torch.no_grad()
def evaluate(epoch, phase_name="Train"):
    model.eval()
    criterion_arc.eval()

    reg_feats, reg_labels = _extract_embeddings(registration_loader)
    qry_feats, qry_labels = _extract_embeddings(query_loader)

    G = len(reg_labels)
    Q = len(qry_labels)

    sim     = torch.mm(qry_feats, reg_feats.t())
    nn_idx  = sim.argmax(dim=1)
    pred    = reg_labels[nn_idx]
    acc     = (pred == qry_labels).float().mean().item() * 100

    sim_np = sim.numpy()
    gen_scores, imp_scores = [], []
    for i in range(Q):
        for j in range(G):
            s = sim_np[i, j]
            if qry_labels[i] == reg_labels[j]:
                gen_scores.append(s)
            else:
                imp_scores.append(s)
    eer = compute_eer(gen_scores, imp_scores)

    print(f"\n  ┌─ Evaluation | {phase_name} Epoch {epoch} "
          f"| registration={train_domains} ({G} imgs) "
          f"| query={test_domains[0]} ({Q} imgs)")
    print(f"  │  Rank-1 Accuracy : {acc:6.2f}%")
    print(f"  │  EER             : {eer:5.2f}%")
    print(f"  └{'─'*65}")

    return acc, eer

# ----------------------------
# 8. Training Loop
# ----------------------------
print(f"Starting Training | Mode: {augmentation_expansion_mode} | SupCon Only Aug: {use_aug_only_for_supcon} | Global Scout: {use_global_scout}")

# ==========================================
# PHASE 1: PRETRAINING (ARCFACE WARMUP ONLY)
# ==========================================
print(f"\n{'='*50}\n PHASE 1: PRETRAINING ({pretrain_epochs} Epochs) \n{'='*50}")

for epoch in range(pretrain_epochs):
    model.train(); proj_head.train(); criterion_arc.train()
    train_loss = 0.0; train_correct = 0; total_train = 0

    for batch_idx, (img_orig, img_spatial, y_i, y_d) in enumerate(tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{pretrain_epochs}")):
        img_orig, img_spatial, y_i, y_d = img_orig.to(device), img_spatial.to(device), y_i.to(device), y_d.to(device)

        images_list = [img_orig]; labels_list = [y_i]; domains_list = [y_d]

        if torch.rand(1) < 0.5:
            img_xor = label_guided_fft_mixup(img_orig, y_d)
        else:
            img_xor = img_spatial

        if augmentation_expansion_mode == '2x':
            images_list.append(img_xor); labels_list.append(y_i); domains_list.append(y_d)
        elif augmentation_expansion_mode == '4x':
            img_fft = label_guided_fft_mixup(img_orig, y_d)
            images_list.extend([img_spatial, img_fft, img_xor])
            labels_list.extend([y_i, y_i, y_i])
            domains_list.extend([y_d, y_d, y_d])

        images_all = torch.cat(images_list, dim=0)
        y_i_all    = torch.cat(labels_list, dim=0)

        optimizer.zero_grad()
        embeddings_all  = model(images_all)

        batch_curr = img_orig.size(0)
        emb_orig   = embeddings_all[:batch_curr]

        if use_aug_only_for_supcon:
            loss_arc = criterion_arc(emb_orig, y_i)
        else:
            loss_arc = criterion_arc(embeddings_all, y_i_all)

        loss_aux = model.aux_loss if use_global_scout else 0.0

        loss = loss_arc + (aux_weight * loss_aux)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        if use_aug_only_for_supcon:
            preds = criterion_arc.get_logits(emb_orig).argmax(dim=1)
            train_correct += (preds == y_i).sum().item()
            total_train   += img_orig.size(0)
        else:
            preds = criterion_arc.get_logits(embeddings_all).argmax(dim=1)
            train_correct += (preds == y_i_all).sum().item()
            total_train   += images_all.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc  = train_correct / total_train

    print(f"Pretrain Epoch [{epoch+1}/{pretrain_epochs}] Summary: Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f}")

    if (epoch + 1) % 5 == 0 or epoch == pretrain_epochs - 1:
        evaluate(epoch + 1, "Pretrain")

    scheduler.step()

# ==========================================
# PHASE 2: FULL TRAINING (ALL LOSSES ACTIVE)
# ==========================================
print(f"\n{'='*50}\n PHASE 2: FULL FINE-TUNING ({epochs} Epochs) \n{'='*50}")

for epoch in range(epochs):
    model.train(); proj_head.train(); criterion_arc.train()
    if use_grl: domain_classifier.train()
    train_loss = 0.0; train_correct = 0; total_train = 0

    for batch_idx, (img_orig, img_spatial, y_i, y_d) in enumerate(tqdm(train_loader, desc=f"Full Train Epoch {epoch+1}/{epochs}")):
        img_orig, img_spatial, y_i, y_d = img_orig.to(device), img_spatial.to(device), y_i.to(device), y_d.to(device)

        images_list = [img_orig]; labels_list = [y_i]; domains_list = [y_d]

        if torch.rand(1) < 0.5:
            img_xor = label_guided_fft_mixup(img_orig, y_d)
        else:
            img_xor = img_spatial

        if augmentation_expansion_mode == '2x':
            images_list.append(img_xor); labels_list.append(y_i); domains_list.append(y_d)
        elif augmentation_expansion_mode == '4x':
            img_fft = label_guided_fft_mixup(img_orig, y_d)
            images_list.extend([img_spatial, img_fft, img_xor])
            labels_list.extend([y_i, y_i, y_i])
            domains_list.extend([y_d, y_d, y_d])

        images_all = torch.cat(images_list, dim=0)
        y_i_all    = torch.cat(labels_list, dim=0)
        y_d_all    = torch.cat(domains_list, dim=0)

        optimizer.zero_grad()
        embeddings_all  = model(images_all)
        projections_all = proj_head(embeddings_all)

        batch_curr = img_orig.size(0)
        emb_orig   = embeddings_all[:batch_curr]

        norm_routing_loss = 0.0
        if use_aug_only_for_supcon:
            loss_arc = criterion_arc(emb_orig, y_i)
            if use_global_scout:
                norm_routing_loss = F.cross_entropy(model.scout_logits[:batch_curr], y_d)
        else:
            loss_arc = criterion_arc(embeddings_all, y_i_all)
            if use_global_scout:
                norm_routing_loss = F.cross_entropy(model.scout_logits, y_d_all)

        loss_con = criterion_supcon(projections_all, y_i_all)
        
        loss_domain = 0.0
        if use_grl:
            p = float(batch_idx + epoch * len(train_loader)) / (len(train_loader) * epochs)
            alpha_grl = 2. / (1. + np.exp(-10 * p)) - 1
            loss_domain = criterion_domain(domain_classifier(emb_orig, alpha_grl), y_d)
        
        loss_aux = model.aux_loss if use_global_scout else 0.0

        loss = loss_arc + (lamb * loss_con) + loss_domain + (aux_weight * loss_aux) + (norm_weight * norm_routing_loss)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        if use_aug_only_for_supcon:
            preds = criterion_arc.get_logits(emb_orig).argmax(dim=1)
            train_correct += (preds == y_i).sum().item()
            total_train   += img_orig.size(0)
        else:
            preds = criterion_arc.get_logits(embeddings_all).argmax(dim=1)
            train_correct += (preds == y_i_all).sum().item()
            total_train   += images_all.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc  = train_correct / total_train

    print(f"Full Train Epoch [{epoch+1}/{epochs}] Summary: Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f}")

    acc, eer = evaluate(epoch + 1, "Full Train")
    print("-" * 50)

    scheduler.step()
