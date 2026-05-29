# ==============================================================
#  main.py — FLClient, FLServer, federated training loop
# ==============================================================

import os
import copy
import time
import random
import pickle
import warnings
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from configs  import CONFIG
from models   import build_model, MultiExpertCompNet
from datasets import (PalmDataset, AugmentedDataset,
                      PairedDataset, FFTAugmentedDataset,
                      EvalDatasetDINO,
                      get_federated_splits)
from utils    import (extract_style_template, extract_radial_template,
                      evaluate_model,
                      train_compnet_epoch, train_ccnet_epoch,
                      CenterLoss)


# ══════════════════════════════════════════════════════════════
#  EVALUATION DATASET HELPER
# ══════════════════════════════════════════════════════════════

def make_eval_dataset(samples, cfg):
    if cfg["model"].strip().lower() == "dinov2":
        return EvalDatasetDINO(samples, cfg.get("dino_img_side", 224))
    return PalmDataset(samples, cfg["img_side"])


# ══════════════════════════════════════════════════════════════
#  EVALUATION WITH DOMAIN IDS
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model_with_domain(model, gallery_loader, probe_loader,
                                device, use_whitening=False):
    """
    Evaluation wrapper that passes domain_ids to MultiExpertCompNet.

    PalmDataset stores (path, label, domain_id) in samples; get_domain_ids()
    returns the domain_id list.  We inject domain_ids into the embedding
    extraction so the correct domain expert is selected per sample.

    Falls back to the standard evaluate_model() for non-MoE models.
    """
    from utils import extract_features, compute_eer, whiten_features

    is_moe = isinstance(model, MultiExpertCompNet)

    if not is_moe:
        return evaluate_model(model, gallery_loader, probe_loader,
                              device, use_whitening=use_whitening)

    def _extract(loader):
        model.eval()
        feats, labels = [], []
        for batch in loader:
            if len(batch) == 3:
                imgs, labs, dom_ids = batch
                dom_ids = dom_ids.to(device)
            else:
                imgs, labs = batch
                dom_ids = None
            imgs = imgs.to(device)
            emb  = model.get_embedding(imgs, dom_ids)
            emb_np = emb.cpu().numpy()
            nan_rows = np.isnan(emb_np).any(axis=1)
            if nan_rows.any():
                emb_np[nan_rows] = 0.0
            feats.append(emb_np)
            labels.append(labs.numpy())
        return np.concatenate(feats), np.concatenate(labels)

    gal_feats, gal_labels = _extract(gallery_loader)
    prb_feats, prb_labels = _extract(probe_loader)

    if use_whitening:
        gal_feats, prb_feats = whiten_features(gal_feats, prb_feats)

    sim_matrix  = np.nan_to_num(prb_feats @ gal_feats.T, nan=0.0)
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
#  FL CLIENT
# ══════════════════════════════════════════════════════════════

class FLClient:
    """
    One federated learning client.

    MultiExpertCompNet weight management
    ─────────────────────────────────────
    get_weights() / set_weights() delegate to MultiExpertCompNet's own
    methods which handle the 'experts.{i}.' key prefix and arc.* exclusion
    transparently.  Plain CompNet / CCNet / DINOv2 use the old dict-based
    approach unchanged.

    MoE warmup transparency
    ───────────────────────
    The client is unaware of warmup state — it trains whatever model it
    holds.  The server upgrades the global model via activate_moe() and
    pushes new weights to all clients before the next round.
    """

    def __init__(self, client_id, spectrum, train_samples, label_map,
                 num_classes, cfg, device):
        self.client_id     = client_id
        self.spectrum      = spectrum
        self.train_samples = train_samples
        self.label_map     = label_map
        self.num_classes   = num_classes
        self.cfg           = cfg
        self.device        = device
        self.model         = build_model(cfg, num_classes).to(device)

        model_name = cfg["model"].strip().lower()
        if cfg.get("use_center_loss", False):
            embed_dim = (cfg["embedding_dim"] if model_name == "compnet"
                         else 2048 if model_name == "ccnet" else 384)
            self.center_loss = CenterLoss(num_classes, embed_dim, device)
            self.center_optimizer = optim.SGD(
                self.center_loss.parameters(),
                lr=cfg.get("center_loss_lr", 0.5))
        else:
            self.center_loss      = None
            self.center_optimizer = None

        n_experts_info = (f"  [MoE: {1 + cfg.get('n_domains',6)} CompNets, "
                          f"warmup={cfg.get('moe_warmup_round',0)}]"
                          if cfg.get("use_moe") and model_name == "compnet"
                          else "")
        print(f"  Client {client_id} [{spectrum}] [{cfg['model']}] — "
              f"train IDs: {num_classes}  samples: {len(train_samples)}"
              f"{n_experts_info}")

    # ── weight management ─────────────────────────────────────────────────────

    def get_weights(self):
        """Return backbone weights for FedAvg — arc.* excluded."""
        if isinstance(self.model, MultiExpertCompNet):
            return self.model.get_weights()
        return {k: v.cpu().clone()
                for k, v in self.model.state_dict().items()
                if not k.startswith("arc.")}

    def set_weights(self, weights):
        """Load global backbone weights into local model."""
        if isinstance(self.model, MultiExpertCompNet):
            self.model.set_weights(weights)
            return
        local = self.model.state_dict()
        for k, v in weights.items():
            if k in local and local[k].shape == v.shape:
                local[k] = v.clone()
        self.model.load_state_dict(local)

    def activate_moe(self):
        """Upgrade local model from warmup to full MoE mode."""
        if isinstance(self.model, MultiExpertCompNet):
            self.model.activate_moe()

    # ── local training ────────────────────────────────────────────────────────

    def local_train(self, local_epochs, active_style_bank, M, rnd,
                    mean_bank=None):
        model_name = self.cfg["model"].strip().lower()
        is_dino    = model_name == "dinov2"
        img_side   = self.cfg.get("dino_img_side", 224) if is_dino else self.cfg["img_side"]
        grayscale  = not is_dino

        if model_name in ("compnet", "dinov2"):
            if active_style_bank and M > 1:
                dataset = FFTAugmentedDataset(
                    samples              = self.train_samples,
                    style_bank           = active_style_bank,
                    client_id            = self.client_id,
                    M                    = M,
                    beta                 = self.cfg["fft_beta"],
                    img_side             = img_side,
                    grayscale            = grayscale,
                    mean_bank            = mean_bank
                                           if self.cfg.get("domain_aware_mixing")
                                           else None,
                    prefer_distant       = self.cfg.get("prefer_distant_domain", True),
                    use_mean_template    = self.cfg.get("use_mean_template", False),
                    deterministic_donors = False,
                    fft_method           = self.cfg.get("fft_aug_method", "amplitude"),
                    local_only           = self.cfg.get("fft_local_only", False),
                )
            else:
                dataset = AugmentedDataset(self.train_samples, img_side,
                                           grayscale=grayscale)

        elif model_name == "ccnet":
            dataset = PairedDataset(
                samples    = self.train_samples,
                img_side   = img_side,
                style_bank = active_style_bank,
                client_id  = self.client_id,
                beta       = self.cfg["fft_beta"],
            )
        else:
            raise ValueError(f"Unknown model: '{self.cfg['model']}'")

        round_seed   = self.cfg["random_seed"] + rnd * 1000 + self.client_id
        train_loader = DataLoader(
            dataset,
            batch_size     = self.cfg["batch_size"],
            shuffle        = True,
            num_workers    = self.cfg["num_workers"],
            pin_memory     = True,
            worker_init_fn = lambda wid, s=round_seed: (
                np.random.seed(s + wid),
                random.seed(s + wid),
                torch.manual_seed(s + wid),
            ),
        )

        criterion = nn.CrossEntropyLoss()
        if is_dino:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg["lr"],
                                    weight_decay=self.cfg.get("dino_weight_decay", 1e-4))
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.cfg["lr"])

        lambda_c = (self.cfg.get("center_loss_weight", 0.0)
                    if self.cfg.get("use_center_loss") else 0.0)

        avg_loss, accuracy = 0.0, 0.0
        for _ in range(local_epochs):
            if model_name in ("compnet", "dinov2"):
                avg_loss, accuracy = train_compnet_epoch(
                    self.model, train_loader, criterion, optimizer, self.device,
                    center_loss         = self.center_loss,
                    center_optimizer    = self.center_optimizer,
                    lambda_center       = lambda_c,
                    lambda_style        = self.cfg.get("lambda_style", 0.0),
                    lambda_grl          = (self.cfg.get("lambda_grl", 0.0)
                                           if self.cfg.get("use_grl") else 0.0),
                    lambda_load_balance = 0.0,
                    lambda_supcon       = (self.cfg.get("lambda_supcon", 0.0)
                                           if self.cfg.get("use_supcon") else 0.0),
                    temperature         = self.cfg.get("temperature", 0.07),
                )
            elif model_name == "ccnet":
                avg_loss, accuracy = train_ccnet_epoch(
                    self.model, train_loader, criterion, optimizer, self.device,
                    ce_weight        = self.cfg.get("ce_weight", 0.8),
                    con_weight       = self.cfg.get("con_weight", 0.2),
                    temperature      = self.cfg.get("temperature", 0.07),
                    center_loss      = self.center_loss,
                    center_optimizer = self.center_optimizer,
                    lambda_center    = lambda_c,
                )

        return avg_loss, accuracy

    # ── style template extraction ─────────────────────────────────────────────

    def extract_style_templates(self):
        model_name = self.cfg["model"].strip().lower()
        is_dino    = model_name == "dinov2"
        img_side   = self.cfg.get("dino_img_side", 224) if is_dino \
                     else self.cfg["img_side"]
        mode       = "RGB" if is_dino else "L"
        fft_method = self.cfg.get("fft_aug_method", "amplitude")

        templates = []
        for path, _ in self.train_samples:
            img    = Image.open(path).convert(mode).resize(
                (img_side, img_side), Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            templates.append(
                extract_radial_template(img_np)
                if fft_method == "radial" and not is_dino
                else extract_style_template(img_np))

        method_label = (f"radial({self.cfg.get('fft_beta',0.1)})"
                        if fft_method == "radial" else "amplitude")
        print(f"  Client {self.client_id} [{self.spectrum}] "
              f"— {len(templates)} templates  [{method_label}]")
        return templates


# ══════════════════════════════════════════════════════════════
#  FL SERVER
# ══════════════════════════════════════════════════════════════

class FLServer:
    """
    Central server — FedAvg aggregation and global evaluation.

    MoE activation
    ──────────────
    After aggregation at round cfg["moe_warmup_round"] the server calls
    activate_moe() which:
      1. Calls global_model.activate_moe() — copies base weights into all
         domain experts on the global model.
      2. Calls client.activate_moe() on every client.
      3. Pushes upgraded global weights to all clients via set_weights().

    FedAvg with MultiExpertCompNet
    ───────────────────────────────
    aggregate() averages weights across clients using the same prefixed
    key scheme ('experts.{i}.{param}') that MultiExpertCompNet uses.
    Each expert's backbone is averaged independently across clients.
    """

    def __init__(self, num_classes, gallery_samples, probe_samples,
                 cfg, device, clients):
        self.cfg     = cfg
        self.device  = device
        self.clients = clients

        self.global_model = build_model(cfg, num_classes).to(device)

        # PalmDataset returns (img, label, domain_id) when samples have 3 elements
        # — the DataLoader therefore yields 3-tuples, which evaluate_model_with_domain
        # handles by forwarding domain_ids to get_embedding().
        self.gallery_loader = DataLoader(
            make_eval_dataset(gallery_samples, cfg),
            batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True)
        self.probe_loader = DataLoader(
            make_eval_dataset(probe_samples, cfg),
            batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True)

        print(f"  Server [{cfg['model']}] — "
              f"gallery: {len(gallery_samples)}  probe: {len(probe_samples)}")

    def get_global_weights(self):
        if isinstance(self.global_model, MultiExpertCompNet):
            return self.global_model.get_weights()
        return {k: v.cpu().clone()
                for k, v in self.global_model.state_dict().items()
                if not k.startswith("arc.")}

    def aggregate(self, client_weight_dicts):
        """FedAvg — averages all backbone weights across clients."""
        n        = len(client_weight_dicts)
        avg_dict = {}
        for key in client_weight_dicts[0].keys():
            stacked      = torch.stack(
                [client_weight_dicts[i][key].float() for i in range(n)], dim=0)
            avg_dict[key] = stacked.mean(dim=0)

        if isinstance(self.global_model, MultiExpertCompNet):
            self.global_model.set_weights(avg_dict)
        else:
            global_state = self.global_model.state_dict()
            global_state.update(avg_dict)
            self.global_model.load_state_dict(global_state)

    def activate_moe(self):
        """
        Upgrade global model and all clients to full MoE mode.

        Order:
          1. Upgrade global model (copies base → all domain experts).
          2. Upgrade each client model.
          3. Push new global weights to every client.
        """
        print(f"\n{'─'*56}")
        print(f"  MoE activation — {self.cfg.get('n_domains',6)} domain experts")
        print(f"  Warm-starting domain experts from trained base …")

        self.global_model.activate_moe()

        for client in self.clients:
            client.activate_moe()

        global_weights = self.get_global_weights()
        for client in self.clients:
            client.set_weights(global_weights)

        print(f"  All clients upgraded and synced.")
        print(f"{'─'*56}\n")

    def evaluate(self, use_whitening=False):
        return evaluate_model_with_domain(
            self.global_model,
            self.gallery_loader, self.probe_loader,
            self.device, use_whitening=use_whitening)


# ══════════════════════════════════════════════════════════════
#  AUGMENTATION HELPERS
# ══════════════════════════════════════════════════════════════

def resolve_aug_mode(cfg):
    if cfg.get("use_mixed_aug"):  return "mixed"
    if cfg.get("use_fft_aug"):    return "fft"
    return "spatial"


def get_active_style_bank(style_bank_full, rnd, cfg, is_dinov2):
    mode = resolve_aug_mode(cfg)
    if mode == "spatial": return {}
    if mode == "fft":     return style_bank_full
    switch = cfg.get("mixed_aug_round", cfg["n_rounds"] // 2)
    return style_bank_full if rnd > switch else {}


def aug_mode_label(rnd, cfg, is_dinov2):
    mode = resolve_aug_mode(cfg)
    if mode == "spatial": return "Spatial"
    if mode == "fft":     return "FFT"
    switch = cfg.get("mixed_aug_round", cfg["n_rounds"] // 2)
    return "FFT" if rnd > switch else "Spatial"


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    cfg  = CONFIG
    seed = cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir    = cfg["base_results_dir"]
    is_dinov2   = cfg["model"].strip().lower() == "dinov2"
    aug_mode    = resolve_aug_mode(cfg)
    use_moe     = cfg.get("use_moe", False) and cfg["model"].strip().lower() == "compnet"
    warmup_rnd  = cfg.get("moe_warmup_round", 0) if use_moe else 0
    os.makedirs(base_dir, exist_ok=True)

    # ── header ────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Federated Learning — Palmprint")
    print(f"  Dataset  : {cfg['dataset'].upper()}")
    print(f"  Model    : {cfg['model'].upper()}")
    print(f"  Device   : {device}")
    print(f"  Rounds   : {cfg['n_rounds']}   Local epochs: {cfg['local_epochs']}")
    if use_moe:
        n_dom = cfg.get("n_domains", 6)
        print(f"  MoE      : {1+n_dom} CompNets "
              f"(1 base + {n_dom} domain experts)")
        if warmup_rnd > 0:
            print(f"  Warmup   : {warmup_rnd} rounds (base only) → "
                  f"domain experts activated at round {warmup_rnd}")
        else:
            print(f"  Warmup   : none (full MoE from round 1)")
    print(f"  Aug mode : {aug_mode.upper()}   M={cfg['M']}   beta={cfg['fft_beta']}")
    print(f"  LR       : {cfg['lr']}")
    print(f"{'='*62}\n")

    # ── paths ─────────────────────────────────────────────────────────────────
    dataset_key       = cfg["dataset"].strip().lower()
    model_key         = cfg["model"].strip().lower()
    splits_path       = cfg["splits_path"].format(dataset=dataset_key)
    init_weights_path = cfg["init_weights_path"].format(
                            dataset=dataset_key, model=model_key)

    # ── splits ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    if os.path.exists(splits_path):
        print(f"Loading splits from: {splits_path}")
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        (client_data, gallery_samples, probe_samples,
         test_label_map, domain_names) = splits
    else:
        print(f"Building splits for {cfg['dataset'].upper()} …")
        splits = get_federated_splits(cfg, seed=seed)
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)
        (client_data, gallery_samples, probe_samples,
         test_label_map, domain_names) = splits

    num_classes = client_data[0]["num_classes"]
    n_clients   = len(client_data)
    print(f"\n  Clients: {n_clients}  ({domain_names})")
    print(f"  IDs/client: {num_classes}   Test: {len(test_label_map)}")
    print(f"  Gallery: {len(gallery_samples)}   Probe: {len(probe_samples)}\n")

    # ── clients ───────────────────────────────────────────────────────────────
    print("Initialising clients …")
    clients = []
    for i, cd in enumerate(client_data):
        clients.append(FLClient(
            client_id     = i,
            spectrum      = cd["spectrum"],
            train_samples = cd["train_samples"],
            label_map     = cd["label_map"],
            num_classes   = num_classes,
            cfg           = cfg,
            device        = device,
        ))

    # ── server ────────────────────────────────────────────────────────────────
    print("Initialising server …")
    server = FLServer(num_classes, gallery_samples, probe_samples,
                      cfg, device, clients)

    # ── initial weights ───────────────────────────────────────────────────────
    if os.path.exists(init_weights_path):
        print(f"\nLoading initial weights from: {init_weights_path}")
        init_state = torch.load(init_weights_path, map_location=device)
        server.global_model.load_state_dict(init_state, strict=False)
        global_weights = server.get_global_weights()
        for client in clients:
            client.set_weights(global_weights)
        print("  Initial weights loaded.")
    else:
        print(f"\nSaving initial weights to: {init_weights_path}")
        torch.save(server.global_model.state_dict(), init_weights_path)
        global_weights = server.get_global_weights()
        for client in clients:
            client.set_weights(global_weights)
        print("  Initial weights saved.")

    # ── results file ──────────────────────────────────────────────────────────
    results_path  = os.path.join(base_dir, "results.txt")
    client_header = "\t".join(
        f"Client{i}_EER(%)\tClient{i}_Rank1(%)" for i in range(n_clients))
    with open(results_path, "w") as f:
        f.write(f"Round\tAug\tMoE\tGlobal_EER(%)\tGlobal_Rank1(%)\t{client_header}\n")

    # ── style templates ───────────────────────────────────────────────────────
    print("\nExtracting style templates …")
    style_bank_full = {c.client_id: c.extract_style_templates() for c in clients}
    total = sum(len(v) for v in style_bank_full.values())
    print(f"  Style bank: {total} templates across {len(style_bank_full)} clients\n")

    mean_bank_full = {cid: np.mean(t, axis=0)
                      for cid, t in style_bank_full.items() if t}

    use_whitening = cfg.get("use_whitening", False)

    # ── round 0 ───────────────────────────────────────────────────────────────
    print("--- Round 0 (random init) ---")
    g_eer_0, g_rank1_0 = server.evaluate(use_whitening=use_whitening)
    print(f"  [Global]  EER={g_eer_0*100:.4f}%  Rank-1={g_rank1_0:.2f}%")
    with open(results_path, "a") as f:
        f.write(f"0\tInit\tFalse\t{g_eer_0*100:.4f}\t{g_rank1_0:.2f}\t"
                + "\t".join("-1\t-1" for _ in range(n_clients)) + "\n")

    recent_history = []
    moe_activated  = False

    # ── FL rounds ─────────────────────────────────────────────────────────────
    for rnd in range(1, cfg["n_rounds"] + 1):
        t_start = time.time()

        # MoE activation: triggered at the START of round warmup_rnd+1,
        # i.e. after the warmup-round aggregation has already run.
        if use_moe and not moe_activated and warmup_rnd > 0 and rnd == warmup_rnd + 1:
            server.activate_moe()
            moe_activated = True

        active_style_bank = get_active_style_bank(
            style_bank_full, rnd, cfg, is_dinov2)
        mode_label = aug_mode_label(rnd, cfg, is_dinov2)
        moe_label  = str(moe_activated or (use_moe and warmup_rnd == 0))

        # moe phase tag for console
        if use_moe:
            if warmup_rnd > 0 and not moe_activated:
                phase_tag = f" [warmup {rnd}/{warmup_rnd}]"
            else:
                phase_tag = f" [MoE: base+domain]"
        else:
            phase_tag = ""

        global_weights = server.get_global_weights()
        client_weights, client_metrics = [], []

        # ── local training ────────────────────────────────────────────────────
        for client in clients:
            client.set_weights(global_weights)
            loss, acc = client.local_train(
                cfg["local_epochs"], active_style_bank, cfg["M"], rnd,
                mean_bank=mean_bank_full if cfg.get("domain_aware_mixing") else None)

            c_eer, c_rank1 = evaluate_model_with_domain(
                client.model,
                server.gallery_loader, server.probe_loader,
                device, use_whitening=use_whitening)

            client_weights.append(client.get_weights())
            client_metrics.append({
                "client_id" : client.client_id,
                "spectrum"  : client.spectrum,
                "train_loss": round(loss, 6),
                "train_acc" : round(acc, 3),
                "eer"       : round(c_eer, 6),
                "rank1"     : round(c_rank1, 3),
            })

        # ── FedAvg + evaluation ───────────────────────────────────────────────
        server.aggregate(client_weights)
        g_eer, g_rank1 = server.evaluate(use_whitening=use_whitening)
        elapsed = time.time() - t_start

        recent_history.append((g_eer, g_rank1))
        if len(recent_history) > cfg.get("avg_last_rounds", 5):
            recent_history.pop(0)

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Round {rnd:04d}/{cfg['n_rounds']} "
              f"[{mode_label}]{phase_tag} | "
              f"Global EER={g_eer*100:.4f}%  Rank-1={g_rank1:.2f}%  "
              f"({elapsed:.1f}s)")
        for cm in client_metrics:
            print(f"  Client {cm['client_id']} [{cm['spectrum']:>14}] | "
                  f"loss={cm['train_loss']:.4f}  acc={cm['train_acc']:.1f}%  "
                  f"EER={cm['eer']*100:.3f}%  R1={cm['rank1']:.1f}%")

        client_cols = "\t".join(
            f"{cm['eer']*100:.4f}\t{cm['rank1']:.2f}" for cm in client_metrics)
        with open(results_path, "a") as f:
            f.write(f"{rnd}\t{mode_label}\t{moe_label}\t"
                    f"{g_eer*100:.4f}\t{g_rank1:.2f}\t{client_cols}\n")

    # ── final report ──────────────────────────────────────────────────────────
    n_avg     = len(recent_history)
    avg_eer   = sum(e for e, _ in recent_history) / n_avg
    avg_rank1 = sum(r for _, r in recent_history) / n_avg

    print(f"\n{'='*62}")
    print(f"  FL COMPLETE — {cfg['n_rounds']} rounds")
    print(f"  Dataset   : {cfg['dataset'].upper()}")
    print(f"  Model     : {cfg['model'].upper()}")
    print(f"  Aug mode  : {aug_mode.upper()}")
    if use_moe:
        print(f"  MoE       : {1+cfg.get('n_domains',6)} CompNets  "
              f"warmup={warmup_rnd}")
    print(f"  Avg EER   : {avg_eer*100:.4f}%  (last {n_avg} rounds)")
    print(f"  Avg Rank-1: {avg_rank1:.2f}%  (last {n_avg} rounds)")
    print(f"  Results   : {results_path}")
    print(f"{'='*62}")

    with open(results_path, "a") as f:
        f.write(f"\n# Average of last {n_avg} rounds\n")
        f.write(f"avg_{n_avg}\t—\t—\t{avg_eer*100:.4f}\t{avg_rank1:.2f}\n")


if __name__ == "__main__":
    main()
