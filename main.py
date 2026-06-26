"""
main.py — Federated Palmprint with Domain-Aware MoE Routing.

Each client maintains:
  - local_model:  trained on local data + local FFT augmentation
  - global_model: trained with cross-client FFT, FedAvg'd each round

Evaluation (global test set, open-set protocol):
  - Global:  global model only
  - Local:   per-sample local model (using sample's domain_id)
  - MoE:     soft routing — α·local + (1-α)·global per sample

Domain prediction modes:
  - ideal:     oracle — uses true domain_id (default, upper bound)
  - predicted: trained domain predictor on FFT amplitudes
"""

import os, json, time, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from PIL import Image

from configs import CONFIG
from models import build_model, build_domain_predictor
from datasets import (PalmDataset, FFTAugmentedDataset,
                       get_federated_splits)
from utils import (extract_style_template, build_dp_dataset,
                    train_domain_predictor, evaluate_all_modes)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_style_bank(client_data, img_side):
    """Build FFT amplitude templates per client."""
    style_bank = {}
    for ci, cd in enumerate(client_data):
        templates = []
        for path, _ in cd["train_samples"]:
            img = Image.open(path).convert("L").resize(
                (img_side, img_side), Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            templates.append(extract_style_template(img_np))
        style_bank[ci] = templates
    return style_bank


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0; correct = 0; total = 0
    ce = nn.CrossEntropyLoss()
    for batch in loader:
        if len(batch) == 3:
            imgs_pair, labels, _ = batch
            imgs = imgs_pair[0] if isinstance(imgs_pair, list) else imgs_pair
        else:
            imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs, labels)
        loss = ce(logits, labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(1, len(loader)), 100.0 * correct / max(total, 1)


def fedavg(models, exclude_prefixes=("arc.",)):
    avg_state = {}
    n = len(models)
    for key in models[0].state_dict():
        if any(key.startswith(pfx) for pfx in exclude_prefixes):
            continue
        avg_state[key] = sum(m.state_dict()[key].float() for m in models) / n
    for m in models:
        state = m.state_dict()
        for key, val in avg_state.items():
            state[key] = val.to(state[key].dtype)
        m.load_state_dict(state)


def main():
    cfg = CONFIG.copy()
    set_seed(cfg["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    protocol = cfg.get("eval_protocol", "open_set")
    dp_mode = cfg.get("dp_mode", "ideal")
    print(f"\n{'='*80}")
    print(f"  Federated Palmprint — Local / Global / MoE")
    print(f"  Protocol: {protocol} | DP mode: {dp_mode}")
    print(f"  DP arch: {cfg['dp_arch']} | DP input: {cfg['dp_input']}")
    print(f"{'='*80}\n")

    results_dir = cfg["base_results_dir"].replace("{dataset}", cfg["dataset"])
    os.makedirs(results_dir, exist_ok=True)

    # ── Data splits ──
    (client_data, gallery_samples, probe_samples,
     test_label_map, spectra) = get_federated_splits(cfg, cfg["random_seed"])
    n_clients = len(client_data)

    # ── Style bank ──
    print(f"\n  Building FFT style bank...")
    style_bank = build_style_bank(client_data, cfg["img_side"])
    for ci, cd in enumerate(client_data):
        print(f"    Client {ci} [{cd['spectrum']:>6}]: "
              f"{len(style_bank[ci])} templates")

    # ══════════════════════════════════════════════════════════
    #  DOMAIN PREDICTOR (trained once, used only if dp_mode=predicted)
    # ══════════════════════════════════════════════════════════
    domain_predictor = None
    dp_acc = -1

    if dp_mode == "predicted":
        print(f"\n{'─'*70}")
        print(f"  Training Domain Predictor ({cfg['dp_arch'].upper()})")
        print(f"{'─'*70}")
        dp_features, dp_labels = build_dp_dataset(
            style_bank, list(range(n_clients)),
            pool_size=cfg["dp_pool_size"], mode=cfg["dp_input"])
        domain_predictor = build_domain_predictor(cfg, n_clients)
        domain_predictor, dp_acc = train_domain_predictor(
            domain_predictor, dp_features, dp_labels, cfg, device)
    else:
        print(f"\n  Domain prediction: IDEAL (oracle, true domain_id)")

    # ══════════════════════════════════════════════════════════
    #  BUILD MODELS + LOADERS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  Building {n_clients} clients (local + global models)")
    print(f"{'─'*70}")

    local_models = []
    global_models = []
    local_opts = []
    global_opts = []
    local_loaders = []
    global_loaders = []

    for ci, cd in enumerate(client_data):
        n_cls = cd["num_classes"]
        spec = cd["spectrum"]

        l_model = build_model(cfg, n_cls).to(device)
        g_model = build_model(cfg, n_cls).to(device)
        local_models.append(l_model)
        global_models.append(g_model)
        local_opts.append(torch.optim.Adam(l_model.parameters(), lr=cfg["lr"]))
        global_opts.append(torch.optim.Adam(g_model.parameters(), lr=cfg["lr"]))

        # Local loader: FFT swap among own samples only
        local_style = {ci: style_bank[ci]}
        l_ds = FFTAugmentedDataset(
            cd["train_samples"], local_style, client_id=ci,
            M=cfg["local_M"], beta=cfg["local_beta"],
            img_side=cfg["img_side"], grayscale=True)
        local_loaders.append(DataLoader(
            l_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], drop_last=True))

        # Global loader: cross-client FFT augmentation
        other_style = {k: v for k, v in style_bank.items() if k != ci}
        g_ds = FFTAugmentedDataset(
            cd["train_samples"], other_style, client_id=ci,
            M=cfg["M"], beta=cfg["beta"],
            img_side=cfg["img_side"], grayscale=True)
        global_loaders.append(DataLoader(
            g_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], drop_last=True))

        print(f"    Client {ci} [{spec:>6}]  IDs={n_cls}  "
              f"local_aug={len(l_ds)}  global_aug={len(g_ds)}")

    # ── Global test loaders (gallery/probe with domain_id metadata) ──
    gal_ds = PalmDataset(gallery_samples, cfg["img_side"])
    prb_ds = PalmDataset(probe_samples, cfg["img_side"])
    gallery_loader = DataLoader(gal_ds, batch_size=cfg["batch_size"],
                                 num_workers=cfg["num_workers"])
    probe_loader = DataLoader(prb_ds, batch_size=cfg["batch_size"],
                               num_workers=cfg["num_workers"])
    print(f"\n  Test set: Gallery={len(gallery_samples)} | "
          f"Probe={len(probe_samples)}")

    # ══════════════════════════════════════════════════════════
    #  FEDERATED TRAINING
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  Training: {cfg['n_rounds']} rounds × "
          f"{cfg['local_epochs']} local epochs")
    print(f"{'─'*70}")

    history = []

    for rnd in range(1, cfg["n_rounds"] + 1):
        t0 = time.time()
        losses_l, losses_g = [], []

        for ci in range(n_clients):
            for _ in range(cfg["local_epochs"]):
                ll, _ = train_one_epoch(
                    local_models[ci], local_loaders[ci],
                    local_opts[ci], device)
                lg, _ = train_one_epoch(
                    global_models[ci], global_loaders[ci],
                    global_opts[ci], device)
            losses_l.append(ll)
            losses_g.append(lg)

        # FedAvg global models
        exclude = global_models[0].local_only_keys()
        fedavg(global_models, exclude)

        elapsed = time.time() - t0
        avg_ll = np.mean(losses_l)
        avg_lg = np.mean(losses_g)

        if rnd % 5 == 0 or rnd == 1:
            print(f"  Rnd {rnd:3d}/{cfg['n_rounds']}  "
                  f"local={avg_ll:.4f}  global={avg_lg:.4f}  "
                  f"[{elapsed:.1f}s]")

        # ── Evaluation ──
        if rnd % cfg["eval_every"] == 0 or rnd == cfg["n_rounds"]:
            print(f"\n  ── Eval round {rnd} ({dp_mode} domain) ──")

            # Set all models to eval
            for m in local_models + global_models:
                m.eval()

            results = evaluate_all_modes(
                local_models, global_models[0],  # global is same after FedAvg
                domain_predictor,
                gallery_loader, probe_loader,
                cfg, device)

            rg = results["global"]
            rl = results["local"]
            rm = results["moe"]

            print(f"    Global:  R1={rg['rank1']:6.2f}%  "
                  f"EER={rg['eer']:6.2f}%  "
                  f"(Gal={rg['n_gallery']} Prb={rg['n_probe']})")
            print(f"    Local:   R1={rl['rank1']:6.2f}%  "
                  f"EER={rl['eer']:6.2f}%")
            print(f"    MoE:     R1={rm['rank1']:6.2f}%  "
                  f"EER={rm['eer']:6.2f}%")

            history.append({
                "round": rnd,
                "loss_local": avg_ll, "loss_global": avg_lg,
                "global": rg, "local": rl, "moe": rm,
            })
            print()

    # ══════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  COMPLETE — {protocol}, dp_mode={dp_mode}")
    print(f"{'='*80}")

    # History table
    print(f"\n  {'Rnd':>5} │ {'Global':>16s} │ {'Local':>16s} │ "
          f"{'MoE':>16s}")
    print(f"  {'':>5} │ {'R1':>7s} {'EER':>7s} │ "
          f"{'R1':>7s} {'EER':>7s} │ {'R1':>7s} {'EER':>7s}")
    print(f"  {'─'*58}")

    for h in history:
        rg = h["global"]; rl = h["local"]; rm = h["moe"]
        print(f"  {h['round']:>5d} │ "
              f"{rg['rank1']:>6.1f}% {rg['eer']:>6.1f}% │ "
              f"{rl['rank1']:>6.1f}% {rl['eer']:>6.1f}% │ "
              f"{rm['rank1']:>6.1f}% {rm['eer']:>6.1f}%")

    # Best round for each mode
    if history:
        for mode in ["global", "local", "moe"]:
            best = max(history, key=lambda h: h[mode]["rank1"])
            print(f"\n  Best {mode:>6s}: Rnd {best['round']}  "
                  f"R1={best[mode]['rank1']:.2f}%  "
                  f"EER={best[mode]['eer']:.2f}%")

    save_path = os.path.join(results_dir,
                              f"results_{protocol}_{dp_mode}.json")
    with open(save_path, "w") as f:
        json.dump({"config": cfg, "dp_accuracy": dp_acc,
                    "history": history}, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
