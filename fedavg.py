"""
fedavg.py — FedAvg baseline.

Same as proposed but:
  - No FFT augmentation (spatial only, 2× paired views)
  - Loss: w1×CE(orig) + w2×CE(aug)
  - FedAvg each round
"""

import os, json, time, copy, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from configs import get_config
from model_fedpalm import compnet_fedpalm
from datasets import (PalmDataset, FFTAugmentedDataset,
                       get_federated_splits)
from utils import (evaluate_split, evaluate_local_avg,
                    emb_global, compute_eer)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, ce_criterion, device,
                    w1=0.5, w2=0.5):
    """w1×CE(orig) + w2×CE(aug) — spatial augmentation only."""
    model.train()
    total_loss = 0; correct = 0; total = 0
    for batch in loader:
        imgs_pair, labels, _ = batch
        '''
        img1, img2 = imgs_pair[0].to(device), imgs_pair[1].to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output1, _, _ = model(img1, labels)
        output2, _, _ = model(img2, labels)
        loss = w1 * ce_criterion(output1, labels) + w2 * ce_criterion(output2, labels)
        '''
        img1, img2 = imgs_pair[0].to(device), imgs_pair[1].to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        imgs_all = torch.cat([img1, img2], dim=0)           # [2B, C, H, W]
        labels_all = torch.cat([labels, labels], dim=0)     # [2B]
        output, _, _ = model(imgs_all, labels_all)
        loss = ce_criterion(output, labels_all)             # single CE on 2× batch
      
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        correct += (output[:len(labels)].argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(1, len(loader)), 100.0 * correct / max(total, 1)


def fedavg(models, exclude_prefixes=("arc",)):
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["casiams", "xjtu", "xpalm"])
    p.add_argument("--eval_protocol", choices=["open_set", "closed_set"])
    p.add_argument("--closed_set_mode", choices=["cross_spectrum"])
    p.add_argument("--n_rounds", type=int)
    p.add_argument("--random_seed", type=int)
    p.add_argument("--splits_path")
    p.add_argument("--n_ids", type=int)
    p.add_argument("--eval_every", type=int)
    args, _ = p.parse_known_args()

    cfg = get_config("fedavg")
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    set_seed(cfg["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    protocol = cfg.get("eval_protocol", "open_set")
    w1, w2 = cfg.get("w1", 0.5), cfg.get("w2", 0.5)

    print(f"\n{'='*70}")
    print(f"  FedAvg Baseline — CE only, spatial aug, FedAvg")
    print(f"  Protocol: {protocol} | Loss: {w1}×CE(orig) + {w2}×CE(aug)")
    print(f"{'='*70}\n")

    results_dir = cfg["base_results_dir"].format(
        dataset=cfg["dataset"], eval_protocol=protocol)
    os.makedirs(results_dir, exist_ok=True)

    # Data
    splits_path = cfg.get("splits_path")
    if splits_path and os.path.exists(splits_path):
        print(f"  Loading shared splits: {splits_path}")
        with open(splits_path, "rb") as f:
            (client_data, gallery_samples, probe_samples,
             test_label_map, spectra) = pickle.load(f)
    else:
        (client_data, gallery_samples, probe_samples,
         test_label_map, spectra) = get_federated_splits(cfg, cfg["random_seed"])
    n_clients = len(client_data)

    # Models
    global_model = compnet_fedpalm(
        num_classes=client_data[0]["num_classes"]).to(device)
    local_models = []
    optimizers = []
    schedulers = []
    loaders = []

    for ci, cd in enumerate(client_data):
        model = compnet_fedpalm(num_classes=cd["num_classes"]).to(device)
        local_models.append(model)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        optimizers.append(opt)
        schedulers.append(lr_scheduler.StepLR(
            opt, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"]))

        # Spatial-only paired aug (empty style_bank → no FFT)
        ds = FFTAugmentedDataset(
            cd["train_samples"], {}, client_id=ci,
            M=cfg["M"], beta=0.0,
            img_side=cfg["img_side"], grayscale=True)
        loaders.append(DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], drop_last=True))
        print(f"    Client {ci} [{cd['spectrum']}]  IDs={cd['num_classes']}  aug={len(ds)}")

    ce_criterion = nn.CrossEntropyLoss()

    gal_loader = DataLoader(PalmDataset(gallery_samples, cfg["img_side"]),
                             batch_size=cfg["batch_size"],
                             num_workers=cfg["num_workers"])
    prb_loader = DataLoader(PalmDataset(probe_samples, cfg["img_side"]),
                             batch_size=cfg["batch_size"],
                             num_workers=cfg["num_workers"])
    print(f"\n  Test: Gal={len(gallery_samples)} Prb={len(probe_samples)}")

    # Training
    print(f"\n{'─'*70}")
    print(f"  Training: {cfg['n_rounds']} rounds × {cfg['local_epochs']} ep")
    print(f"{'─'*70}")

    history = []
    for rnd in range(1, cfg["n_rounds"] + 1):
        t0 = time.time()

        # Copy global → local
        global_state = global_model.state_dict()
        for ci in range(n_clients):
            local_state = local_models[ci].state_dict()
            for key, val in global_state.items():
                if key.startswith("arc"):
                    continue
                if key in local_state and local_state[key].shape == val.shape:
                    local_state[key] = val.clone()
            local_models[ci].load_state_dict(local_state)

        # Fine-tune
        losses = []
        for ci in range(n_clients):
            for _ in range(cfg["local_epochs"]):
                loss, _ = train_one_epoch(
                    local_models[ci], loaders[ci], optimizers[ci],
                    ce_criterion, device, w1, w2)
            losses.append(loss)
            schedulers[ci].step()

        elapsed = time.time() - t0
        avg_loss = np.mean(losses)

        if rnd % 5 == 0 or rnd == 1:
            print(f"  Rnd {rnd:3d}/{cfg['n_rounds']}  loss={avg_loss:.4f}  [{elapsed:.1f}s]")

        # Eval
        if rnd % cfg["eval_every"] == 0 or rnd == cfg["n_rounds"]:
            print(f"\n  ── Eval round {rnd} ──")
            for m in local_models:
                m.eval()
            global_model.eval()

            print(f"\n    LOCAL EVAL (global scope)")
            print(f"    {'Client':>8s} │ {'Local R1':>9s} {'Local EER':>10s}")
            print(f"    {'─'*32}")
            avg_eer, avg_r1, per_client = evaluate_local_avg(
                local_models, gal_loader, prb_loader, device,
                client_names=[cd["spectrum"] for cd in client_data])
            print(f"    {'─'*32}")
            print(f"    {'Avg Loc':>8s} │ {avg_r1:>8.2f}% {avg_eer*100:>9.3f}%")

            g_eer, g_r1 = evaluate_split(
                lambda x: emb_global(global_model, x),
                gal_loader, prb_loader, device)
            print(f"    {'Global':>8s} │ {g_r1:>8.2f}% {g_eer*100:>9.3f}%")

            history.append({
                "round": rnd, "loss": avg_loss,
                "eval": {
                    "global": {"rank1": g_r1, "eer": g_eer * 100},
                    "avg_local": {"rank1": avg_r1, "eer": avg_eer * 100},
                    "per_client_local": [{"rank1": r, "eer": e * 100}
                                         for e, r in per_client],
                }
            })
            print()

        # FedAvg
        fedavg(local_models, exclude_prefixes=("arc",))
        new_state = global_model.state_dict()
        for key in new_state:
            if key.startswith("arc"):
                continue
            src = local_models[0].state_dict()
            if key in src and new_state[key].shape == src[key].shape:
                new_state[key] = src[key].clone()
        global_model.load_state_dict(new_state)

    # Summary
    print(f"\n{'='*70}")
    print(f"  COMPLETE — FedAvg, {protocol}")
    print(f"{'='*70}")
    for h in history:
        ev = h["eval"]
        print(f"  Rnd {h['round']:>5d} │ "
              f"Global: R1={ev['global']['rank1']:>6.2f}% EER={ev['global']['eer']:>7.3f}% │ "
              f"Avg Local: R1={ev['avg_local']['rank1']:>6.2f}% EER={ev['avg_local']['eer']:>7.3f}%")

    save_path = os.path.join(results_dir, f"results_{protocol}.json")
    with open(save_path, "w") as f:
        json.dump({"config": cfg, "history": history}, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
