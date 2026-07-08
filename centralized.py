"""
centralized.py — Centralized baseline.

Single model trained on ALL client data combined.
No federated learning. Upper bound for comparison.
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
from datasets import (PalmDataset, AugmentedDataset,
                       get_federated_splits)
from utils import evaluate_split, emb_global, compute_eer


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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

    cfg = get_config("centralized")
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    set_seed(cfg["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    protocol = cfg.get("eval_protocol", "open_set")

    print(f"\n{'='*70}")
    print(f"  Centralized Baseline — Single model, all data")
    print(f"  Protocol: {protocol}")
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

    # Combine ALL client training data with unified label space
    all_train = []
    global_label_map = {}
    label_counter = 0
    for cd in client_data:
        for local_label, ident in sorted(
                [(v, k) for k, v in cd["label_map"].items()]):
            if ident not in global_label_map:
                global_label_map[ident] = label_counter
                label_counter += 1

    for cd in client_data:
        inv_map = {v: k for k, v in cd["label_map"].items()}
        for path, local_label in cd["train_samples"]:
            ident = inv_map[local_label]
            all_train.append((path, global_label_map[ident]))

    total_classes = label_counter
    print(f"  Combined training: {len(all_train)} samples, {total_classes} classes")
    print(f"  Test: Gal={len(gallery_samples)} Prb={len(probe_samples)}")

    # Single model with all classes
    model = compnet_fedpalm(num_classes=total_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"])
    ce_criterion = nn.CrossEntropyLoss()

    # 2× spatial augmentation
    train_ds = AugmentedDataset(all_train, cfg["img_side"], grayscale=True)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], drop_last=True)
    print(f"  Training dataset: {len(train_ds)} (with 2× aug)")

    gal_loader = DataLoader(PalmDataset(gallery_samples, cfg["img_side"]),
                             batch_size=cfg["batch_size"],
                             num_workers=cfg["num_workers"])
    prb_loader = DataLoader(PalmDataset(probe_samples, cfg["img_side"]),
                             batch_size=cfg["batch_size"],
                             num_workers=cfg["num_workers"])

    # Training
    n_epochs = cfg.get("n_rounds", 100)  # reuse n_rounds as n_epochs
    eval_every = cfg.get("eval_every", 10)

    print(f"\n{'─'*70}")
    print(f"  Training: {n_epochs} epochs")
    print(f"{'─'*70}")

    history = []
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0; correct = 0; total = 0

        for batch in train_loader:
            if len(batch) == 3:
                imgs, labels, _ = batch
            else:
                imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output, _, _ = model(imgs, labels)
            loss = ce_criterion(output, labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
            correct += (output.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = total_loss / max(1, len(train_loader))
        acc = 100.0 * correct / max(total, 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs}  loss={avg_loss:.4f}  "
                  f"acc={acc:.1f}%  [{elapsed:.1f}s]")

        # Eval
        if epoch % eval_every == 0 or epoch == n_epochs:
            print(f"\n  ── Eval epoch {epoch} ──")
            model.eval()

            g_eer, g_r1 = evaluate_split(
                lambda x: emb_global(model, x),
                gal_loader, prb_loader, device)
            print(f"    {'Global':>8s} │ {g_r1:>8.2f}% {g_eer*100:>9.3f}%")

            history.append({
                "round": epoch, "loss": avg_loss,
                "eval": {
                    "global": {"rank1": g_r1, "eer": g_eer * 100},
                }
            })
            print()

    # Summary
    print(f"\n{'='*70}")
    print(f"  COMPLETE — Centralized, {protocol}")
    print(f"{'='*70}")
    for h in history:
        ev = h["eval"]
        print(f"  Epoch {h['round']:>5d} │ "
              f"Global: R1={ev['global']['rank1']:>6.2f}% "
              f"EER={ev['global']['eer']:>7.3f}%")

    save_path = os.path.join(results_dir, f"results_{protocol}.json")
    with open(save_path, "w") as f:
        json.dump({"config": cfg, "history": history}, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
