"""
main.py — Federated Palmprint with MoE Routing.

Per round:
  1. Server → clients: global_model weights
  2. Each client: local_model ← copy(global), fine-tune on local data + cross-client FFT
  3. Evaluate: global (pre-finetune) vs local (post-finetune) vs MoE
  4. Upload local_models → server → FedAvg → new global_model

MoE: per-sample routing between global and local models.
  dp_mode=ideal:     oracle (true domain_id, upper bound)
  dp_mode=predicted: trained domain predictor on FFT amplitudes
"""

import os, json, time, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
    print(f"  Federated Palmprint — Global / Local / MoE")
    print(f"  Protocol: {protocol} | DP mode: {dp_mode}")
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

    # ── Domain predictor ──
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
        print(f"  Domain prediction: IDEAL (oracle)")

    # ══════════════════════════════════════════════════════════
    #  BUILD MODELS + LOADERS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  Building {n_clients} clients")
    print(f"{'─'*70}")

    # One global model (server state)
    # We use client 0's num_classes for the global model,
    # but embedding extraction doesn't use the arc layer
    global_model = build_model(cfg, client_data[0]["num_classes"]).to(device)

    # N local models (one per client, fine-tuned from global each round)
    local_models = []
    loaders = []

    for ci, cd in enumerate(client_data):
        n_cls = cd["num_classes"]
        spec = cd["spectrum"]

        local_models.append(build_model(cfg, n_cls).to(device))

        # Single loader: local data + cross-client FFT augmentation
        other_style = {k: v for k, v in style_bank.items() if k != ci}
        ds = FFTAugmentedDataset(
            cd["train_samples"], other_style, client_id=ci,
            M=cfg["M"], beta=cfg["beta"],
            img_side=cfg["img_side"], grayscale=True)
        loaders.append(DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], drop_last=True))

        print(f"    Client {ci} [{spec:>6}]  IDs={n_cls}  "
              f"samples={len(cd['train_samples'])}  aug={len(ds)}")

    # ── Test loaders ──
    gal_ds = PalmDataset(gallery_samples, cfg["img_side"])
    prb_ds = PalmDataset(probe_samples, cfg["img_side"])
    gallery_loader = DataLoader(gal_ds, batch_size=cfg["batch_size"],
                                 num_workers=cfg["num_workers"])
    probe_loader = DataLoader(prb_ds, batch_size=cfg["batch_size"],
                               num_workers=cfg["num_workers"])
    print(f"\n  Test: Gallery={len(gallery_samples)} | "
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

        # ── Step 1: Server → clients (copy global → local) ──
        global_state = global_model.state_dict()
        for ci in range(n_clients):
            # Load global weights into local model
            # Skip arc layer (different num_classes per client)
            local_state = local_models[ci].state_dict()
            for key, val in global_state.items():
                if key.startswith("arc."):
                    continue
                if key in local_state and local_state[key].shape == val.shape:
                    local_state[key] = val.clone()
            local_models[ci].load_state_dict(local_state)

        # ── Step 2: Fine-tune local models on local data + cross-client FFT ──
        # Fresh optimizer each round (local model was just reset from global)
        losses = []
        for ci in range(n_clients):
            opt = torch.optim.Adam(local_models[ci].parameters(), lr=cfg["lr"])
            for _ in range(cfg["local_epochs"]):
                loss, acc = train_one_epoch(
                    local_models[ci], loaders[ci], opt, device)
            losses.append(loss)

        elapsed = time.time() - t0
        avg_loss = np.mean(losses)

        if rnd % 5 == 0 or rnd == 1:
            print(f"  Rnd {rnd:3d}/{cfg['n_rounds']}  "
                  f"loss={avg_loss:.4f}  [{elapsed:.1f}s]")

        # ── Evaluate BEFORE FedAvg (global=server, local=fine-tuned) ──
        if rnd % cfg["eval_every"] == 0 or rnd == cfg["n_rounds"]:
            print(f"\n  ── Eval round {rnd} ──")

            global_model.eval()
            for m in local_models:
                m.eval()

            results = evaluate_all_modes(
                local_models, global_model, domain_predictor,
                gallery_loader, probe_loader, cfg, device)

            rg = results["global"]
            rl = results["local"]
            rm = results["moe"]

            print(f"    Global:  R1={rg['rank1']:6.2f}%  EER={rg['eer']:6.2f}%")
            print(f"    Local:   R1={rl['rank1']:6.2f}%  EER={rl['eer']:6.2f}%")
            print(f"    MoE:     R1={rm['rank1']:6.2f}%  EER={rm['eer']:6.2f}%")

            history.append({
                "round": rnd, "loss": avg_loss,
                "global": rg, "local": rl, "moe": rm,
            })
            print()

        # ── Step 3: FedAvg local models → new global ──
        fedavg(local_models, exclude_prefixes=("arc.",))
        # After FedAvg all local models have same shared weights = new global
        new_global_state = global_model.state_dict()
        for key in new_global_state:
            if key.startswith("arc."):
                continue
            src = local_models[0].state_dict()
            if key in src and new_global_state[key].shape == src[key].shape:
                new_global_state[key] = src[key].clone()
        global_model.load_state_dict(new_global_state)

    # ══════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  COMPLETE — {protocol}, dp_mode={dp_mode}")
    print(f"{'='*80}")

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
