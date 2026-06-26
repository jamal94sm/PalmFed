"""
main.py — Federated Palmprint with Domain-Aware Soft Routing.

Each client maintains:
  - personal_model: trained on local data + local FFT augmentation
  - general_model:  trained with cross-client FFT augmentation, FedAvg'd

Domain predictor (trained once on server):
  predicts α = P(local_domain | test_batch)
  → final_emb = α·personal + (1-α)·general

Two evaluation protocols:
  - open_set:   test IDs ≠ train IDs
  - closed_set: test IDs = train IDs (held-out samples)

Three routing baselines reported:
  - personal:  personal model only
  - general:   generalized model only
  - soft:      α-blended routing
"""

import os, json, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from PIL import Image

from configs import CONFIG, CASIA_SPECTRUMS
from models import build_model, build_domain_predictor
from datasets import (PalmDataset, FFTAugmentedDataset,
                       get_federated_splits, NormSingleROI)
from utils import (extract_style_template, build_dp_dataset,
                    train_domain_predictor, evaluate_with_routing)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
#  STYLE BANK CONSTRUCTION
# ══════════════════════════════════════════════════════════════

def build_style_bank_from_client_data(client_data, img_side):
    """Build FFT amplitude templates per client from training samples."""
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


# ══════════════════════════════════════════════════════════════
#  TRAINING HELPERS
# ══════════════════════════════════════════════════════════════

def train_local_epoch(model, loader, optimizer, device):
    """One epoch of local training."""
    model.train()
    total_loss = 0; correct = 0; total = 0
    ce = nn.CrossEntropyLoss()

    for batch in loader:
        if len(batch) == 3:
            imgs_pair, labels, domain_ids = batch
            imgs = imgs_pair[0] if isinstance(imgs_pair, list) else imgs_pair
        elif len(batch) == 2:
            imgs, labels = batch
        else:
            continue

        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs, labels)
        loss = ce(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(1, len(loader)), 100.0 * correct / max(total, 1)


def fedavg(models, exclude_prefixes=("arc.",)):
    """FedAvg all params except excluded prefixes."""
    avg_state = {}
    n = len(models)
    for key in models[0].state_dict():
        if any(key.startswith(pfx) for pfx in exclude_prefixes):
            continue
        avg_state[key] = sum(m.state_dict()[key].float()
                             for m in models) / n
    for m in models:
        state = m.state_dict()
        for key, val in avg_state.items():
            state[key] = val.to(state[key].dtype)
        m.load_state_dict(state)


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG.copy()
    set_seed(cfg["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    protocol = cfg.get("eval_protocol", "open_set")
    print(f"\n{'='*80}")
    print(f"  Federated Palmprint + Domain Predictor")
    print(f"  Protocol: {protocol} | Routing: {cfg['routing_mode']}")
    print(f"  DP: arch={cfg['dp_arch']}, input={cfg['dp_input']}")
    print(f"  Test domain: {cfg['test_domain']}")
    print(f"{'='*80}\n")

    results_dir = cfg["base_results_dir"].replace("{dataset}", cfg["dataset"])
    os.makedirs(results_dir, exist_ok=True)

    # ── Data splits ──
    (client_data, gallery_samples, probe_samples,
     test_label_map, spectra) = get_federated_splits(cfg, cfg["random_seed"])

    n_clients = len(client_data)

    # ── Build style bank ──
    print(f"\n  Building FFT style bank...")
    style_bank = build_style_bank_from_client_data(
        client_data, cfg["img_side"])
    for ci, cd in enumerate(client_data):
        print(f"    Client {ci} [{cd['spectrum']:>6}]: "
              f"{len(style_bank[ci])} templates")

    # ══════════════════════════════════════════════════════════
    #  TRAIN DOMAIN PREDICTOR (once on server)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  Domain Predictor ({cfg['dp_arch'].upper()}, "
          f"input={cfg['dp_input']})")
    print(f"{'─'*70}")

    client_ids_for_dp = list(range(n_clients))
    dp_features, dp_labels = build_dp_dataset(
        style_bank, client_ids_for_dp,
        pool_size=cfg["dp_pool_size"], mode=cfg["dp_input"])
    print(f"  DP dataset: {len(dp_features)} samples, "
          f"{n_clients} classes")

    domain_predictor = build_domain_predictor(cfg, n_clients)
    domain_predictor, dp_acc = train_domain_predictor(
        domain_predictor, dp_features, dp_labels, cfg, device)

    # ══════════════════════════════════════════════════════════
    #  BUILD CLIENT MODELS + DATALOADERS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  Building {n_clients} clients")
    print(f"{'─'*70}")

    personal_models = []
    general_models = []
    personal_opts = []
    general_opts = []
    personal_loaders = []
    general_loaders = []

    for ci, cd in enumerate(client_data):
        n_cls = cd["num_classes"]
        spec = cd["spectrum"]

        # Models
        p_model = build_model(cfg, n_cls).to(device)
        g_model = build_model(cfg, n_cls).to(device)
        personal_models.append(p_model)
        general_models.append(g_model)
        personal_opts.append(
            torch.optim.Adam(p_model.parameters(), lr=cfg["lr"]))
        general_opts.append(
            torch.optim.Adam(g_model.parameters(), lr=cfg["lr"]))

        # Personal loader: local FFT only (swap among own samples)
        local_style_bank = {ci: style_bank[ci]}  # self-templates only
        p_ds = FFTAugmentedDataset(
            cd["train_samples"], local_style_bank,
            client_id=ci, M=cfg["personal_M"], beta=cfg["personal_beta"],
            img_side=cfg["img_side"], grayscale=True)
        personal_loaders.append(DataLoader(
            p_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], drop_last=True))

        # General loader: cross-client FFT augmentation
        other_style_bank = {k: v for k, v in style_bank.items() if k != ci}
        g_ds = FFTAugmentedDataset(
            cd["train_samples"], other_style_bank,
            client_id=ci, M=cfg["M"], beta=cfg["beta"],
            img_side=cfg["img_side"], grayscale=True)
        general_loaders.append(DataLoader(
            g_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], drop_last=True))

        print(f"    Client {ci} [{spec:>6}]  "
              f"IDs={n_cls}  train={len(cd['train_samples'])}  "
              f"p_aug={len(p_ds)}  g_aug={len(g_ds)}")

    # ── Test loaders ──
    # Filter gallery/probe by test_domain toggle
    test_domain = cfg["test_domain"]

    def filter_by_domain(samples, client_idx):
        """Filter test samples based on test_domain toggle."""
        if test_domain == "all":
            return samples
        elif test_domain == "same":
            return [s for s in samples if s[2] == client_idx]
        elif test_domain == "cross":
            return [s for s in samples if s[2] != client_idx]
        return samples

    # Build per-client test loaders
    client_test_loaders = []
    for ci in range(n_clients):
        gal = filter_by_domain(gallery_samples, ci)
        prb = filter_by_domain(probe_samples, ci)
        if not gal or not prb:
            client_test_loaders.append(None)
            continue
        gal_ds = PalmDataset(gal, cfg["img_side"])
        prb_ds = PalmDataset(prb, cfg["img_side"])
        client_test_loaders.append({
            "gallery_loader": DataLoader(
                gal_ds, batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"]),
            "probe_loader": DataLoader(
                prb_ds, batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"]),
            "n_gallery": len(gal),
            "n_probe": len(prb),
        })

    # ══════════════════════════════════════════════════════════
    #  FEDERATED TRAINING LOOP
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  Training: {cfg['n_rounds']} rounds × "
          f"{cfg['local_epochs']} local epochs")
    print(f"{'─'*70}")

    history = []

    for rnd in range(1, cfg["n_rounds"] + 1):
        t0 = time.time()
        rnd_losses_p = []
        rnd_losses_g = []

        for ci in range(n_clients):
            for _ in range(cfg["local_epochs"]):
                lp, _ = train_local_epoch(
                    personal_models[ci], personal_loaders[ci],
                    personal_opts[ci], device)
                lg, _ = train_local_epoch(
                    general_models[ci], general_loaders[ci],
                    general_opts[ci], device)
            rnd_losses_p.append(lp)
            rnd_losses_g.append(lg)

        # FedAvg general models only
        exclude = general_models[0].local_only_keys()
        fedavg(general_models, exclude)

        elapsed = time.time() - t0
        avg_lp = np.mean(rnd_losses_p)
        avg_lg = np.mean(rnd_losses_g)

        if rnd % 5 == 0 or rnd == 1:
            print(f"  Round {rnd:3d}/{cfg['n_rounds']}  "
                  f"loss_p={avg_lp:.4f}  loss_g={avg_lg:.4f}  "
                  f"[{elapsed:.1f}s]")

        # ── Evaluation ──
        if rnd % cfg["eval_every"] == 0 or rnd == cfg["n_rounds"]:
            print(f"\n  ── Eval at round {rnd} ({protocol}) ──")

            all_results = {}
            for ci in range(n_clients):
                ts = client_test_loaders[ci]
                if ts is None:
                    continue
                spec = client_data[ci]["spectrum"]

                # All three routing modes
                results_ci = {}
                for mode in ["personal", "general", "soft"]:
                    res = evaluate_with_routing(
                        personal_models[ci], general_models[ci],
                        domain_predictor,
                        ts["gallery_loader"], ts["probe_loader"],
                        ci, {**cfg, "routing_mode": mode}, device)
                    results_ci[mode] = res

                all_results[spec] = results_ci
                rp = results_ci["personal"]
                rg = results_ci["general"]
                rs = results_ci["soft"]
                print(f"    {spec:>6s} │ "
                      f"P: R1={rp['rank1']:5.1f} EER={rp['eer']:5.1f} │ "
                      f"G: R1={rg['rank1']:5.1f} EER={rg['eer']:5.1f} │ "
                      f"S: R1={rs['rank1']:5.1f} EER={rs['eer']:5.1f} "
                      f"α={rs['alpha_probe']:.2f}")

            # Means
            for mode in ["personal", "general", "soft"]:
                vals = [v[mode] for v in all_results.values()]
                mr1 = np.mean([v["rank1"] for v in vals])
                meer = np.mean([v["eer"] for v in vals])
                tag = f"α={np.mean([v.get('alpha_probe',0) for v in vals]):.2f}" if mode == "soft" else ""
                print(f"    {'MEAN':>6s} │ {mode:>8s}: "
                      f"R1={mr1:5.1f}%  EER={meer:5.1f}%  {tag}")

            history.append({
                "round": rnd, "results": all_results,
                "loss_p": avg_lp, "loss_g": avg_lg,
            })
            print()

    # ══════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  COMPLETE ({protocol})")
    print(f"{'='*80}")

    if history:
        last = history[-1]["results"]
        print(f"\n  {'Client':>8s} │ {'Personal':>18s} │ "
              f"{'General':>18s} │ {'Soft':>22s}")
        print(f"  {'':>8s} │ {'R1':>8s} {'EER':>8s} │ "
              f"{'R1':>8s} {'EER':>8s} │ "
              f"{'R1':>8s} {'EER':>8s} {'α':>5s}")
        print(f"  {'─'*76}")
        for spec in sorted(last.keys()):
            v = last[spec]
            p = v["personal"]; g = v["general"]; s = v["soft"]
            print(f"  {spec:>8s} │ "
                  f"{p['rank1']:>7.1f}% {p['eer']:>7.1f}% │ "
                  f"{g['rank1']:>7.1f}% {g['eer']:>7.1f}% │ "
                  f"{s['rank1']:>7.1f}% {s['eer']:>7.1f}% "
                  f"{s['alpha_probe']:>5.2f}")

        print(f"  {'─'*76}")
        for mode in ["personal", "general", "soft"]:
            vals = [v[mode] for v in last.values()]
            mr1 = np.mean([v["rank1"] for v in vals])
            meer = np.mean([v["eer"] for v in vals])
            print(f"  {'MEAN':>8s} │ {mode:>8s}: "
                  f"R1={mr1:>6.1f}%  EER={meer:>6.1f}%")

    save_path = os.path.join(results_dir, f"results_{protocol}.json")
    with open(save_path, "w") as f:
        json.dump({"config": cfg, "dp_accuracy": dp_acc,
                    "history": history}, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
