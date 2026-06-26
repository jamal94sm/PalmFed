"""
main.py — Federated Palmprint with Domain-Aware Soft Routing.

Each client (one spectral domain) maintains:
  - personal_model: trained on local data only
  - general_model:  trained with FFT augmentation, FedAvg'd

Domain predictor (trained once on server from FFT amplitudes):
  predicts α = P(local_domain | test_batch)
  → final_emb = α·personal + (1-α)·general

Three evaluation modes:
  - personal:  baseline, personal model only
  - general:   baseline, generalized model only
  - soft:      domain-aware routing (α blending)
"""

import os, sys, json, time, copy, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

from configs import CONFIG, CASIA_SPECTRUMS, XJTU_VARIATIONS
from models import build_model, build_domain_predictor
from datasets import (PalmDataset, AugmentedDataset, FFTAugmentedDataset,
                       NormSingleROI)
from utils import (extract_style_template, build_dp_dataset,
                    train_domain_predictor, evaluate_with_routing)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ══════════════════════════════════════════════════════════════

def scan_casia_ms(data_root):
    """Scan CASIA-MS → dict[spectrum → list[(path, identity)]]."""
    from PIL import Image
    by_spectrum = defaultdict(list)
    all_ids = set()
    for fname in sorted(os.listdir(data_root)):
        if not fname.endswith((".jpg", ".png", ".bmp")):
            continue
        parts = fname.replace(".jpg", "").replace(".png", "").split("_")
        if len(parts) < 4:
            continue
        subj, hand, spec = parts[0], parts[1], parts[2]
        identity = f"{subj}_{hand}"
        by_spectrum[spec].append((os.path.join(data_root, fname), identity))
        all_ids.add(identity)
    return by_spectrum, sorted(all_ids)


def split_train_test_ids(all_ids, k_test, seed):
    rng = random.Random(seed)
    ids = list(all_ids)
    rng.shuffle(ids)
    n_test = int(len(ids) * k_test)
    test_ids = set(ids[:n_test])
    train_ids = set(ids[n_test:])
    return train_ids, test_ids


def split_gallery_probe(samples, gallery_ratio, seed):
    rng = random.Random(seed)
    by_id = defaultdict(list)
    for s in samples:
        by_id[s[1]].append(s)
    gallery, probe = [], []
    for identity, id_samples in by_id.items():
        rng.shuffle(id_samples)
        n_gal = max(1, int(len(id_samples) * gallery_ratio))
        gallery.extend(id_samples[:n_gal])
        probe.extend(id_samples[n_gal:])
    return gallery, probe


def build_style_bank(by_spectrum, train_ids, img_side, spectrums):
    """Build FFT amplitude templates for each client."""
    from PIL import Image
    style_bank = {}
    for spec in spectrums:
        templates = []
        for path, identity in by_spectrum[spec]:
            if identity not in train_ids:
                continue
            img = Image.open(path).convert("L").resize(
                (img_side, img_side), Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            tmpl = extract_style_template(img_np)
            templates.append(tmpl)
        style_bank[spec] = templates
    return style_bank


def build_test_loaders(by_spectrum, test_ids, cfg, client_spectrum):
    """
    Build gallery/probe loaders for a client based on test_domain toggle.

    test_domain = "same"  → only client's own spectrum
    test_domain = "cross" → all OTHER spectrums
    test_domain = "all"   → all spectrums
    """
    spectrums = sorted(by_spectrum.keys())
    test_domain = cfg["test_domain"]

    test_sets = {}
    for spec in spectrums:
        if test_domain == "same" and spec != client_spectrum:
            continue
        if test_domain == "cross" and spec == client_spectrum:
            continue

        samples = [(p, identity) for p, identity in by_spectrum[spec]
                   if identity in test_ids]
        if not samples:
            continue

        # Build label map for this test set
        id_map = {}
        for _, identity in samples:
            if identity not in id_map:
                id_map[identity] = len(id_map)

        labeled = [(p, id_map[identity]) for p, identity in samples]
        gal, prb = split_gallery_probe(labeled, cfg["gallery_ratio"],
                                        cfg["random_seed"])

        gal_ds = PalmDataset(gal, cfg["img_side"])
        prb_ds = PalmDataset(prb, cfg["img_side"])
        gal_loader = DataLoader(gal_ds, batch_size=cfg["batch_size"],
                                shuffle=False, num_workers=cfg["num_workers"])
        prb_loader = DataLoader(prb_ds, batch_size=cfg["batch_size"],
                                shuffle=False, num_workers=cfg["num_workers"])

        test_sets[spec] = {
            "gallery_loader": gal_loader,
            "probe_loader": prb_loader,
            "n_gallery": len(gal),
            "n_probe": len(prb),
            "n_ids": len(id_map),
        }

    return test_sets


# ══════════════════════════════════════════════════════════════
#  FEDERATED TRAINING
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
            domain_ids = None
        else:
            continue

        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if domain_ids is not None:
            logits = model(imgs, labels, domain_ids.to(device))
        else:
            logits = model(imgs, labels)

        loss = ce(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(1, len(loader)), 100.0 * correct / max(total, 1)


def fedavg(models, exclude_prefixes=("arc.",)):
    """FedAvg: average all params except excluded prefixes."""
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

    print(f"\n{'='*80}")
    print(f"  Federated Palmprint + Domain Predictor")
    print(f"  Model: {cfg['model']} | Routing: {cfg['routing_mode']}")
    print(f"  DP: arch={cfg['dp_arch']}, input={cfg['dp_input']}")
    print(f"  Test domain: {cfg['test_domain']}")
    print(f"{'='*80}\n")

    results_dir = cfg["base_results_dir"].replace("{dataset}", cfg["dataset"])
    os.makedirs(results_dir, exist_ok=True)

    # ── Scan dataset ──
    by_spectrum, all_ids = scan_casia_ms(cfg["data_root"])
    spectrums = sorted(by_spectrum.keys())
    n_clients = len(spectrums)
    print(f"  Dataset: {len(all_ids)} identities, {n_clients} spectrums")
    for spec in spectrums:
        print(f"    {spec}: {len(by_spectrum[spec])} samples")

    # ── Split IDs ──
    train_ids, test_ids = split_train_test_ids(
        all_ids, cfg["k_test"], cfg["random_seed"])
    n_train_cls = len(train_ids)
    n_test_cls = len(test_ids)
    print(f"\n  Train IDs: {n_train_cls} | Test IDs: {n_test_cls}")

    # ── Build label maps per client ──
    client_label_maps = {}
    for spec in spectrums:
        ids_in_spec = sorted(set(identity for _, identity in by_spectrum[spec]
                                 if identity in train_ids))
        client_label_maps[spec] = {name: idx for idx, name in enumerate(ids_in_spec)}

    # ── Build style bank ──
    print(f"\n  Building FFT style bank...")
    style_bank = build_style_bank(by_spectrum, train_ids,
                                   cfg["img_side"], spectrums)
    for spec in spectrums:
        print(f"    {spec}: {len(style_bank[spec])} templates")

    # ══════════════════════════════════════════════════════════
    #  TRAIN DOMAIN PREDICTOR (once on server)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  Training Domain Predictor ({cfg['dp_arch'].upper()}, "
          f"input={cfg['dp_input']})")
    print(f"{'─'*70}")

    dp_features, dp_labels = build_dp_dataset(
        style_bank, spectrums,
        pool_size=cfg["dp_pool_size"], mode=cfg["dp_input"])
    print(f"  DP dataset: {len(dp_features)} samples, "
          f"{len(spectrums)} classes")

    domain_predictor = build_domain_predictor(cfg, n_clients)
    domain_predictor, dp_acc = train_domain_predictor(
        domain_predictor, dp_features, dp_labels, cfg, device)

    # ══════════════════════════════════════════════════════════
    #  BUILD CLIENT MODELS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  Building client models ({n_clients} clients)")
    print(f"{'─'*70}")

    personal_models = {}
    general_models = {}
    personal_opts = {}
    general_opts = {}

    for spec in spectrums:
        n_cls = len(client_label_maps[spec])
        personal_models[spec] = build_model(cfg, n_cls).to(device)
        general_models[spec] = build_model(cfg, n_cls).to(device)
        personal_opts[spec] = torch.optim.Adam(
            personal_models[spec].parameters(), lr=cfg["lr"])
        general_opts[spec] = torch.optim.Adam(
            general_models[spec].parameters(), lr=cfg["lr"])
        print(f"    {spec}: {n_cls} train IDs")

    # ── Build training dataloaders ──
    personal_loaders = {}
    general_loaders = {}
    for spec in spectrums:
        lmap = client_label_maps[spec]
        train_samples = [(p, lmap[identity])
                         for p, identity in by_spectrum[spec]
                         if identity in train_ids and identity in lmap]

        # Personal: local data only with spatial augmentation
        p_ds = AugmentedDataset(train_samples, cfg["img_side"],
                                 grayscale=True, client_id=0)
        personal_loaders[spec] = DataLoader(
            p_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], drop_last=True)

        # General: local + FFT augmented
        other_clients = [s for s in spectrums if s != spec]
        client_style_bank = {spectrums.index(s): style_bank[s]
                             for s in other_clients if style_bank[s]}
        g_ds = FFTAugmentedDataset(
            train_samples, client_style_bank,
            client_id=spectrums.index(spec),
            M=cfg["M"], beta=cfg["beta"],
            img_side=cfg["img_side"], grayscale=True)
        general_loaders[spec] = DataLoader(
            g_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], drop_last=True)

    # ── Build test loaders ──
    test_sets_per_client = {}
    for spec in spectrums:
        test_sets_per_client[spec] = build_test_loaders(
            by_spectrum, test_ids, cfg, spec)

    # ══════════════════════════════════════════════════════════
    #  FEDERATED TRAINING LOOP
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  Federated Training: {cfg['n_rounds']} rounds × "
          f"{cfg['local_epochs']} local epochs")
    print(f"{'─'*70}")

    history = []

    for rnd in range(1, cfg["n_rounds"] + 1):
        t0 = time.time()
        rnd_losses_p = {}
        rnd_losses_g = {}

        for spec in spectrums:
            # Personal model: local data only
            for _ in range(cfg["local_epochs"]):
                loss_p, acc_p = train_local_epoch(
                    personal_models[spec], personal_loaders[spec],
                    personal_opts[spec], device)
            rnd_losses_p[spec] = loss_p

            # General model: local + FFT aug
            for _ in range(cfg["local_epochs"]):
                loss_g, acc_g = train_local_epoch(
                    general_models[spec], general_loaders[spec],
                    general_opts[spec], device)
            rnd_losses_g[spec] = loss_g

        # FedAvg general models
        exclude = general_models[spectrums[0]].local_only_keys()
        fedavg(list(general_models.values()), exclude)

        elapsed = time.time() - t0
        avg_lp = np.mean(list(rnd_losses_p.values()))
        avg_lg = np.mean(list(rnd_losses_g.values()))
        print(f"  Round {rnd:3d}/{cfg['n_rounds']}  "
              f"loss_p={avg_lp:.4f}  loss_g={avg_lg:.4f}  "
              f"[{elapsed:.1f}s]")

        # ── Evaluation ──
        if rnd % cfg["eval_every"] == 0 or rnd == cfg["n_rounds"]:
            print(f"\n  ── Eval at round {rnd} ──")
            rnd_results = {}

            for ci, spec in enumerate(spectrums):
                test_sets = test_sets_per_client[spec]
                if not test_sets:
                    continue

                for test_spec, ts in test_sets.items():
                    key = f"{spec}→{test_spec}"

                    # Soft routing
                    res_soft = evaluate_with_routing(
                        personal_models[spec], general_models[spec],
                        domain_predictor,
                        ts["gallery_loader"], ts["probe_loader"],
                        ci, {**cfg, "routing_mode": "soft"}, device)

                    # Baselines
                    if cfg["eval_baselines"]:
                        res_p = evaluate_with_routing(
                            personal_models[spec], general_models[spec],
                            domain_predictor,
                            ts["gallery_loader"], ts["probe_loader"],
                            ci, {**cfg, "routing_mode": "personal"}, device)
                        res_g = evaluate_with_routing(
                            personal_models[spec], general_models[spec],
                            domain_predictor,
                            ts["gallery_loader"], ts["probe_loader"],
                            ci, {**cfg, "routing_mode": "general"}, device)

                        print(f"    {key:>12s} | "
                              f"P: R1={res_p['rank1']:5.1f} EER={res_p['eer']:5.1f} | "
                              f"G: R1={res_g['rank1']:5.1f} EER={res_g['eer']:5.1f} | "
                              f"S: R1={res_soft['rank1']:5.1f} EER={res_soft['eer']:5.1f} "
                              f"α={res_soft['alpha_probe']:.2f}")

                        rnd_results[key] = {
                            "personal": res_p,
                            "general": res_g,
                            "soft": res_soft,
                        }
                    else:
                        print(f"    {key:>12s} | "
                              f"R1={res_soft['rank1']:5.1f} EER={res_soft['eer']:5.1f} "
                              f"α={res_soft['alpha_probe']:.2f}")
                        rnd_results[key] = {"soft": res_soft}

            # Summary
            soft_r1 = np.mean([v["soft"]["rank1"]
                               for v in rnd_results.values()])
            soft_eer = np.mean([v["soft"]["eer"]
                                for v in rnd_results.values()])
            print(f"\n    Mean Soft: R1={soft_r1:.2f}% EER={soft_eer:.2f}%")

            if cfg["eval_baselines"]:
                p_r1 = np.mean([v["personal"]["rank1"]
                                for v in rnd_results.values()])
                g_r1 = np.mean([v["general"]["rank1"]
                                for v in rnd_results.values()])
                p_eer = np.mean([v["personal"]["eer"]
                                 for v in rnd_results.values()])
                g_eer = np.mean([v["general"]["eer"]
                                 for v in rnd_results.values()])
                print(f"    Mean Personal: R1={p_r1:.2f}% EER={p_eer:.2f}%")
                print(f"    Mean General:  R1={g_r1:.2f}% EER={g_eer:.2f}%")

            history.append({
                "round": rnd, "results": rnd_results,
                "loss_p": avg_lp, "loss_g": avg_lg,
            })
            print()

    # ══════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*80}")

    if history:
        last = history[-1]["results"]
        print(f"\n  {'Client→Test':>15s} │ {'Personal':>12s} │ "
              f"{'General':>12s} │ {'Soft (α)':>15s}")
        print(f"  {'─'*60}")
        for key in sorted(last.keys()):
            v = last[key]
            p = v.get("personal", {})
            g = v.get("general", {})
            s = v["soft"]
            print(f"  {key:>15s} │ "
                  f"R1={p.get('rank1',0):5.1f}% │ "
                  f"R1={g.get('rank1',0):5.1f}% │ "
                  f"R1={s['rank1']:5.1f}% α={s['alpha_probe']:.2f}")

    save_path = os.path.join(results_dir, "results.json")
    with open(save_path, "w") as f:
        json.dump({"config": cfg, "dp_accuracy": dp_acc,
                    "history": history}, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
