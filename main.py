"""
main.py — Proposed: FedAvg + FFT augmentation + SupCon.

Uses compnet_fedpalm (same as baselines) for fair comparison.
Loss: w1 × CE(ArcFace) + w2 × SupCon
Augmentation: FFT cross-client amplitude swap (unique to proposed)

Per round:
  1. Server → clients: global weights (persistent optimizer + LR scheduler)
  2. Each client: fine-tune on local data + cross-client FFT
  3. Evaluate: global + per-client local
  4. FedAvg → new global
"""

import os, json, time, copy, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from PIL import Image

from configs import get_config
from model_fedpalm import compnet_fedpalm
from models import build_domain_predictor
from datasets import (PalmDataset, FFTAugmentedDataset,
                       get_federated_splits)
from loss_fedpalm import SupConLoss
from utils import (extract_style_template, build_dp_dataset,
                    train_domain_predictor, evaluate_split,
                    evaluate_local_avg, compute_eer)


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


def train_one_epoch(model, loader, optimizer, ce_criterion, con_criterion,
                    device, w1=0.8, w2=0.2):
    """One epoch: w1×CE + w2×SupCon on FFT-augmented paired views."""
    model.train()
    total_loss = 0; total_ce = 0; total_con = 0
    correct = 0; total = 0

    for batch in loader:
        imgs_pair, labels, domain_ids = batch
        img1, img2 = imgs_pair[0].to(device), imgs_pair[1].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward both views
        output1, fe1, _ = model(img1, labels)
        _,       fe2, _ = model(img2, labels)

        # CE on first view
        ce = ce_criterion(output1, labels)

        # SupCon on paired features
        fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        supcon = con_criterion(fe, labels)

        loss = w1 * ce + w2 * supcon

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce.item()
        total_con += supcon.item()
        correct += (output1.argmax(1) == labels).sum().item()
        total += labels.size(0)

    n = max(1, len(loader))
    acc = 100.0 * correct / max(total, 1)
    return total_loss / n, total_ce / n, total_con / n, acc


@torch.no_grad()
def emb_global(model, x):
    """Global model embedding — same as fedpalm/psfed."""
    model.eval()
    _, fe, _ = model(x, None, None)
    return fe


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


def parse_overrides():
    import argparse
    p = argparse.ArgumentParser(description="Proposed Method")
    p.add_argument("--method", default="proposed")
    p.add_argument("--dataset", choices=["casiams", "xjtu"])
    p.add_argument("--eval_protocol", choices=["open_set", "closed_set"])
    p.add_argument("--closed_set_mode", choices=["cross_spectrum"])
    p.add_argument("--local_eval_scope", choices=["client", "global"])
    p.add_argument("--dp_mode", choices=["ideal", "predicted"])
    p.add_argument("--dp_arch", choices=["mlp", "cnn", "transformer"])
    p.add_argument("--n_rounds", type=int)
    p.add_argument("--local_epochs", type=int)
    p.add_argument("--n_ids", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--M", type=int)
    p.add_argument("--beta", type=float)
    p.add_argument("--eval_every", type=int)
    p.add_argument("--random_seed", type=int)
    p.add_argument("--splits_path")
    args, _ = p.parse_known_args()
    method = args.method
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k != "method"}
    return method, overrides


def main():
    method, overrides = parse_overrides()
    cfg = get_config(method)
    cfg.update(overrides)
    set_seed(cfg["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    protocol = cfg.get("eval_protocol", "open_set")
    dp_mode = cfg.get("dp_mode", "ideal")
    is_closed = (protocol == "closed_set")
    local_eval_scope = cfg.get("local_eval_scope", "client")
    w1 = cfg.get("w1", 0.8)
    w2 = cfg.get("w2", 0.2)
    temperature = cfg.get("temperature", 0.07)

    print(f"\n{'='*80}")
    print(f"  Proposed Method — FedAvg + FFT + SupCon")
    cs_mode = cfg.get("closed_set_mode", "cross_spectrum")
    proto_str = f"{protocol}" + (f" ({cs_mode})" if is_closed else "")
    print(f"  Protocol: {proto_str} | Loss: {w1}×CE + {w2}×SupCon")
    print(f"  Model: compnet_fedpalm (grayscale)")
    print(f"{'='*80}\n")

    results_dir = cfg["base_results_dir"].format(
        dataset=cfg["dataset"], eval_protocol=protocol)
    os.makedirs(results_dir, exist_ok=True)

    # ── Data ──
    splits_path = cfg.get("splits_path")
    if splits_path and os.path.exists(splits_path):
        print(f"  Loading shared splits: {splits_path}")
        with open(splits_path, "rb") as f:
            (client_data, gallery_samples, probe_samples,
             test_label_map, spectra) = pickle.load(f)
        n_test = len(set(s[1] for s in gallery_samples + probe_samples))
        print(f"  Verified: {len(client_data)} clients, "
              f"gal={len(gallery_samples)}, prb={len(probe_samples)}, "
              f"test_IDs={n_test}")
    else:
        (client_data, gallery_samples, probe_samples,
         test_label_map, spectra) = get_federated_splits(cfg, cfg["random_seed"])
    n_clients = len(client_data)

    style_bank = build_style_bank(client_data, cfg["img_side"])

    # ── Models (compnet_fedpalm, same as baselines) ──
    print(f"\n  Building {n_clients} clients (compnet_fedpalm)")
    local_models = []
    optimizers = []
    schedulers = []
    loaders = []

    for ci, cd in enumerate(client_data):
        n_cls = cd["num_classes"]
        model = compnet_fedpalm(num_classes=n_cls).to(device)
        local_models.append(model)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        optimizers.append(opt)
        schedulers.append(lr_scheduler.StepLR(
            opt, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"]))

        # FFT-augmented paired loader (unique to proposed)
        other_style = {k: v for k, v in style_bank.items() if k != ci}
        ds = FFTAugmentedDataset(
            cd["train_samples"], other_style, client_id=ci,
            M=cfg["M"], beta=cfg["beta"],
            img_side=cfg["img_side"], grayscale=True)
        loaders.append(DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], drop_last=True))
        print(f"    Client {ci} [{cd['spectrum']}]  IDs={n_cls}  aug={len(ds)}")

    # Global model (FedAvg result)
    global_model = compnet_fedpalm(
        num_classes=client_data[0]["num_classes"]).to(device)

    # Loss
    ce_criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=temperature)

    # ── Test loaders ──
    gal_ds = PalmDataset(gallery_samples, cfg["img_side"])
    prb_ds = PalmDataset(probe_samples, cfg["img_side"])
    global_gal_loader = DataLoader(gal_ds, batch_size=cfg["batch_size"],
                                    num_workers=cfg["num_workers"])
    global_prb_loader = DataLoader(prb_ds, batch_size=cfg["batch_size"],
                                    num_workers=cfg["num_workers"])
    print(f"\n  Test: Gal={len(gallery_samples)} Prb={len(probe_samples)}")

    # Per-client test loaders (closed-set client scope)
    local_test_loaders = []
    if is_closed and local_eval_scope == "client":
        for ci, cd in enumerate(client_data):
            local_gal = cd.get("local_test_gal", [])
            local_prb = cd.get("local_test_prb", [])
            if local_gal and local_prb:
                gds = PalmDataset(local_gal, cfg["img_side"])
                pds = PalmDataset(local_prb, cfg["img_side"])
                local_test_loaders.append({
                    "gal_loader": DataLoader(gds, batch_size=cfg["batch_size"],
                                             num_workers=cfg["num_workers"]),
                    "prb_loader": DataLoader(pds, batch_size=cfg["batch_size"],
                                             num_workers=cfg["num_workers"]),
                })
                print(f"    Client test [{cd['spectrum']}]: "
                      f"Gal={len(local_gal)} Prb={len(local_prb)}")
            else:
                local_test_loaders.append(None)

    # ══════════════════════════════════════════════════════════
    #  TRAINING
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  Training: {cfg['n_rounds']} rounds × {cfg['local_epochs']} ep")
    print(f"{'─'*70}")

    history = []

    for rnd in range(1, cfg["n_rounds"] + 1):
        t0 = time.time()

        # Step 1: copy global → local (skip arc layer)
        global_state = global_model.state_dict()
        for ci in range(n_clients):
            local_state = local_models[ci].state_dict()
            for key, val in global_state.items():
                if key.startswith("arc"):
                    continue
                if key in local_state and local_state[key].shape == val.shape:
                    local_state[key] = val.clone()
            local_models[ci].load_state_dict(local_state)

        # Step 2: fine-tune local (persistent optimizer + scheduler)
        losses = []
        for ci in range(n_clients):
            for _ in range(cfg["local_epochs"]):
                loss, _, _, _ = train_one_epoch(
                    local_models[ci], loaders[ci], optimizers[ci],
                    ce_criterion, con_criterion, device, w1, w2)
            losses.append(loss)
            schedulers[ci].step()

        elapsed = time.time() - t0
        avg_loss = np.mean(losses)

        if rnd % 5 == 0 or rnd == 1:
            lr_now = optimizers[0].param_groups[0]["lr"]
            print(f"  Rnd {rnd:3d}/{cfg['n_rounds']}  "
                  f"loss={avg_loss:.4f}  lr={lr_now:.6f}  [{elapsed:.1f}s]")

        # ── Evaluation ──
        if rnd % cfg["eval_every"] == 0 or rnd == cfg["n_rounds"]:
            print(f"\n  ── Eval round {rnd} ──")
            global_model.eval()
            for m in local_models:
                m.eval()

            rnd_entry = {"round": rnd, "loss": avg_loss}

            emb_fn = lambda x, _m=None: None  # placeholder

            # Local eval
            if is_closed and local_eval_scope == "client" and local_test_loaders:
                print(f"\n    LOCAL EVAL (cross-spectrum, per-client scope)")
                print(f"    {'Client':>8s} │ {'Local R1':>9s} {'Local EER':>10s}")
                print(f"    {'─'*32}")
                client_results = []
                for ci in range(n_clients):
                    lt = local_test_loaders[ci]
                    spec = client_data[ci]["spectrum"]
                    if lt is None:
                        client_results.append(None); continue
                    m = local_models[ci]
                    m.eval()
                    eer, r1 = evaluate_split(
                        lambda x, _m=m: emb_global(_m, x),
                        lt["gal_loader"], lt["prb_loader"], device)
                    client_results.append({"rank1": r1, "eer": eer * 100})
                    print(f"    {spec:>8s} │ {r1:>8.2f}% {eer*100:>9.3f}%")
                valid = [r for r in client_results if r]
                avg_lr1 = np.mean([r["rank1"] for r in valid])
                avg_leer = np.mean([r["eer"] for r in valid])
                print(f"    {'─'*32}")
                print(f"    {'Avg Loc':>8s} │ {avg_lr1:>8.2f}% {avg_leer:>9.3f}%")
            else:
                # Same evaluate_local_avg as fedpalm/psfed
                print(f"\n    LOCAL EVAL (global scope)")
                print(f"    {'Client':>8s} │ {'Local R1':>9s} {'Local EER':>10s}")
                print(f"    {'─'*32}")
                avg_eer, avg_r1, per_client = evaluate_local_avg(
                    local_models, global_gal_loader, global_prb_loader,
                    device, client_names=[cd["spectrum"] for cd in client_data])
                avg_lr1 = avg_r1
                avg_leer = avg_eer * 100
                client_results = [{"rank1": r, "eer": e * 100}
                                  for e, r in per_client]
                print(f"    {'─'*32}")
                print(f"    {'Avg Loc':>8s} │ {avg_lr1:>8.2f}% {avg_leer:>9.3f}%")

            # Global eval (always full test set, same as baselines)
            g_eer, g_r1 = evaluate_split(
                lambda x: emb_global(global_model, x),
                global_gal_loader, global_prb_loader, device)
            rg = {"rank1": g_r1, "eer": g_eer * 100}
            print(f"    {'Global':>8s} │ {g_r1:>8.2f}% {g_eer*100:>9.3f}%")

            rnd_entry["eval"] = {
                "global": rg,
                "avg_local": {"rank1": avg_lr1, "eer": avg_leer},
                "per_client_local": client_results,
            }
            history.append(rnd_entry)
            print()

        # Step 3: FedAvg → new global
        fedavg(local_models, exclude_prefixes=("arc",))
        new_state = global_model.state_dict()
        for key in new_state:
            if key.startswith("arc"):
                continue
            src = local_models[0].state_dict()
            if key in src and new_state[key].shape == src[key].shape:
                new_state[key] = src[key].clone()
        global_model.load_state_dict(new_state)

    # ══════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  COMPLETE — {proto_str}")
    print(f"{'='*80}")

    print(f"\n  SUMMARY")
    print(f"  {'Rnd':>5} │ {'Global':>18s} │ {'Avg Local':>18s}")
    print(f"  {'':>5} │ {'R1':>8s} {'EER':>9s} │ {'R1':>8s} {'EER':>9s}")
    print(f"  {'─'*46}")

    for h in history:
        ev = h["eval"]
        rg = ev["global"]; rl = ev["avg_local"]
        print(f"  {h['round']:>5d} │ "
              f"{rg['rank1']:>7.2f}% {rg['eer']:>8.3f}% │ "
              f"{rl['rank1']:>7.2f}% {rl['eer']:>8.3f}%")

    if history:
        for mode, key in [("Global", "global"), ("Avg Local", "avg_local")]:
            best = max(history, key=lambda h: h["eval"][key]["rank1"])
            r = best["eval"][key]
            print(f"\n  Best {mode:>9s}: Rnd {best['round']}  "
                  f"R1={r['rank1']:.2f}%  EER={r['eer']:.3f}%")

    save_path = os.path.join(results_dir, f"results_{protocol}.json")
    with open(save_path, "w") as f:
        json.dump({"config": cfg, "history": history},
                  f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
