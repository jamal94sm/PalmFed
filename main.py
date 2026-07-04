"""
main.py — Federated Palmprint: Global / Local evaluation.

Per round:
  1. Server → clients: global_model weights
  2. Each client: local_model ← copy(global), fine-tune
  3. Evaluate: global + per-client local on test set
  4. FedAvg local_models → new global_model
"""

import os, json, time, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from PIL import Image

from configs import get_config, CASIA_SPECTRUMS
from models import build_model, build_domain_predictor
from datasets import (PalmDataset, FFTAugmentedDataset,
                       get_federated_splits)
from utils import (extract_style_template, build_dp_dataset,
                    train_domain_predictor, evaluate_single_model,
                    extract_embeddings_routed)


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


def compute_eer_r1(prb_feats, prb_labels, gal_feats, gal_labels):
    from sklearn.metrics import roc_curve
    sim = prb_feats @ gal_feats.T
    top_idx = sim.argmax(dim=1)
    predicted = gal_labels[top_idx]
    rank1 = (predicted == prb_labels).float().mean().item() * 100
    genuine, impostor = [], []
    for i in range(len(prb_labels)):
        pid = prb_labels[i].item()
        sims = sim[i].numpy(); glabs = gal_labels.numpy()
        gen_mask = glabs == pid; imp_mask = glabs != pid
        if gen_mask.any(): genuine.extend(sims[gen_mask].tolist())
        if imp_mask.any(): impostor.extend(sims[imp_mask].tolist())
    genuine = np.array(genuine); impostor = np.array(impostor)
    all_lab = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    all_sc = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(all_lab, all_sc)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2) * 100
    return rank1, eer


def parse_overrides():
    """Parse command-line overrides for CONFIG dict."""
    import argparse
    p = argparse.ArgumentParser(description="Federated Palmprint")
    p.add_argument("--method", default="proposed",
                   choices=["proposed", "fedpalm", "psfed"])
    p.add_argument("--dataset", choices=["casiams", "xjtu"])
    p.add_argument("--eval_protocol", choices=["open_set", "closed_set"])
    p.add_argument("--closed_set_mode", choices=["holdout", "cross_spectrum"])
    p.add_argument("--dp_mode", choices=["ideal", "predicted"])
    p.add_argument("--dp_arch", choices=["mlp", "cnn", "transformer"])
    p.add_argument("--dp_input", choices=["style", "full"])
    p.add_argument("--n_rounds", type=int)
    p.add_argument("--local_epochs", type=int)
    p.add_argument("--n_ids", type=int)
    p.add_argument("--k_test", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--M", type=int)
    p.add_argument("--beta", type=float)
    p.add_argument("--eval_every", type=int)
    p.add_argument("--random_seed", type=int)
    p.add_argument("--data_root")
    p.add_argument("--xjtu_data_root")
    p.add_argument("--gallery_ratio", type=float)
    p.add_argument("--closed_set_sample_ratio", type=float)
    p.add_argument("--model", choices=["compnet", "ccnet"])
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

    print(f"\n{'='*80}")
    print(f"  Federated Palmprint — {method.upper()}")
    cs_mode = cfg.get("closed_set_mode", "holdout")
    proto_str = f"{protocol}" + (f" ({cs_mode})" if protocol == "closed_set" else "")
    print(f"  Protocol: {proto_str} | DP mode: {dp_mode}")
    print(f"{'='*80}\n")

    results_dir = cfg["base_results_dir"].replace("{dataset}", cfg["dataset"])
    os.makedirs(results_dir, exist_ok=True)

    # ── Data ──
    splits_path = cfg.get("splits_path")
    if splits_path and os.path.exists(splits_path):
        print(f"  Loading shared splits: {splits_path}")
        import pickle
        with open(splits_path, "rb") as f:
            (client_data, gallery_samples, probe_samples,
             test_label_map, spectra) = pickle.load(f)
    else:
        (client_data, gallery_samples, probe_samples,
         test_label_map, spectra) = get_federated_splits(cfg, cfg["random_seed"])
    n_clients = len(client_data)

    style_bank = build_style_bank(client_data, cfg["img_side"])

    # ── Domain predictor ──
    domain_predictor = None
    dp_acc = -1
    if dp_mode == "predicted":
        print(f"  Training Domain Predictor ({cfg['dp_arch'].upper()})...")
        dp_features, dp_labels = build_dp_dataset(
            style_bank, list(range(n_clients)),
            pool_size=cfg["dp_pool_size"], mode=cfg["dp_input"])
        domain_predictor = build_domain_predictor(cfg, n_clients)
        domain_predictor, dp_acc = train_domain_predictor(
            domain_predictor, dp_features, dp_labels, cfg, device)
    else:
        print(f"  Domain prediction: IDEAL (oracle)")

    # ── Models + loaders ──
    global_model = build_model(cfg, client_data[0]["num_classes"]).to(device)
    local_models = []
    loaders = []

    for ci, cd in enumerate(client_data):
        local_models.append(build_model(cfg, cd["num_classes"]).to(device))
        other_style = {k: v for k, v in style_bank.items() if k != ci}
        ds = FFTAugmentedDataset(
            cd["train_samples"], other_style, client_id=ci,
            M=cfg["M"], beta=cfg["beta"],
            img_side=cfg["img_side"], grayscale=True)
        loaders.append(DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], drop_last=True))
        print(f"    Client {ci} [{cd['spectrum']:>6}]  "
              f"IDs={cd['num_classes']}  aug={len(ds)}")

    # ── Global test loaders ──
    gal_ds = PalmDataset(gallery_samples, cfg["img_side"])
    prb_ds = PalmDataset(probe_samples, cfg["img_side"])
    global_gal_loader = DataLoader(gal_ds, batch_size=cfg["batch_size"],
                                    num_workers=cfg["num_workers"])
    global_prb_loader = DataLoader(prb_ds, batch_size=cfg["batch_size"],
                                    num_workers=cfg["num_workers"])
    print(f"\n  Global test: Gal={len(gallery_samples)} Prb={len(probe_samples)}")

    # ── Local test loaders (closed-set only) ──
    local_test_loaders = []
    if is_closed:
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
                    "n_gal": len(local_gal), "n_prb": len(local_prb),
                })
                print(f"    Local test [{cd['spectrum']:>6}]: "
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

        # Step 1: copy global → local
        global_state = global_model.state_dict()
        for ci in range(n_clients):
            local_state = local_models[ci].state_dict()
            for key, val in global_state.items():
                if key.startswith("arc."):
                    continue
                if key in local_state and local_state[key].shape == val.shape:
                    local_state[key] = val.clone()
            local_models[ci].load_state_dict(local_state)

        # Step 2: fine-tune local
        losses = []
        for ci in range(n_clients):
            opt = torch.optim.Adam(local_models[ci].parameters(), lr=cfg["lr"])
            for _ in range(cfg["local_epochs"]):
                loss, _ = train_one_epoch(
                    local_models[ci], loaders[ci], opt, device)
            losses.append(loss)

        elapsed = time.time() - t0
        avg_loss = np.mean(losses)

        if rnd % 5 == 0 or rnd == 1:
            print(f"  Rnd {rnd:3d}/{cfg['n_rounds']}  "
                  f"loss={avg_loss:.4f}  [{elapsed:.1f}s]")


        # ── Evaluation ──
        if rnd % cfg["eval_every"] == 0 or rnd == cfg["n_rounds"]:
            print(f"\n  ── Eval round {rnd} ──")
            global_model.eval()
            for m in local_models:
                m.eval()

            rnd_entry = {"round": rnd, "loss": avg_loss}

            # ────────────────────────────────────
            #  LOCAL EVALUATION (closed-set only)
            # ────────────────────────────────────
            if is_closed and local_test_loaders:
                cs_mode = cfg.get("closed_set_mode", "cross_spectrum")
                cs_label = ("cross-spectrum" if cs_mode == "cross_spectrum"
                            else "held-out samples")
                print(f"\n    LOCAL EVALUATION ({cs_label})")
                print(f"    {'Client':>8s} │ {'Local R1':>9s} {'Local EER':>10s}")
                print(f"    {'─'*32}")

                local_test_results = []
                for ci in range(n_clients):
                    lt = local_test_loaders[ci]
                    spec = client_data[ci]["spectrum"]
                    if lt is None:
                        local_test_results.append(None)
                        continue
                    rl = evaluate_single_model(
                        local_models[ci], lt["gal_loader"],
                        lt["prb_loader"], device)
                    local_test_results.append(rl)
                    print(f"    {spec:>8s} │ {rl['rank1']:>8.2f}% {rl['eer']:>9.3f}%")

                valid = [r for r in local_test_results if r is not None]
                if valid:
                    alr = np.mean([r["rank1"] for r in valid])
                    ale = np.mean([r["eer"] for r in valid])
                    print(f"    {'─'*32}")
                    print(f"    {'Avg':>8s} │ {alr:>8.2f}% {ale:>9.3f}%")
                    rnd_entry["local_test"] = {
                        "avg_local": {"rank1": alr, "eer": ale},
                        "per_client": local_test_results,
                    }

            # ────────────────────────────────────
            #  EVALUATION
            # ────────────────────────────────────
            print(f"\n    EVALUATION")
            print(f"    {'Client':>8s} │ {'Local R1':>9s} {'Local EER':>10s}")
            print(f"    {'─'*32}")

            client_global_results = []
            for ci in range(n_clients):
                spec = client_data[ci]["spectrum"]
                rl = evaluate_single_model(
                    local_models[ci], global_gal_loader,
                    global_prb_loader, device)
                client_global_results.append(rl)
                print(f"    {spec:>8s} │ {rl['rank1']:>8.2f}% {rl['eer']:>9.3f}%")

            avg_lr1 = np.mean([r["rank1"] for r in client_global_results])
            avg_leer = np.mean([r["eer"] for r in client_global_results])

            rg = evaluate_single_model(
                global_model, global_gal_loader, global_prb_loader, device)

            print(f"    {'─'*32}")
            print(f"    {'Avg Loc':>8s} │ {avg_lr1:>8.2f}% {avg_leer:>9.3f}%")
            print(f"    {'Global':>8s} │ {rg['rank1']:>8.2f}% {rg['eer']:>9.3f}%")

            rnd_entry["global_test"] = {
                "global": rg,
                "avg_local": {"rank1": avg_lr1, "eer": avg_leer},
                "per_client_local": client_global_results,
            }
            history.append(rnd_entry)
            print()

        # Step 3: FedAvg → new global
        fedavg(local_models, exclude_prefixes=("arc.",))
        new_state = global_model.state_dict()
        for key in new_state:
            if key.startswith("arc."):
                continue
            src = local_models[0].state_dict()
            if key in src and new_state[key].shape == src[key].shape:
                new_state[key] = src[key].clone()
        global_model.load_state_dict(new_state)

    # ══════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  COMPLETE — {protocol}")
    print(f"{'='*80}")

    print(f"\n  EVALUATION SUMMARY")
    print(f"  {'Rnd':>5} │ {'Global':>18s} │ {'Avg Local':>18s}")
    print(f"  {'':>5} │ {'R1':>8s} {'EER':>9s} │ {'R1':>8s} {'EER':>9s}")
    print(f"  {'─'*46}")

    for h in history:
        gt = h["global_test"]
        rg = gt["global"]; rl = gt["avg_local"]
        print(f"  {h['round']:>5d} │ "
              f"{rg['rank1']:>7.2f}% {rg['eer']:>8.3f}% │ "
              f"{rl['rank1']:>7.2f}% {rl['eer']:>8.3f}%")

    if is_closed and any("local_test" in h for h in history):
        print(f"\n  LOCAL EVALUATION SUMMARY")
        print(f"  {'Rnd':>5} │ {'Avg Local':>18s}")
        print(f"  {'':>5} │ {'R1':>8s} {'EER':>9s}")
        print(f"  {'─'*28}")
        for h in history:
            lt = h.get("local_test", {})
            al = lt.get("avg_local", {})
            if al:
                print(f"  {h['round']:>5d} │ "
                      f"{al['rank1']:>7.2f}% {al['eer']:>8.3f}%")

    if history:
        for mode, key in [("Global", "global"), ("Avg Local", "avg_local")]:
            best = max(history,
                       key=lambda h: h["global_test"][key]["rank1"])
            r = best["global_test"][key]
            print(f"\n  Best {mode:>9s}: Rnd {best['round']}  "
                  f"R1={r['rank1']:.2f}%  EER={r['eer']:.3f}%")

    save_path = os.path.join(results_dir,
                              f"results_{protocol}.json")
    with open(save_path, "w") as f:
        json.dump({"config": cfg,
                    "history": history}, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
