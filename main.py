"""
main.py — Federated Palmprint with MoE Routing.

Per round:
  1. Server → clients: global_model weights
  2. Each client: local_model ← copy(global), fine-tune on local data + cross-client FFT
  3. Evaluate + print per-client local & MoE, then averages + global
  4. Upload local_models → server → FedAvg → new global_model
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
    """Compute R1 + EER from pre-extracted features."""
    from sklearn.metrics import roc_curve
    sim = prb_feats @ gal_feats.T
    top_idx = sim.argmax(dim=1)
    predicted = gal_labels[top_idx]
    rank1 = (predicted == prb_labels).float().mean().item() * 100

    genuine, impostor = [], []
    for i in range(len(prb_labels)):
        pid = prb_labels[i].item()
        sims = sim[i].numpy()
        glabs = gal_labels.numpy()
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

    # ── Data ──
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

    # Test loaders
    gal_ds = PalmDataset(gallery_samples, cfg["img_side"])
    prb_ds = PalmDataset(probe_samples, cfg["img_side"])
    gallery_loader = DataLoader(gal_ds, batch_size=cfg["batch_size"],
                                 num_workers=cfg["num_workers"])
    probe_loader = DataLoader(prb_ds, batch_size=cfg["batch_size"],
                               num_workers=cfg["num_workers"])
    print(f"\n  Test: Gal={len(gallery_samples)} Prb={len(probe_samples)}")

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

        # ── Evaluation (before FedAvg) ──
        if rnd % cfg["eval_every"] == 0 or rnd == cfg["n_rounds"]:
            print(f"\n  ── Eval round {rnd} ──")
            global_model.eval()
            for m in local_models:
                m.eval()

            # Per-client: local model on full test set
            print(f"    {'Client':>8s} │ {'Local R1':>9s} {'Local EER':>10s} │ "
                  f"{'MoE R1':>8s} {'MoE EER':>9s}")
            print(f"    {'─'*52}")

            client_local_results = []
            for ci in range(n_clients):
                spec = client_data[ci]["spectrum"]
                rl = evaluate_single_model(
                    local_models[ci], gallery_loader, probe_loader, device)
                client_local_results.append(rl)

            # MoE: per-sample routing on full test set
            gal_feats_moe, gal_labels_moe = extract_embeddings_routed(
                local_models, global_model, gallery_loader,
                domain_predictor, cfg, device, mode="moe")
            prb_feats_moe, prb_labels_moe = extract_embeddings_routed(
                local_models, global_model, probe_loader,
                domain_predictor, cfg, device, mode="moe")
            moe_r1, moe_eer = compute_eer_r1(
                prb_feats_moe, prb_labels_moe,
                gal_feats_moe, gal_labels_moe)

            # Per-client MoE: filter test set by domain, show MoE on that subset
            client_moe_results = []
            for ci in range(n_clients):
                # Filter gallery/probe to samples from this domain
                gal_mask = torch.tensor(
                    [s[2] == ci for s in gallery_samples], dtype=torch.bool)
                prb_mask = torch.tensor(
                    [s[2] == ci for s in probe_samples], dtype=torch.bool)

                if gal_mask.any() and prb_mask.any():
                    gf = gal_feats_moe[gal_mask]
                    gl = gal_labels_moe[gal_mask]
                    pf = prb_feats_moe[prb_mask]
                    pl = prb_labels_moe[prb_mask]
                    mr1, meer = compute_eer_r1(pf, pl, gf, gl)
                else:
                    mr1, meer = 0.0, 50.0
                client_moe_results.append({"rank1": mr1, "eer": meer})

            # Print per-client
            for ci in range(n_clients):
                spec = client_data[ci]["spectrum"]
                rl = client_local_results[ci]
                rm = client_moe_results[ci]
                print(f"    {spec:>8s} │ {rl['rank1']:>8.2f}% {rl['eer']:>9.2f}% │ "
                      f"{rm['rank1']:>7.2f}% {rm['eer']:>8.2f}%")

            # Averages
            avg_lr1 = np.mean([r["rank1"] for r in client_local_results])
            avg_leer = np.mean([r["eer"] for r in client_local_results])

            # Global model on full test set
            rg = evaluate_single_model(
                global_model, gallery_loader, probe_loader, device)

            print(f"    {'─'*52}")
            print(f"    {'Avg Local':>8s} │ {avg_lr1:>8.2f}% {avg_leer:>9.2f}% │")
            print(f"    {'MoE':>8s} │ {'':>20s} │ {moe_r1:>7.2f}% {moe_eer:>8.2f}%")
            print(f"    {'Global':>8s} │ {rg['rank1']:>8.2f}% {rg['eer']:>9.2f}% │")

            history.append({
                "round": rnd, "loss": avg_loss,
                "global": rg,
                "avg_local": {"rank1": avg_lr1, "eer": avg_leer},
                "moe": {"rank1": moe_r1, "eer": moe_eer},
                "per_client_local": client_local_results,
                "per_client_moe": client_moe_results,
            })
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
    print(f"  COMPLETE — {protocol}, dp_mode={dp_mode}")
    print(f"{'='*80}")

    print(f"\n  {'Rnd':>5} │ {'Global':>16s} │ {'Avg Local':>16s} │ "
          f"{'MoE':>16s}")
    print(f"  {'':>5} │ {'R1':>7s} {'EER':>7s} │ "
          f"{'R1':>7s} {'EER':>7s} │ {'R1':>7s} {'EER':>7s}")
    print(f"  {'─'*58}")

    for h in history:
        rg = h["global"]; rl = h["avg_local"]; rm = h["moe"]
        print(f"  {h['round']:>5d} │ "
              f"{rg['rank1']:>6.1f}% {rg['eer']:>6.1f}% │ "
              f"{rl['rank1']:>6.1f}% {rl['eer']:>6.1f}% │ "
              f"{rm['rank1']:>6.1f}% {rm['eer']:>6.1f}%")

    if history:
        for mode, key in [("Global", "global"), ("Avg Local", "avg_local"),
                          ("MoE", "moe")]:
            best = max(history, key=lambda h: h[key]["rank1"])
            print(f"\n  Best {mode:>9s}: Rnd {best['round']}  "
                  f"R1={best[key]['rank1']:.2f}%  EER={best[key]['eer']:.2f}%")

    save_path = os.path.join(results_dir,
                              f"results_{protocol}_{dp_mode}.json")
    with open(save_path, "w") as f:
        json.dump({"config": cfg, "dp_accuracy": dp_acc,
                    "history": history}, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
