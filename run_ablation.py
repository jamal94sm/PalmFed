# ==============================================================
#  run_ablation.py
#  Hyperparameter ablation for lambda_style, lambda_supcon, lambda_grl.
#
#  Strategy: one-at-a-time (OAT) sweep — vary one λ while holding
#  the others at 0. This isolates each term's individual contribution
#  before combining the best values.
#
#  Grid:
#    lambda_style  : 0, 0.05, 0.1, 0.5, 1.0   (5 values)
#    lambda_supcon : 0, 0.1,  0.2, 0.5         (4 values)
#    lambda_grl    : 0, 0.05, 0.1, 0.2         (4 values)
#
#  Phase 1 — single-term sweep (13 runs including shared baseline):
#    baseline:        style=0, supcon=0, grl=0
#    style sweep:     style∈{0.05,0.1,0.5,1.0}, supcon=0, grl=0
#    supcon sweep:    style=0, supcon∈{0.1,0.2,0.5}, grl=0
#    grl sweep:       style=0, supcon=0, grl∈{0.05,0.1,0.2}
#
#  Phase 2 — best-pair combinations (optional, run after Phase 1):
#    Take best λ_style and best λ_supcon, combine.
#    Take best λ_style and best λ_grl, combine.
#    Take all three best values together.
#
#  Output: ablation_results.csv + printed table.
#
#  Usage:
#    python run_ablation.py               # Phase 1 only
#    python run_ablation.py --phase2      # Phase 1 + Phase 2
#    python run_ablation.py --n_rounds 50 # fewer rounds for quick check
#    python run_ablation.py --resume      # skip completed runs
# ==============================================================

import os
import copy
import csv
import argparse
import time
import random
import pickle
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from configs  import CONFIG
from models   import build_model
from datasets import (PalmDataset, AugmentedDataset,
                      FFTAugmentedDataset,
                      get_federated_splits)
from utils    import (extract_style_template, evaluate_model,
                      train_compnet_epoch, CenterLoss)
from PIL import Image


# ══════════════════════════════════════════════════════════════
#  ABLATION GRID
# ══════════════════════════════════════════════════════════════

STYLE_VALUES  = [0.0, 0.05, 0.1, 0.5, 1.0]
SUPCON_VALUES = [0.0, 0.1,  0.2, 0.5]
GRL_VALUES    = [0.0, 0.05, 0.1, 0.2]


def build_phase1_grid():
    """
    One-at-a-time sweep. Baseline (0,0,0) appears once.
    Returns list of (lambda_style, lambda_supcon, lambda_grl) tuples.
    """
    runs = []
    runs.append((0.0, 0.0, 0.0))                              # baseline
    for v in STYLE_VALUES:
        if v > 0:
            runs.append((v,   0.0, 0.0))                      # style sweep
    for v in SUPCON_VALUES:
        if v > 0:
            runs.append((0.0, v,   0.0))                      # supcon sweep
    for v in GRL_VALUES:
        if v > 0:
            runs.append((0.0, 0.0, v  ))                      # grl sweep
    return runs


def build_phase2_grid(best_style, best_supcon, best_grl):
    """
    Best-pair and full combinations from Phase 1 winners.
    Only adds non-zero combinations not already run in Phase 1.
    """
    runs = []
    if best_style > 0 and best_supcon > 0:
        runs.append((best_style,  best_supcon, 0.0))
    if best_style > 0 and best_grl > 0:
        runs.append((best_style,  0.0,         best_grl))
    if best_supcon > 0 and best_grl > 0:
        runs.append((0.0,         best_supcon, best_grl))
    if best_style > 0 and best_supcon > 0 and best_grl > 0:
        runs.append((best_style,  best_supcon, best_grl))
    return runs


# ══════════════════════════════════════════════════════════════
#  SINGLE RUN
# ══════════════════════════════════════════════════════════════

def run_one(cfg, lambda_style, lambda_supcon, lambda_grl, n_rounds, seed):
    """
    Run a full FL experiment with the given λ values.
    Returns (avg_eer, avg_rank1) averaged over cfg["avg_last_rounds"].
    """
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_k = cfg["dataset"].strip().lower()
    model_k   = cfg["model"].strip().lower()

    splits_path = cfg["splits_path"].format(dataset=dataset_k)
    init_path   = cfg["init_weights_path"].format(
                      dataset=dataset_k, model=model_k)

    # ── splits ──────────────────────────────────────────────
    if os.path.exists(splits_path):
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        client_data, gallery_samples, probe_samples, _, domain_names = splits
    else:
        splits = get_federated_splits(cfg, seed=seed)
        os.makedirs(os.path.dirname(splits_path), exist_ok=True)
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)
        client_data, gallery_samples, probe_samples, _, domain_names = splits

    num_classes = client_data[0]["num_classes"]
    n_clients   = len(client_data)

    # ── server + clients ────────────────────────────────────
    server_model = build_model(cfg, num_classes).to(device)

    gallery_loader = DataLoader(
        PalmDataset(gallery_samples, cfg["img_side"]),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"])
    probe_loader   = DataLoader(
        PalmDataset(probe_samples, cfg["img_side"]),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"])

    # ── init weights ────────────────────────────────────────
    if os.path.exists(init_path):
        init_state = torch.load(init_path, map_location=device)
        server_model.load_state_dict(init_state, strict=False)
    else:
        torch.save(server_model.state_dict(), init_path)

    backbone_init = {k: v for k, v in server_model.state_dict().items()
                     if not k.startswith("arc.")}

    # local models (one per client) — start from shared backbone init
    local_models = []
    for cd in client_data:
        m = build_model(cfg, num_classes).to(device)
        local_state = m.state_dict()
        for k, v in backbone_init.items():
            if k in local_state and local_state[k].shape == v.shape:
                local_state[k] = v.clone()
        m.load_state_dict(local_state)
        local_models.append(m)

    # ── style bank ──────────────────────────────────────────
    style_bank = {}
    if cfg.get("use_fft_aug", False):
        for i, cd in enumerate(client_data):
            templates = []
            for path, _ in cd["train_samples"]:
                img    = Image.open(path).convert("L").resize(
                    (cfg["img_side"], cfg["img_side"]), Image.BILINEAR)
                img_np = np.array(img, dtype=np.float32) / 255.0
                templates.append(extract_style_template(img_np))
            style_bank[i] = templates

    # determine active grl flag
    use_grl_active = (lambda_grl > 0 and cfg.get("use_grl", False))

    # ── FL rounds ───────────────────────────────────────────
    recent_history = []

    for rnd in range(1, n_rounds + 1):
        client_weights = []

        for i, (client_model, cd) in enumerate(zip(local_models, client_data)):
            # set global backbone weights
            global_w = {k: v.cpu().clone()
                        for k, v in server_model.state_dict().items()
                        if not k.startswith("arc.")}
            local_state = client_model.state_dict()
            for k, v in global_w.items():
                if k in local_state and local_state[k].shape == v.shape:
                    local_state[k] = v.clone()
            client_model.load_state_dict(local_state)

            # dataset
            if style_bank and cfg["M"] > 1:
                dataset = FFTAugmentedDataset(
                    samples    = cd["train_samples"],
                    style_bank = style_bank,
                    client_id  = i,
                    M          = cfg["M"],
                    beta       = cfg["fft_beta"],
                    img_side   = cfg["img_side"],
                    grayscale  = True,
                )
            else:
                dataset = AugmentedDataset(
                    cd["train_samples"], cfg["img_side"],
                    grayscale=True, client_id=i)

            loader = DataLoader(
                dataset,
                batch_size     = cfg["batch_size"],
                shuffle        = True,
                num_workers    = cfg["num_workers"],
                pin_memory     = True,
                worker_init_fn = lambda wid, s=seed+rnd*1000+i: (
                    np.random.seed(s+wid), random.seed(s+wid)))

            optimizer = optim.Adam(client_model.parameters(), lr=cfg["lr"])
            criterion = nn.CrossEntropyLoss()

            train_compnet_epoch(
                client_model, loader, criterion, optimizer, device,
                lambda_style    = lambda_style,
                lambda_supcon   = lambda_supcon if cfg.get("use_supcon", False)
                                  or lambda_supcon > 0 else 0.0,
                temperature     = cfg.get("temperature", 0.07),
                lambda_grl      = lambda_grl if use_grl_active else 0.0,
                lambda_load_balance = 0.0,
            )

            client_weights.append(
                {k: v.cpu().clone()
                 for k, v in client_model.state_dict().items()
                 if not k.startswith("arc.")})

        # FedAvg
        avg_dict = {}
        for key in client_weights[0].keys():
            stacked      = torch.stack([cw[key].float()
                                        for cw in client_weights], dim=0)
            avg_dict[key] = stacked.mean(dim=0)
        global_state = server_model.state_dict()
        global_state.update(avg_dict)
        server_model.load_state_dict(global_state)

        # evaluate
        eer, r1 = evaluate_model(server_model, gallery_loader,
                                  probe_loader, device)
        recent_history.append((eer, r1))
        if len(recent_history) > cfg["avg_last_rounds"]:
            recent_history.pop(0)

    n_avg    = len(recent_history)
    avg_eer  = sum(e for e, _ in recent_history) / n_avg * 100
    avg_r1   = sum(r for _, r in recent_history) / n_avg
    return avg_eer, avg_r1


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase2",    action="store_true",
                   help="also run best-pair combinations after Phase 1")
    p.add_argument("--n_rounds",  type=int, default=None,
                   help="override n_rounds from configs (e.g. 50 for quick runs)")
    p.add_argument("--resume",    action="store_true",
                   help="skip combinations already in ablation_results.csv")
    p.add_argument("--out",       type=str, default="ablation_results.csv",
                   help="output CSV filename")
    return p.parse_args()


def load_done(csv_path):
    """Return set of (style, supcon, grl) tuples already in the CSV."""
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((float(row["lambda_style"]),
                          float(row["lambda_supcon"]),
                          float(row["lambda_grl"])))
    return done


def print_table(rows):
    """Pretty-print results table to console."""
    print(f"\n{'='*70}")
    print(f"  {'λ_style':>10}  {'λ_supcon':>10}  {'λ_grl':>8}  "
          f"{'EER (%)':>10}  {'Rank-1 (%)':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*12}")
    for r in rows:
        marker = " ◀ best" if r.get("best") else ""
        print(f"  {r['lambda_style']:>10.3f}  {r['lambda_supcon']:>10.3f}  "
              f"{r['lambda_grl']:>8.3f}  {r['eer']:>10.4f}  "
              f"{r['rank1']:>12.2f}{marker}")
    print(f"{'='*70}\n")


def main():
    args   = parse_args()
    cfg    = copy.deepcopy(CONFIG)

    # override configs for ablation
    cfg["use_grl"]    = True    # GRL branch must be in model for lambda_grl
    cfg["use_supcon"] = True    # SupCon flag must be on for lambda_supcon
    cfg["use_moe"]    = False
    cfg["use_center_loss"] = False

    if args.n_rounds is not None:
        cfg["n_rounds"] = args.n_rounds

    n_rounds = cfg["n_rounds"]
    seed     = cfg["random_seed"]
    csv_path = args.out
    done     = load_done(csv_path) if args.resume else set()

    # write CSV header if new file
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "phase", "lambda_style", "lambda_supcon", "lambda_grl",
                "eer_pct", "rank1_pct", "n_rounds"])
            writer.writeheader()

    # ── Phase 1 ─────────────────────────────────────────────
    phase1_grid = build_phase1_grid()
    phase1_rows = []

    print(f"\n{'='*70}")
    print(f"  ABLATION — Phase 1 (one-at-a-time sweep)")
    print(f"  Rounds: {n_rounds}   Dataset: {cfg['dataset'].upper()}")
    print(f"  Runs: {len(phase1_grid)}")
    print(f"{'='*70}")

    for ls, lsc, lgrl in phase1_grid:
        key = (ls, lsc, lgrl)
        if args.resume and key in done:
            print(f"  SKIP  style={ls}  supcon={lsc}  grl={lgrl}  (already done)")
            continue

        tag = f"style={ls}  supcon={lsc}  grl={lgrl}"
        print(f"\n  RUN  {tag}  ({n_rounds} rounds) …")
        t0 = time.time()

        eer, r1 = run_one(cfg, ls, lsc, lgrl, n_rounds, seed)

        elapsed = time.time() - t0
        print(f"  → EER={eer:.4f}%  Rank-1={r1:.2f}%  "
              f"({elapsed/60:.1f} min)")

        row = {"phase": 1, "lambda_style": ls, "lambda_supcon": lsc,
               "lambda_grl": lgrl, "eer_pct": eer, "rank1_pct": r1,
               "n_rounds": n_rounds}
        phase1_rows.append({"lambda_style": ls, "lambda_supcon": lsc,
                             "lambda_grl": lgrl, "eer": eer, "rank1": r1})

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "phase","lambda_style","lambda_supcon","lambda_grl",
                "eer_pct","rank1_pct","n_rounds"]).writerow(row)

    # mark best in Phase 1
    if phase1_rows:
        best_eer = min(phase1_rows, key=lambda r: r["eer"])
        best_eer["best"] = True
        print("\n  Phase 1 results:")
        print_table(phase1_rows)
        print(f"  Best: style={best_eer['lambda_style']}  "
              f"supcon={best_eer['lambda_supcon']}  "
              f"grl={best_eer['lambda_grl']}  "
              f"EER={best_eer['eer']:.4f}%")

    # ── Phase 2 ─────────────────────────────────────────────
    if args.phase2 and phase1_rows:
        # find best individual value for each term
        style_rows  = [r for r in phase1_rows
                       if r["lambda_supcon"] == 0 and r["lambda_grl"] == 0]
        supcon_rows = [r for r in phase1_rows
                       if r["lambda_style"]  == 0 and r["lambda_grl"] == 0]
        grl_rows    = [r for r in phase1_rows
                       if r["lambda_style"]  == 0 and r["lambda_supcon"] == 0]

        best_style  = min(style_rows,  key=lambda r: r["eer"])["lambda_style"]
        best_supcon = min(supcon_rows, key=lambda r: r["eer"])["lambda_supcon"]
        best_grl    = min(grl_rows,    key=lambda r: r["eer"])["lambda_grl"]

        print(f"\n  Best individual values: "
              f"style={best_style}  supcon={best_supcon}  grl={best_grl}")

        phase2_grid = build_phase2_grid(best_style, best_supcon, best_grl)
        phase2_rows = []

        print(f"\n{'='*70}")
        print(f"  ABLATION — Phase 2 (best-pair combinations)")
        print(f"  Runs: {len(phase2_grid)}")
        print(f"{'='*70}")

        for ls, lsc, lgrl in phase2_grid:
            key = (ls, lsc, lgrl)
            if args.resume and key in done:
                print(f"  SKIP  style={ls}  supcon={lsc}  grl={lgrl}")
                continue

            print(f"\n  RUN  style={ls}  supcon={lsc}  grl={lgrl} …")
            t0  = time.time()
            eer, r1 = run_one(cfg, ls, lsc, lgrl, n_rounds, seed)
            elapsed = time.time() - t0
            print(f"  → EER={eer:.4f}%  Rank-1={r1:.2f}%  "
                  f"({elapsed/60:.1f} min)")

            row = {"phase": 2, "lambda_style": ls, "lambda_supcon": lsc,
                   "lambda_grl": lgrl, "eer_pct": eer, "rank1_pct": r1,
                   "n_rounds": n_rounds}
            phase2_rows.append({"lambda_style": ls, "lambda_supcon": lsc,
                                 "lambda_grl": lgrl, "eer": eer, "rank1": r1})

            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=[
                    "phase","lambda_style","lambda_supcon","lambda_grl",
                    "eer_pct","rank1_pct","n_rounds"]).writerow(row)

        if phase2_rows:
            best_p2 = min(phase2_rows, key=lambda r: r["eer"])
            best_p2["best"] = True
            print("\n  Phase 2 results:")
            print_table(phase2_rows)

        # combined table
        all_rows = phase1_rows + phase2_rows
        overall_best = min(all_rows, key=lambda r: r["eer"])
        for r in all_rows:
            r.pop("best", None)
        overall_best["best"] = True
        print("\n  Combined results (Phase 1 + Phase 2):")
        print_table(all_rows)

    print(f"  Results saved to: {csv_path}\n")


if __name__ == "__main__":
    main()
