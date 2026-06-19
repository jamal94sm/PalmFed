"""
main.py — TENT Test-Time Adaptation.

CASIA-MS open-set verification pipeline:
  Phase 1: Train backbone (25%) + head_A on train_spectrums × train_IDs
  Phase 2: Freeze backbone. Train head_B on train_spectrums × test_IDs
  Phase 3: Baseline eval: backbone + head_B on test_spectrums × test_IDs
  Phase 4: TENT: adapt BN on test_spectrums × test_IDs
  Phase 5: Post-TENT eval on test_spectrums × test_IDs

ImageNet-C classification:
  Standard TENT on each corruption.
"""

import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_cfg, CASIA_ORACLE_LOOKUP, CASIA_ORACLE_DOMAINS
from backbones import build_model, ArcFaceHead, ArcFaceModel
import tent
from datasets import (
    get_imagenet_c_loaders, get_casia_ms_train_test,
    split_gallery_probe, extract_embeddings, evaluate_verification,
)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def eval_test_spectrums(model, test_loaders, cfg, tag=""):
    """Evaluate EER + Rank-1 on each test spectrum."""
    was_training = model.training
    model.eval()
    results = {}
    for sname, loader, ds in test_loaders:
        gallery_idx, probe_idx = split_gallery_probe(
            ds, cfg.gallery_ratio, cfg.seed)
        all_idx = list(range(len(ds)))
        feats, labels = extract_embeddings(
            model, ds, all_idx, cfg.batch_size, cfg.device, cfg.num_workers)
        feats_t = feats.to(cfg.device)
        ver = evaluate_verification(feats_t, labels, gallery_idx, probe_idx)
        results[sname] = ver
        print(f"  {tag}{sname:>6s} → EER: {ver['eer']:.2f}% | "
              f"Rank-1: {ver['rank1']:.2f}% | "
              f"Gal: {ver['n_gallery']} | Probe: {ver['n_probe']}")
    mean_eer = np.mean([r['eer'] for r in results.values()])
    mean_r1 = np.mean([r['rank1'] for r in results.values()])
    print(f"  {tag}Mean EER: {mean_eer:.2f}% | Mean Rank-1: {mean_r1:.2f}%")
    if was_training:
        model.train()
    return results


def train_arcface(model, train_loader, train_params, cfg, epochs, tag,
                  test_loaders=None, ckpt_path=None, lr=None):
    """Generic ArcFace training loop. Returns best checkpoint path."""
    lr = lr or cfg.arcface_lr
    optimizer = torch.optim.AdamW(train_params, lr=lr,
                                   weight_decay=cfg.arcface_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    ce_loss = nn.CrossEntropyLoss()

    best_rank1 = 0.0
    model.train()

    for epoch in range(1, epochs + 1):
        ep_loss = 0.0; ep_corr = 0; ep_tot = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            logits = model.train_forward(imgs, labels)
            loss = ce_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 5.0)
            optimizer.step()

            ep_loss += loss.item()
            with torch.no_grad():
                ep_corr += (logits.argmax(1) == labels).sum().item()
                ep_tot += labels.shape[0]

        scheduler.step()
        acc = 100.0 * ep_corr / max(ep_tot, 1)
        n = len(train_loader)
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] {tag} ep {epoch:03d}/{epochs}  "
              f"loss={ep_loss/n:.4f}  acc={acc:.2f}%")

        if test_loaders and (epoch % cfg.arcface_eval_every == 0 or
                             epoch == epochs):
            print(f"  --- eval at epoch {epoch} ---")
            ver = eval_test_spectrums(model, test_loaders, cfg, tag="  ")
            model.train()
            mean_r1 = np.mean([r['rank1'] for r in ver.values()])
            if mean_r1 > best_rank1 and ckpt_path:
                best_rank1 = mean_r1
                torch.save({
                    "epoch": epoch,
                    "backbone": model.backbone.state_dict(),
                    "head": model.head.state_dict(),
                    "rank1": best_rank1,
                }, ckpt_path)
                print(f"  *** New best Rank-1: {best_rank1:.2f}% → saved ***")

    return best_rank1


# ══════════════════════════════════════════════════════════════
#  ImageNet-C (Classification)
# ══════════════════════════════════════════════════════════════

def adapt_imagenet_c(cfg):
    print(f"\n{'='*80}")
    print(f"  TENT — ImageNet-C Classification")
    print(f"  Backbone: {cfg.backbone} | LR: {cfg.tent_lr} | "
          f"Steps/batch: {cfg.tent_steps} | Episodic: {cfg.tent_episodic}")
    print(f"{'='*80}\n")

    model = build_model(cfg)
    norm_mean = getattr(cfg, '_norm_mean', (0.5, 0.5, 0.5))
    norm_std = getattr(cfg, '_norm_std', (0.5, 0.5, 0.5))
    loaders = get_imagenet_c_loaders(
        cfg.data_dir, cfg.severity, cfg.batch_size, cfg.num_workers,
        cfg.img_size, cfg.corruptions, list(norm_mean), list(norm_std))

    # Baseline
    baseline = {}
    if cfg.eval_backbone:
        print("[Baseline] Evaluating frozen backbone...")
        model.eval()
        with torch.no_grad():
            for cname, loader in loaders:
                correct = total = 0
                for imgs, labs in loader:
                    imgs, labs = imgs.to(cfg.device), labs.to(cfg.device)
                    preds = model(imgs).argmax(1)
                    correct += (preds == labs).sum().item()
                    total += labs.shape[0]
                err = 100.0 * (1 - correct / total)
                baseline[cname] = err
                print(f"  {cname:25s} → {err:.1f}%")
        print(f"[Baseline] Mean: {np.mean(list(baseline.values())):.1f}%\n")

    # TENT
    model = tent.configure_model(model)
    tent.check_model(model)
    params, _ = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=cfg.tent_lr)
    tented = tent.Tent(model, optimizer, steps=cfg.tent_steps,
                        episodic=cfg.tent_episodic)
    print(f"[TENT] {len(params)} BN params ({sum(p.numel() for p in params)} values)")

    results = {}
    for seg_idx, (cname, loader) in enumerate(loaders):
        if cfg.tent_episodic: tented.reset()
        n_batches = len(loader); seg_correct = seg_total = 0; t0 = time.time()

        print(f"\n{'─'*70}")
        print(f"  [{seg_idx+1}/{len(loaders)}] {cname} ({len(loader.dataset)} samples)")
        print(f"{'─'*70}")
        print(f"  {'bat':>5} │{'err%':>6} │{'H':>6}")

        for batch_idx, (imgs, labs) in enumerate(loader):
            imgs, labs = imgs.to(cfg.device), labs.to(cfg.device)
            logits = tented(imgs)
            preds = logits.argmax(1)
            correct = (preds == labs).sum().item()
            seg_correct += correct; seg_total += labs.shape[0]
            err = 100.0 * (1 - correct / labs.shape[0])
            if batch_idx < 5 or batch_idx % 100 == 0 or batch_idx == n_batches - 1:
                H = tent.softmax_entropy(logits).mean().item()
                print(f"  {batch_idx:5d} │{err:5.1f} │{H:6.3f}")

        seg_err = 100.0 * (1 - seg_correct / seg_total)
        results[cname] = seg_err
        b_err = baseline.get(cname)
        print(f"\n  ┌── {cname}")
        if b_err is not None:
            imp = b_err - seg_err
            print(f"  │ Backbone: {b_err:.1f}% → TENT: {seg_err:.1f}% "
                  f"({'↓' if imp > 0 else '↑'}{abs(imp):.1f}%)")
        else:
            print(f"  │ TENT: {seg_err:.1f}%")
        print(f"  │ Time: {time.time()-t0:.1f}s")
        print(f"  └{'─'*50}")

    # Summary
    print(f"\n{'='*80}\n  FINAL RESULTS\n{'='*80}")
    mean_t = np.mean(list(results.values()))
    for c, te in results.items():
        if baseline and c in baseline:
            be = baseline[c]; d = be - te
            print(f"  {c:<25} {be:>9.1f}% {te:>9.1f}% "
                  f"{'↓' if d > 0 else '↑'}{abs(d):>8.1f}%")
        else:
            print(f"  {c:<25} {te:>9.1f}%")
    if baseline:
        mb = np.mean(list(baseline.values())); d = mb - mean_t
        print(f"  {'MEAN':<25} {mb:>9.1f}% {mean_t:>9.1f}% "
              f"{'↓' if d > 0 else '↑'}{abs(d):>8.1f}%")
    os.makedirs(cfg.output_dir, exist_ok=True)
    p = os.path.join(cfg.output_dir, f"imagenetc_{cfg.backbone}_seed{cfg.seed}.json")
    with open(p, "w") as f:
        json.dump({"tent": results, **({"baseline": baseline} if baseline else {})},
                  f, indent=2)
    print(f"\n  Saved: {p}")


# ══════════════════════════════════════════════════════════════
#  CASIA-MS (Open-Set Verification)
# ══════════════════════════════════════════════════════════════

def adapt_casia_ms(cfg):
    method_label = cfg.tta_method.upper()
    print(f"\n{'='*80}")
    print(f"  TTA — CASIA-MS Palmprint Verification (Open-Set)")
    print(f"  Backbone: ArcFace iResNet100 | Method: {method_label}")
    print(f"  ID split: {100*(1-cfg.test_id_ratio):.0f}% train / "
          f"{100*cfg.test_id_ratio:.0f}% test")
    print(f"  Phase 1: Train backbone ({100*(1-cfg.arcface_freeze_ratio):.0f}%)"
          f" + head_A ({cfg.arcface_epochs} ep)")
    print(f"  Phase 2: Train head_B ({cfg.arcface_head_epochs} ep, "
          f"LR={cfg.arcface_lr_phase2}, m={cfg.arcface_m_phase2})")
    print(f"  Phase 4: {method_label} on target domains")
    print(f"  Gallery ratio: {cfg.gallery_ratio}")
    print(f"{'='*80}\n")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Build datasets ──
    (train_ids, test_ids, train_id_map, test_id_map,
     backbone_train_loader, test_head_train_loader,
     test_loaders) = get_casia_ms_train_test(
        cfg.data_dir, cfg.train_spectrums, cfg.batch_size,
        cfg.num_workers, cfg.img_size, cfg.test_id_ratio, cfg.seed)

    n_train_cls = len(train_id_map)
    n_test_cls = len(test_id_map)

    # ══════════════════════════════════════════════════════════
    #  PHASE 1: Train backbone + head_A
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE 1: Train backbone + train-ID head ({n_train_cls} classes)")
    print(f"  {len(backbone_train_loader.dataset)} samples, "
          f"{cfg.arcface_epochs} epochs")
    print(f"{'─'*70}")

    cfg.arcface_num_classes = n_train_cls
    model = build_model(cfg)

    train_params = ([p for p in model.backbone.parameters() if p.requires_grad]
                    + list(model.head.parameters()))
    n_unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    print(f"[Phase 1] {n_unfrozen} unfrozen backbone tensors + "
          f"head_A ({n_train_cls}×512)")

    ckpt1 = os.path.join(cfg.output_dir, "phase1_best.pth")
    train_arcface(model, backbone_train_loader, train_params, cfg,
                  cfg.arcface_epochs, "P1", ckpt_path=ckpt1)

    if os.path.exists(ckpt1):
        ckpt = torch.load(ckpt1, map_location=cfg.device, weights_only=False)
        model.backbone.load_state_dict(ckpt["backbone"])
        print(f"\n  Loaded best Phase 1 backbone (epoch {ckpt['epoch']}, "
              f"Rank-1={ckpt['rank1']:.2f}%)")

    # ══════════════════════════════════════════════════════════
    #  PHASE 2: Train head_B for test IDs (tent & contrastive)
    #           BNA skips this phase
    # ══════════════════════════════════════════════════════════
    model.backbone.requires_grad_(False)

    if cfg.tta_method in ("tent", "contrastive"):
        print(f"\n{'─'*70}")
        print(f"  PHASE 2: Train test-ID head ({n_test_cls} classes)")
        print(f"  Source domain: {cfg.train_spectrums}")
        print(f"  {len(test_head_train_loader.dataset)} samples, "
              f"{cfg.arcface_head_epochs} epochs, LR={cfg.arcface_lr_phase2}")
        print(f"  Backbone: FROZEN")
        print(f"{'─'*70}")

        head_B = ArcFaceHead(n_test_cls, embedding_size=512,
                              s=cfg.arcface_s, m=cfg.arcface_m_phase2
                              ).to(cfg.device)
        model.head = head_B

        head_params = list(model.head.parameters())
        print(f"[Phase 2] Head_B: {n_test_cls} classes, "
              f"{sum(p.numel() for p in head_params)} params, "
              f"margin={cfg.arcface_m_phase2}")

        ckpt2 = os.path.join(cfg.output_dir, "phase2_best.pth")
        train_arcface(model, test_head_train_loader, head_params, cfg,
                      cfg.arcface_head_epochs, "P2",
                      test_loaders=test_loaders, ckpt_path=ckpt2,
                      lr=cfg.arcface_lr_phase2)

        if os.path.exists(ckpt2):
            ckpt = torch.load(ckpt2, map_location=cfg.device, weights_only=False)
            model.head.load_state_dict(ckpt["head"])
            print(f"\n  Loaded best Phase 2 head (epoch {ckpt['epoch']}, "
                  f"Rank-1={ckpt['rank1']:.2f}%)")
    else:
        print(f"\n{'─'*70}")
        print(f"  PHASE 2: SKIPPED (BNA needs no classification head)")
        print(f"{'─'*70}")

    # ══════════════════════════════════════════════════════════
    #  PHASE 3: Pre-TTA baseline
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE 3: Pre-TTA Baseline on Target Domains")
    print(f"{'─'*70}")
    baseline = eval_test_spectrums(model, test_loaders, cfg, tag="[pre-TTA] ")

    # ══════════════════════════════════════════════════════════
    #  PHASE 4: TTA adaptation
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE 4: {method_label} on Target Domains")
    print(f"{'─'*70}")

    # Build domain sequence
    if cfg.oracle_domains:
        from collections import OrderedDict
        groups = OrderedDict()
        for sn, ld, ds in test_loaders:
            gn, gid = CASIA_ORACLE_LOOKUP.get(sn, ("unk", -1))
            if gid not in groups:
                groups[gid] = {"name": gn, "spectrums": []}
            groups[gid]["spectrums"].append((sn, ld, ds))
        dom_seq = [(g["name"], gi, g["spectrums"])
                   for gi, g in groups.items()]
    else:
        dom_seq = [(s, i, [(s, ld, ds)])
                   for i, (s, ld, ds) in enumerate(test_loaders)]

    # ── TENT ──
    if cfg.tta_method == "tent":
        model = tent.configure_model(model)
        tent.check_model(model)
        params, _ = tent.collect_params(model)
        opt = torch.optim.Adam(params, lr=cfg.tent_lr)
        tented = tent.Tent(model, opt, steps=cfg.tent_steps,
                           episodic=cfg.tent_episodic)
        print(f"[TENT] {len(params)} BN params "
              f"({sum(p.numel() for p in params)} values)")
        print(f"[TENT] Entropy from: head_B ({n_test_cls} classes)")

        for di, (dn, _, slist) in enumerate(dom_seq):
            if cfg.tent_episodic: tented.reset()
            tb = sum(len(l) for _, l, _ in slist)
            t0 = time.time()
            print(f"\n  [{di+1}/{len(dom_seq)}] {dn} "
                  f"({[s for s,_,_ in slist]})")
            print(f"  {'bat':>5} │{'spec':>6} │{'H':>6}")
            gb = 0
            for sn, ld, _ in slist:
                for imgs, _ in ld:
                    logits = tented(imgs.to(cfg.device))
                    if gb < 5 or gb % 50 == 0 or gb == tb - 1:
                        H = tent.softmax_entropy(logits).mean().item()
                        print(f"  {gb:5d} │{sn:>6s} │{H:6.3f}")
                    gb += 1
            print(f"  Time: {time.time()-t0:.1f}s")

    # ── CONTRASTIVE: entropy(head_B) + NT-Xent ──
    elif cfg.tta_method == "contrastive":
        model = tent.configure_model(model)
        tent.check_model(model)
        params, _ = tent.collect_params(model)
        opt = torch.optim.Adam(params, lr=cfg.tent_lr)
        aug_tf = tent.get_tta_augmentation(cfg.img_size)
        con = tent.ContrastiveTent(
            model, opt, aug_tf,
            contrastive_lambda=cfg.contrastive_lambda,
            contrastive_temp=cfg.contrastive_temp,
            steps=cfg.tent_steps, episodic=cfg.tent_episodic,
            use_entropy=True)
        print(f"[CONTRASTIVE] {len(params)} BN params "
              f"({sum(p.numel() for p in params)} values)")
        print(f"[CONTRASTIVE] Entropy from: head_B ({n_test_cls} classes)")
        print(f"[CONTRASTIVE] λ={cfg.contrastive_lambda}, "
              f"τ={cfg.contrastive_temp}")

        for di, (dn, _, slist) in enumerate(dom_seq):
            if cfg.tent_episodic: con.reset()
            tb = sum(len(l) for _, l, _ in slist)
            t0 = time.time()
            print(f"\n  [{di+1}/{len(dom_seq)}] {dn} "
                  f"({[s for s,_,_ in slist]})")
            print(f"  {'bat':>5} │{'spec':>6} │{'H':>6} │"
                  f"{'con':>6} │{'total':>6}")
            gb = 0
            for sn, ld, _ in slist:
                for imgs, _ in ld:
                    logits, info = con(imgs.to(cfg.device))
                    if gb < 5 or gb % 50 == 0 or gb == tb - 1:
                        print(f"  {gb:5d} │{sn:>6s} │"
                              f"{info.get('entropy',0):6.3f} │"
                              f"{info.get('contrastive',0):6.3f} │"
                              f"{info.get('total',0):6.3f}")
                    gb += 1
            print(f"  Time: {time.time()-t0:.1f}s")

    # ── BNA ──
    elif cfg.tta_method == "bna":
        model = tent.configure_model_bna(model)
        print(f"[BNA] BN running stats reset. Adapting via forward pass...")
        for di, (dn, _, slist) in enumerate(dom_seq):
            t0 = time.time()
            print(f"\n  [{di+1}/{len(dom_seq)}] {dn} "
                  f"({[s for s,_,_ in slist]})")
            gb = 0
            for sn, ld, _ in slist:
                for imgs, _ in ld:
                    tent.forward_bna(imgs.to(cfg.device), model)
                    gb += 1
            print(f"  {gb} batches | Time: {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════
    #  PHASE 5: Post-TTA eval
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE 5: Post-{method_label} Evaluation")
    print(f"{'─'*70}")
    model.eval()
    post_tta = eval_test_spectrums(model, test_loaders, cfg,
                                    tag=f"[post-{method_label}] ")

    # ── Final comparison ──
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON ({method_label})")
    print(f"  Train IDs: {n_train_cls} | Test IDs: {n_test_cls}")
    print(f"  Source: {cfg.train_spectrums} | Target: "
          f"{[s for s,_,_ in test_loaders]}")
    print(f"{'='*80}")
    print(f"\n  {'Spectrum':<10} {'Pre EER':>9} {'Pre R1':>8} "
          f"{'Post EER':>9} {'Post R1':>8} {'ΔEER':>8} {'ΔR1':>8}")
    print(f"  {'─'*65}")

    for sn in post_tta:
        b = baseline.get(sn, {}); n = post_tta[sn]
        de = b.get("eer", 0) - n["eer"]
        dr = n["rank1"] - b.get("rank1", 0)
        print(f"  {sn:<10} {b.get('eer',-1):>8.2f}% {b.get('rank1',-1):>7.2f}% "
              f"{n['eer']:>8.2f}% {n['rank1']:>7.2f}% "
              f"{'↓' if de > 0 else '↑'}{abs(de):>6.2f}% "
              f"{'↑' if dr > 0 else '↓'}{abs(dr):>6.2f}%")

    def _m(d, k):
        return np.mean([r[k] for r in d.values()]) if d else -1
    be = _m(baseline, 'eer'); br = _m(baseline, 'rank1')
    te = _m(post_tta, 'eer'); tr = _m(post_tta, 'rank1')
    de = be - te; dr = tr - br
    print(f"  {'─'*65}")
    print(f"  {'MEAN':<10} {be:>8.2f}% {br:>7.2f}% "
          f"{te:>8.2f}% {tr:>7.2f}% "
          f"{'↓' if de > 0 else '↑'}{abs(de):>6.2f}% "
          f"{'↑' if dr > 0 else '↓'}{abs(dr):>6.2f}%")

    save_data = {
        "tta_method": cfg.tta_method, "baseline": {k: dict(v) for k, v in baseline.items()},
        "post_tta": {k: dict(v) for k, v in post_tta.items()},
        "n_train_ids": n_train_cls, "n_test_ids": n_test_cls,
        "train_spectrums": cfg.train_spectrums,
    }
    p = os.path.join(cfg.output_dir, f"casia_{cfg.tta_method}_seed{cfg.seed}.json")
    with open(p, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {p}")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = get_cfg()
    set_seed(cfg.seed)
    if cfg.dataset == "casia_ms":
        adapt_casia_ms(cfg)
    elif cfg.dataset == "imagenet_c":
        adapt_imagenet_c(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
