"""
Frequency Band Domain-Informativeness Analysis
================================================
Identifies which FFT amplitude frequency bands best discriminate
between domains (spectra for CASIA-MS, device/lighting for XJTU).

Two methods (toggle via METHOD):
  "anova"    — F-statistic: between-domain vs within-domain variance
               per frequency bin. No model training needed.
  "logreg"   — Logistic Regression: learns which bins the classifier
               relies on. Captures inter-bin interactions.

Outputs
-------
  band_scores.txt   — per-band F / importance scores
  band_analysis.png — multi-panel figure with highlighted bands
"""

# ==============================================================
#  CONFIG
# ==============================================================
CONFIG = {
    # ── Dataset ────────────────────────────────────────────────
    "dataset"        : "casiams",   # "casiams" | "xjtu"
    "casiams_root"   : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "xjtu_root"      : "/home/pai-ng/Jamal/XJTU-UP",
    "img_side"       : 128,
    "n_ids"          : 50,      # identities to sample per domain (faster)
    "n_imgs_per_id"  : 5,       # images per identity per domain to sample

    # ── Analysis ───────────────────────────────────────────────
    "method"         : "logreg",   # "anova" | "logreg"
    "n_bins"         : 20,        # number of radial frequency bands to analyse
    "logreg_C"       : 0.1,       # regularisation for logistic regression
    "random_seed"    : 42,

    # ── Output ─────────────────────────────────────────────────
    "out_dir"        : "./band_analysis_results",
}
# ==============================================================

import os
import random
import warnings
import numpy as np
from collections import defaultdict
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

XJTU_VARIATIONS = [
    ("iPhone", "Flash"),
    ("iPhone", "Nature"),
    ("huawei", "Flash"),
    ("huawei", "Nature"),
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ══════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_casiams(data_root, n_ids, n_imgs_per_id, img_side, seed):
    """
    Returns {spectrum_label: [amp_array, ...]}
    Each amp_array is the fftshifted amplitude of one image (H, W).
    """
    rng = random.Random(seed)
    # parse: spectrum → identity → [paths]
    data = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4:
            continue
        identity = f"{parts[0]}_{parts[1]}"
        spectrum = parts[2]
        data[spectrum][identity].append(os.path.join(data_root, fname))

    spectra = sorted(data.keys())
    # common identities across all spectra
    common_ids = set(data[spectra[0]].keys())
    for sp in spectra[1:]:
        common_ids &= set(data[sp].keys())
    common_ids = sorted(common_ids)
    selected = sorted(rng.sample(common_ids, min(n_ids, len(common_ids))))

    domain_amps = defaultdict(list)
    for sp in spectra:
        for ident in selected:
            paths = data[sp][ident]
            chosen = rng.sample(paths, min(n_imgs_per_id, len(paths)))
            for p in chosen:
                domain_amps[sp].append(_extract_amp(p, img_side))
        print(f"  [{sp:>8}]  {len(domain_amps[sp])} amplitude arrays")

    return dict(domain_amps)


def load_xjtu(data_root, n_ids, n_imgs_per_id, img_side, seed):
    """
    Returns {'{device}/{condition}': [amp_array, ...]}
    """
    rng = random.Random(seed)
    data = defaultdict(lambda: defaultdict(list))
    for device, condition in XJTU_VARIATIONS:
        var_dir = os.path.join(data_root, device, condition)
        if not os.path.isdir(var_dir):
            print(f"  WARNING: {var_dir} not found")
            continue
        for id_folder in sorted(os.listdir(var_dir)):
            id_dir = os.path.join(var_dir, id_folder)
            if not os.path.isdir(id_dir):
                continue
            parts = id_folder.split("_")
            if len(parts) < 2 or parts[0].upper() not in ("L", "R"):
                continue
            for fname in sorted(os.listdir(id_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                    continue
                data[(device, condition)][id_folder].append(
                    os.path.join(id_dir, fname))

    # common ids across all 4 domains
    domains   = XJTU_VARIATIONS
    common_ids = set(data[domains[0]].keys())
    for d in domains[1:]:
        common_ids &= set(data[d].keys())
    common_ids = sorted(common_ids)
    selected   = sorted(rng.sample(common_ids, min(n_ids, len(common_ids))))

    domain_amps = defaultdict(list)
    for device, condition in domains:
        label = f"{device}/{condition}"
        for ident in selected:
            paths = data[(device, condition)][ident]
            chosen = rng.sample(paths, min(n_imgs_per_id, len(paths)))
            for p in chosen:
                domain_amps[label].append(_extract_amp(p, img_side))
        print(f"  [{label:>15}]  {len(domain_amps[label])} amplitude arrays")

    return dict(domain_amps)


def _extract_amp(path, img_side):
    """Load image, convert to grayscale, return fftshifted amplitude (H, W)."""
    img = Image.open(path).convert("L").resize(
        (img_side, img_side), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    amp    = np.abs(np.fft.fft2(img_np))
    return np.fft.fftshift(amp).astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  RADIAL BINNING HELPERS
# ══════════════════════════════════════════════════════════════

def make_radial_bins(H, W, n_bins):
    """
    Returns (radial_dist, band_edges, bin_masks) where:
      radial_dist : (H, W) array of distance from DC centre
      band_edges  : (n_bins+1,) array of radial bin edges
      bin_masks   : list of n_bins boolean (H, W) arrays
    """
    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    radial_dist  = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)
    max_r        = np.sqrt(cx**2 + cy**2)
    band_edges   = np.linspace(0, max_r, n_bins + 1)
    bin_masks    = [
        (radial_dist >= band_edges[i]) & (radial_dist < band_edges[i+1])
        for i in range(n_bins)
    ]
    return radial_dist, band_edges, bin_masks


def beta_range(band_edges, i, img_side):
    """Convert radial bin edge indices to beta (fraction of min(H,W))."""
    return band_edges[i] / img_side, band_edges[i+1] / img_side


# ══════════════════════════════════════════════════════════════
#  METHOD 1 — ANOVA F-STATISTIC
# ══════════════════════════════════════════════════════════════

def analyse_anova(domain_amps, n_bins, img_side):
    """
    Per-pixel F-statistic across domains, then averaged per radial band.
    Higher F → more domain-discriminative.

    Returns
    -------
    f_map        : (H, W) per-pixel F-statistic
    band_scores  : list of (beta_lo, beta_hi, mean_F) per band
    """
    domains = sorted(domain_amps.keys())
    H = W   = img_side

    print(f"\n  Computing per-pixel F-statistic "
          f"({H*W} bins × {len(domains)} domains) …")

    # stack arrays: {domain: [N_d, H*W]}
    groups_flat = {
        d: np.stack([a.flatten() for a in domain_amps[d]])
        for d in domains
    }

    # vectorised F-statistic across all pixels at once
    # between-group mean, within-group variance
    n_d    = np.array([groups_flat[d].shape[0] for d in domains])  # [D]
    N      = n_d.sum()
    D      = len(domains)

    # grand mean per pixel [H*W]
    grand_mean = np.concatenate(
        [groups_flat[d] for d in domains], axis=0).mean(axis=0)

    ss_between = np.zeros(H * W)
    ss_within  = np.zeros(H * W)
    for i, d in enumerate(domains):
        g       = groups_flat[d]           # [N_d, H*W]
        g_mean  = g.mean(axis=0)           # [H*W]
        ss_between += n_d[i] * (g_mean - grand_mean)**2
        ss_within  += ((g - g_mean)**2).sum(axis=0)

    df_between = D - 1
    df_within  = N - D
    ms_between = ss_between / (df_between + 1e-10)
    ms_within  = ss_within  / (df_within  + 1e-10)
    f_map      = (ms_between / (ms_within + 1e-10)).reshape(H, W)

    # radial binning
    radial_dist, band_edges, bin_masks = make_radial_bins(H, W, n_bins)
    band_scores = []
    for i, mask in enumerate(bin_masks):
        mean_f = f_map[mask].mean() if mask.any() else 0.0
        b_lo, b_hi = beta_range(band_edges, i, img_side)
        band_scores.append((b_lo, b_hi, float(mean_f)))

    return f_map, band_scores


# ══════════════════════════════════════════════════════════════
#  METHOD 2 — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════

def analyse_logreg(domain_amps, n_bins, img_side, C=0.1, seed=42):
    """
    Train L1-regularised logistic regression on flattened amplitude arrays.
    Feature importance = mean absolute coefficient across all classes.

    Returns
    -------
    imp_map      : (H, W) per-pixel importance
    band_scores  : list of (beta_lo, beta_hi, mean_importance) per band
    """
    domains = sorted(domain_amps.keys())
    H = W   = img_side

    X = np.concatenate(
        [np.stack([a.flatten() for a in domain_amps[d]]) for d in domains])
    y = np.concatenate(
        [np.full(len(domain_amps[d]), i) for i, d in enumerate(domains)])

    print(f"\n  Training logistic regression  "
          f"(X={X.shape}, classes={len(domains)}) …")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        C=C, penalty="l1", solver="saga",
        max_iter=2000, random_state=seed)
    clf.fit(X_scaled, y)
    acc = clf.score(X_scaled, y)
    print(f"  Training accuracy: {acc*100:.1f}%")

    # mean absolute coefficient across all one-vs-rest classifiers
    importance = np.abs(clf.coef_).mean(axis=0)    # [H*W]
    imp_map    = importance.reshape(H, W)

    # radial binning
    radial_dist, band_edges, bin_masks = make_radial_bins(H, W, n_bins)
    band_scores = []
    for i, mask in enumerate(bin_masks):
        mean_imp = imp_map[mask].mean() if mask.any() else 0.0
        b_lo, b_hi = beta_range(band_edges, i, img_side)
        band_scores.append((b_lo, b_hi, float(mean_imp)))

    return imp_map, band_scores


# ══════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_results(score_map, band_scores, domain_amps, method,
                 dataset, img_side, out_dir):
    """
    4-panel figure:
      Panel 1 — per-pixel score map (log scale) with band rings overlaid
      Panel 2 — bar chart of per-band scores with top band highlighted
      Panel 3 — mean amplitude per domain (log scale)
      Panel 4 — recommended beta range annotation
    """
    os.makedirs(out_dir, exist_ok=True)
    domains     = sorted(domain_amps.keys())
    method_name = "ANOVA F-statistic" if method == "anova" else "LogReg importance"
    n_bins      = len(band_scores)
    H = W       = img_side

    # identify top band and recommended beta
    top_idx   = int(np.argmax([s[2] for s in band_scores]))
    top_score = band_scores[top_idx]
    b_lo, b_hi = top_score[0], top_score[1]

    # radial distance map for ring overlays
    cy, cx      = H // 2, W // 2
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    radial_dist  = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)
    max_r        = np.sqrt(cx**2 + cy**2)
    band_edges   = np.linspace(0, max_r, n_bins + 1)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle(
        f"Frequency Band Domain-Informativeness  |  "
        f"Dataset: {dataset.upper()}  |  Method: {method_name}",
        fontsize=13, fontweight="bold", y=1.01)

    # ── Panel 1: per-pixel score map ─────────────────────────────────────
    ax = axes[0]
    sm = score_map.copy()
    sm = np.clip(sm, 1e-3, None)
    im = ax.imshow(sm, cmap="hot", norm=LogNorm(vmin=sm.min(), vmax=sm.max()))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # overlay band rings
    theta = np.linspace(0, 2*np.pi, 360)
    for i, r in enumerate(band_edges[1:]):
        color = "cyan" if i == top_idx else "white"
        lw    = 2.0   if i == top_idx else 0.4
        ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta),
                color=color, lw=lw, alpha=0.8)

    ax.set_title(f"Per-pixel {method_name}\n(cyan ring = most informative band)",
                 fontsize=10)
    ax.axis("off")

    # ── Panel 2: per-band bar chart ───────────────────────────────────────
    ax = axes[1]
    betas_lo = [s[0] for s in band_scores]
    betas_hi = [s[1] for s in band_scores]
    scores   = [s[2] for s in band_scores]
    colors   = ["#e74c3c" if i == top_idx else "#3498db"
                for i in range(n_bins)]
    x_pos    = np.arange(n_bins)
    bars     = ax.bar(x_pos, scores, color=colors, edgecolor="white", lw=0.5)

    ax.set_xticks(x_pos[::2])
    ax.set_xticklabels(
        [f"{betas_lo[i]:.2f}" for i in range(0, n_bins, 2)],
        rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Band low-β edge (fraction of img_side)", fontsize=9)
    ax.set_ylabel(method_name, fontsize=9)
    ax.set_title(f"Per-band domain informativeness\n"
                 f"Top band: β ∈ [{b_lo:.3f}, {b_hi:.3f}]  (red bar)",
                 fontsize=10)

    red_patch  = mpatches.Patch(color="#e74c3c", label=f"Most informative")
    blue_patch = mpatches.Patch(color="#3498db", label="Other bands")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: mean amplitude per domain ───────────────────────────────
    ax = axes[2]
    domain_colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))
    for i, d in enumerate(domains):
        mean_amp = np.mean([a for a in domain_amps[d]], axis=0)
        # radial profile
        r_bins   = np.linspace(0, max_r, 60)
        profile  = []
        for j in range(len(r_bins) - 1):
            mask = (radial_dist >= r_bins[j]) & (radial_dist < r_bins[j+1])
            profile.append(mean_amp[mask].mean() if mask.any() else 0.0)
        beta_vals = r_bins[:-1] / img_side
        ax.semilogy(beta_vals, profile, color=domain_colors[i],
                    label=d, lw=1.5, alpha=0.85)

    # shade top band
    ax.axvspan(b_lo, b_hi, alpha=0.18, color="red",
               label=f"Top band β=[{b_lo:.3f},{b_hi:.3f}]")
    ax.set_xlabel("β (radial frequency as fraction of img_side)", fontsize=9)
    ax.set_ylabel("Mean amplitude (log scale)", fontsize=9)
    ax.set_title("Radial amplitude profiles per domain", fontsize=10)
    ax.legend(fontsize=7, ncol=2 if len(domains) > 4 else 1)
    ax.grid(alpha=0.3)

    # ── Panel 4: recommendation ───────────────────────────────────────────
    ax = axes[3]
    ax.axis("off")
    lines = [
        f"Dataset      : {dataset.upper()}",
        f"Method       : {method_name}",
        f"Img side     : {img_side}",
        f"Domains      : {len(domains)}",
        "",
        "─── Top 3 bands by score ───",
    ]
    sorted_bands = sorted(
        enumerate(band_scores), key=lambda x: x[1][2], reverse=True)
    for rank, (idx, (lo, hi, sc)) in enumerate(sorted_bands[:3]):
        lines.append(
            f"  #{rank+1}  β ∈ [{lo:.3f}, {hi:.3f}]  score={sc:.4f}")

    lines += [
        "",
        "─── Recommended CONFIG ─────",
        f'  "fft_beta" : {(b_lo+b_hi)/2:.3f}',
        f"  (midpoint of top band)",
        "",
        "─── Band edges (all) ────────",
    ]
    for lo, hi, sc in band_scores:
        marker = " ◄" if (lo == b_lo and hi == b_hi) else ""
        lines.append(f"  β [{lo:.3f}–{hi:.3f}]  {sc:.4f}{marker}")

    ax.text(0.05, 0.97, "\n".join(lines),
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow",
                      edgecolor="orange", alpha=0.9))
    ax.set_title("Analysis summary & recommendation", fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"band_analysis_{dataset}_{method}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {fig_path}")
    return fig_path


# ══════════════════════════════════════════════════════════════
#  SAVE SCORES TEXT
# ══════════════════════════════════════════════════════════════

def save_scores(band_scores, method, dataset, img_side, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"band_scores_{dataset}_{method}.txt")
    with open(path, "w") as f:
        f.write(f"Dataset : {dataset}\n")
        f.write(f"Method  : {method}\n")
        f.write(f"Img side: {img_side}\n\n")
        f.write(f"{'Band':>4}  {'beta_lo':>8}  {'beta_hi':>8}  "
                f"{'Score':>12}  {'Note'}\n")
        f.write("─" * 55 + "\n")
        sorted_scores = sorted(band_scores, key=lambda x: x[2], reverse=True)
        top_band = sorted_scores[0]
        for i, (lo, hi, sc) in enumerate(band_scores):
            note = "◄ MOST INFORMATIVE" if (lo, hi) == (top_band[0], top_band[1]) else ""
            f.write(f"{i+1:>4}  {lo:>8.4f}  {hi:>8.4f}  {sc:>12.6f}  {note}\n")
        f.write("\n")
        f.write("Top 3 bands:\n")
        for rank, (lo, hi, sc) in enumerate(sorted_scores[:3]):
            f.write(f"  #{rank+1}  beta in [{lo:.4f}, {hi:.4f}]  "
                    f"score={sc:.6f}  "
                    f"recommended fft_beta={( lo+hi)/2:.4f}\n")
    print(f"  Scores saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    cfg     = CONFIG
    dataset = cfg["dataset"].strip().lower()
    method  = cfg["method"].strip().lower()
    seed    = cfg["random_seed"]
    H = W   = cfg["img_side"]

    random.seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  Frequency Band Analysis")
    print(f"  Dataset : {dataset.upper()}")
    print(f"  Method  : {method}")
    print(f"  Bins    : {cfg['n_bins']}")
    print(f"{'='*60}\n")

    # ── load amplitudes ───────────────────────────────────────────────────
    print("Loading amplitude templates …")
    if dataset == "casiams":
        domain_amps = load_casiams(
            cfg["casiams_root"], cfg["n_ids"],
            cfg["n_imgs_per_id"], H, seed)
    elif dataset == "xjtu":
        domain_amps = load_xjtu(
            cfg["xjtu_root"], cfg["n_ids"],
            cfg["n_imgs_per_id"], H, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_total = sum(len(v) for v in domain_amps.values())
    print(f"\n  Domains: {list(domain_amps.keys())}")
    print(f"  Total amplitude arrays: {n_total}")

    # ── run selected method ───────────────────────────────────────────────
    if method == "anova":
        score_map, band_scores = analyse_anova(
            domain_amps, cfg["n_bins"], H)
    elif method == "logreg":
        score_map, band_scores = analyse_logreg(
            domain_amps, cfg["n_bins"], H,
            C=cfg["logreg_C"], seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── print results ─────────────────────────────────────────────────────
    print(f"\n  {'Band':>4}  {'beta_lo':>8}  {'beta_hi':>8}  "
          f"{'Score':>12}")
    print("  " + "─" * 40)
    top_score = max(s[2] for s in band_scores)
    for i, (lo, hi, sc) in enumerate(band_scores):
        marker = "  ◄ TOP" if sc == top_score else ""
        print(f"  {i+1:>4}  {lo:>8.4f}  {hi:>8.4f}  {sc:>12.4f}{marker}")

    best = max(band_scores, key=lambda x: x[2])
    print(f"\n  Most informative band : β ∈ [{best[0]:.4f}, {best[1]:.4f}]")
    print(f"  Recommended fft_beta  : {(best[0]+best[1])/2:.4f}")

    # ── save ──────────────────────────────────────────────────────────────
    save_scores(band_scores, method, dataset, H, cfg["out_dir"])
    fig_path = plot_results(
        score_map, band_scores, domain_amps,
        method, dataset, H, cfg["out_dir"])

    print(f"\n{'='*60}")
    print(f"  Done. Results in: {cfg['out_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
