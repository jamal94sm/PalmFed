# ==============================================================
#  visualize_fft_domains.py
#
#  FFT-based domain separability for CASIA-MS, XJTU, and X-Palm.
# ==============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from PIL import Image
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


# ══════════════════════════════════════════════════════════════
#  DATASET PARSERS
# ══════════════════════════════════════════════════════════════

CASIAMS_ROOT = "/home/pai-ng/Jamal/CASIA-MS-ROI"
XJTU_ROOT    = "/home/pai-ng/Jamal/XJTU-UP"
XPALM_ROOT   = "/home/pai-ng/Jamal/xpalm"

XJTU_VARIATIONS = [
    ("iPhone", "Flash"), ("iPhone", "Nature"),
    ("huawei", "Flash"), ("huawei", "Nature"),
]

# X-Palm client domain assignments (matching configs.py)
XPALM_DOMAINS = {
    "scanner-A":       ["orange", "pink", "ir"],
    "scanner-B":       ["green", "white", "yellow"],
    "phone-gesture":   ["bf", "jf", "sf", "rnd_1", "rnd_2", "rnd_3", "rnd_4", "rnd_5"],
    "phone-condition": ["close", "far", "wet", "text", "fl", "pitch", "roll"],
}


def parse_casia_ms(data_root):
    """spectrum → {identity: [path, ...]}"""
    data     = defaultdict(lambda: defaultdict(list))
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for fname in sorted(os.listdir(data_root)):
        if Path(fname).suffix.lower() not in img_exts:
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4:
            continue
        identity = f"{parts[0]}_{parts[1]}"
        spectrum = parts[2]
        data[spectrum][identity].append(os.path.join(data_root, fname))
    return data


def parse_xjtu(data_root):
    """domain_label → {identity: [path, ...]}"""
    IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}
    data     = defaultdict(lambda: defaultdict(list))
    for device, condition in XJTU_VARIATIONS:
        label   = f"{device}/{condition}"
        var_dir = os.path.join(data_root, device, condition)
        if not os.path.isdir(var_dir):
            continue
        for id_folder in sorted(os.listdir(var_dir)):
            id_dir = os.path.join(var_dir, id_folder)
            if not os.path.isdir(id_dir):
                continue
            for fname in sorted(os.listdir(id_dir)):
                if Path(fname).suffix.lower() not in IMG_EXTS:
                    continue
                data[label][id_folder].append(os.path.join(id_dir, fname))
    return data


def parse_xpalm(data_root):
    """
    Groups images by client domain (scanner-A, scanner-B, etc.)
    Returns {domain_label: {identity: [path, ...]}}
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    data = defaultdict(lambda: defaultdict(list))

    # Build reverse map: variation → domain_label
    var_to_domain = {}
    for domain_label, variations in XPALM_DOMAINS.items():
        for var in variations:
            var_to_domain[var.lower()] = domain_label

    # Scanner
    scanner_dir = os.path.join(data_root, "scanner_roi")
    if os.path.isdir(scanner_dir):
        for subj_folder in sorted(os.listdir(scanner_dir)):
            subj_dir = os.path.join(scanner_dir, subj_folder)
            if not os.path.isdir(subj_dir):
                continue
            for fname in sorted(os.listdir(subj_dir)):
                if Path(fname).suffix.lower() not in IMG_EXTS:
                    continue
                parts = os.path.splitext(fname)[0].split("_")
                if len(parts) < 4:
                    continue
                hand = parts[1].lower()
                spectrum = parts[2].lower()
                identity = f"{subj_folder}_{hand}"
                domain = var_to_domain.get(spectrum)
                if domain:
                    data[domain][identity].append(
                        os.path.join(subj_dir, fname))

    # Smartphone
    phone_dir = os.path.join(data_root, "smartphone_roi")
    if os.path.isdir(phone_dir):
        for subj_folder in sorted(os.listdir(phone_dir)):
            subj_dir = os.path.join(phone_dir, subj_folder)
            if not os.path.isdir(subj_dir):
                continue
            for fname in sorted(os.listdir(subj_dir)):
                if Path(fname).suffix.lower() not in IMG_EXTS:
                    continue
                parts = os.path.splitext(fname)[0].split("_")
                if len(parts) < 3:
                    continue
                hand = parts[1].lower()
                variation = "_".join(parts[2:]).lower()
                identity = f"{subj_folder}_{hand}"
                domain = var_to_domain.get(variation)
                if domain:
                    data[domain][identity].append(
                        os.path.join(subj_dir, fname))

    return data


# ══════════════════════════════════════════════════════════════
#  COLOUR PALETTES
# ══════════════════════════════════════════════════════════════

CASIAMS_COLORS = {
    "460": "#4477EE", "630": "#EE4444", "700": "#FF8800",
    "850": "#9944CC", "940": "#22AA44", "WHT": "#888888",
}
CASIAMS_LABELS = {
    "460": "460nm (blue visible)", "630": "630nm (red visible)",
    "700": "700nm (deep red)",     "850": "850nm (NIR)",
    "940": "940nm (NIR)",          "WHT": "WHT (broadband white)",
}

XJTU_COLORS = {
    "iPhone/Flash":  "#FFCC99", "iPhone/Nature": "#FFFF88",
    "huawei/Flash":  "#E1D5E7", "huawei/Nature": "#DAE8FC",
}
XJTU_LABELS = {
    "iPhone/Flash":  "iPhone — Flash",     "iPhone/Nature": "iPhone — Natural",
    "huawei/Flash":  "Huawei — Flash",     "huawei/Nature": "Huawei — Natural",
}

XPALM_COLORS = {
    "scanner-A":       "#FFCC99",
    "scanner-B":       "#FFFF88",
    "phone-gesture":   "#E1D5E7",
    "phone-condition": "#DAE8FC",
}
XPALM_LABELS = {
    "scanner-A":       "Scanner-A (orange, pink, ir)",
    "scanner-B":       "Scanner-B (green, white, yellow)",
    "phone-gesture":   "Phone — Gesture (bf, jf, sf, rnd)",
    "phone-condition": "Phone — Condition (close, far, wet, ...)",
}


def get_palette(dataset):
    if dataset == "casiams":
        return CASIAMS_COLORS, CASIAMS_LABELS
    elif dataset == "xpalm":
        return XPALM_COLORS, XPALM_LABELS
    return XJTU_COLORS, XJTU_LABELS


# ══════════════════════════════════════════════════════════════
#  FFT DESCRIPTOR
# ══════════════════════════════════════════════════════════════

try:
    from scipy.ndimage import gaussian_filter
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


def hard_mask(H, W, beta):
    cy, cx = H // 2, W // 2
    ys, xs = np.arange(H) - cy, np.arange(W) - cx
    xs, ys = np.meshgrid(xs, ys)
    radius = min(H, W) * beta
    return ((xs**2 + ys**2) <= radius**2).astype(np.float32)


def radial_profile(amp_2d, n_bins=64):
    H, W   = amp_2d.shape
    cy, cx = H // 2, W // 2
    Y, X   = np.ogrid[:H, :W]
    R      = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r_max  = min(H, W) / 2.0
    bins   = np.zeros(n_bins, dtype=np.float32)
    edges  = np.linspace(0, r_max, n_bins + 1)
    for b in range(n_bins):
        ring = (R >= edges[b]) & (R < edges[b + 1])
        if ring.any():
            bins[b] = amp_2d[ring].mean()
    return bins


def _radial_distance_map(H, W):
    cy, cx = H // 2, W // 2
    Y, X   = np.ogrid[:H, :W]
    return np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)


def _band_energy_ratios(amp_2d, R, r_max):
    def E(r0, r1):
        m = (R >= r0 * r_max) & (R < r1 * r_max)
        return float(amp_2d[m].mean()) if m.any() else 1e-8
    eps = 1e-8
    E_ul = E(0.00, 0.05); E_l = E(0.05, 0.15)
    E_m  = E(0.15, 0.35); E_h = E(0.35, 0.65); E_uh = E(0.65, 1.00)
    return np.array([E_h/(E_l+eps), E_uh/(E_l+eps), E_h/(E_ul+eps),
                     E_m/(E_l+eps), E_uh/(E_m+eps), E_h/(E_m+eps)],
                    dtype=np.float32)


def extract_descriptor(path, img_side, beta, n_bins=64, mode="radial",
                       alpha=1.0):
    img    = Image.open(path).convert("L").resize(
        (img_side, img_side), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    amp    = np.fft.fftshift(np.abs(np.fft.fft2(img_np)))

    if mode == "raw":
        mask = hard_mask(img_side, img_side, beta)
        desc = (amp * mask).flatten()
    elif mode == "radial":
        amp_log = np.log1p(amp)
        mask    = hard_mask(img_side, img_side, beta)
        desc    = (amp_log * mask).flatten()
    else:  # sensorprint
        if _SCIPY_OK:
            blurred  = gaussian_filter(img_np, sigma=2.0)
            residual = img_np - blurred
        else:
            residual = img_np
        amp_res   = np.fft.fftshift(np.abs(np.fft.fft2(residual)))
        amp_log   = np.log1p(amp_res)
        H, W      = amp_log.shape
        R         = _radial_distance_map(H, W)
        divisor   = np.maximum(R ** alpha, 1e-6)
        amp_white = amp_log / divisor
        amp_white[R < 1] = 0.0
        mask   = hard_mask(H, W, beta)
        r_prof = radial_profile(amp_white * mask, n_bins=n_bins)
        r_max  = min(H, W) / 2.0
        ratios = _band_energy_ratios(amp_white, R, r_max)
        desc   = np.concatenate([r_prof, ratios])

    return amp, desc


# ══════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════

def plot_scatter(coords, domain_labels_list, domains, title, ax,
                 colors, labels_map, alpha=0.5, size=12):
    for sp in domains:
        mask = np.array(domain_labels_list) == sp
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors.get(sp, "#333333"),
                       label=labels_map.get(sp, sp),
                       alpha=alpha, s=size, linewidths=0,
                       edgecolors='#333333')
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)


def plot_mean_heatmaps(mean_amps, domains, img_side, beta,
                       colors, labels_map, out_path):
    n     = len(domains)
    mask  = hard_mask(img_side, img_side, beta)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4*nrows))
    axes = np.array(axes).flatten()

    log_amps = {sp: np.log1p(mean_amps[sp]) * mask for sp in domains}
    vmax = max(a.max() for a in log_amps.values())

    for ax, sp in zip(axes, domains):
        im = ax.imshow(log_amps[sp], cmap="inferno",
                       vmin=0, vmax=vmax, origin="upper")
        ax.set_title(labels_map.get(sp, sp), fontsize=10,
                     color=colors.get(sp, "#333"))
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Mean Low-Frequency FFT Amplitude per Domain\n"
        f"log(1+amp) · hard circular mask  (β={beta})",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_distance_matrix(mean_descs, domains, labels_map, out_path):
    n    = len(domains)
    vecs = np.stack([mean_descs[sp] for sp in domains])
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    dist = 1.0 - (norm @ norm.T)

    fig, ax = plt.subplots(figsize=(max(5, n*1.2), max(4, n*1.0)))
    im      = ax.imshow(dist, cmap="RdYlGn_r", vmin=0, vmax=dist.max())
    short   = [labels_map.get(sp, sp).split("(")[0].split("—")[0].strip()
               for sp in domains]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist[i,j]:.3f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if dist[i,j] > dist.max()*0.5 else "black")
    plt.colorbar(im, ax=ax, label="Cosine Distance")
    ax.set_title("Pairwise Domain Distance\n(FFT profile, log-amplitude)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_radial_profiles(mean_descs, domains, colors, labels_map, beta, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for sp in domains:
        profile = mean_descs[sp]
        ax.plot(profile, color=colors.get(sp, "#333"),
                label=labels_map.get(sp, sp), linewidth=2)
    ax.set_xlabel("Frequency bin (low → high frequency)", fontsize=11)
    ax.set_ylabel("Mean log(1 + amplitude)", fontsize=11)
    ax.set_title(f"Radial Spectral Profile per Domain  (β={beta})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def classify_domains(descriptors, domain_labels_list, domains,
                     classifier="svm", desc_mode="radial",
                     out_path=None, seed=42):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score)
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline

    X  = np.stack(descriptors).astype(np.float32)
    le = LabelEncoder()
    le.fit(domains)
    y  = le.transform(domain_labels_list)

    orig_dim = X.shape[1]
    if orig_dim > 200:
        from sklearn.decomposition import PCA as _PCA
        n_comp = min(64, X.shape[0] - 1, orig_dim)
        X      = _PCA(n_components=n_comp, random_state=seed).fit_transform(X)
        print(f"  PCA pre-reduction: {orig_dim}-d → {n_comp}-d")

    if classifier == "svm":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10.0, gamma="scale",
                           random_state=seed)),
        ])
    else:
        hidden = (256, 128) if X.shape[1] > 100 else (128, 64)
        model  = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(hidden_layer_sizes=hidden,
                                     max_iter=500, random_state=seed,
                                     early_stopping=True)),
        ])

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    acc    = accuracy_score(y, y_pred) * 100
    report = classification_report(y, y_pred,
                                   target_names=le.classes_, digits=3)
    cm     = confusion_matrix(y, y_pred)

    print(f"\n{'='*60}")
    print(f"  Domain Classifier — {classifier.upper()}  |  "
          f"descriptor: {desc_mode}  |  dim={X.shape[1]}")
    print(f"  5-fold CV accuracy: {acc:.2f}%")
    print("=" * 60)
    print(report)

    if out_path is not None:
        fig, ax = plt.subplots(figsize=(max(5, len(domains)*0.9),
                                        max(4, len(domains)*0.8)))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        im      = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        labels_s = [l.split("(")[0].strip() for l in le.classes_]
        ax.set_xticks(range(len(domains))); ax.set_yticks(range(len(domains)))
        ax.set_xticklabels(labels_s, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels_s, fontsize=9)
        for i in range(len(domains)):
            for j in range(len(domains)):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                        fontsize=9,
                        color="white" if cm_norm[i,j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, label="Normalised count")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Domain Classifier — {classifier.upper()}\n"
                     f"Descriptor: {desc_mode} ({X.shape[1]}-d)  "
                     f"Acc={acc:.1f}%  5-fold CV",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    return acc, report


def plot_projections(descriptors, domain_labels_list, domains, methods,
                     colors, labels_map, dataset_name, beta, n_total,
                     out_path, seed=42):
    X = np.stack(descriptors)
    X = normalize(X, norm="l2")

    MAX_TSNE = 2000
    if len(X) > MAX_TSNE and ("tsne" in methods or "umap" in methods):
        rng    = np.random.RandomState(seed)
        idx    = rng.choice(len(X), MAX_TSNE, replace=False)
        X_sub  = X[idx]
        dl_sub = [domain_labels_list[i] for i in idx]
        print(f"  Subsampled to {MAX_TSNE} for t-SNE/UMAP")
    else:
        X_sub, dl_sub = X, domain_labels_list

    results = {}

    if "pca" in methods:
        results["PCA"] = (PCA(n_components=2, random_state=seed)
                          .fit_transform(X), domain_labels_list)
    if "tsne" in methods:
        n_pca = min(50, X_sub.shape[1], X_sub.shape[0] - 1)
        X_pca = PCA(n_components=n_pca, random_state=seed).fit_transform(X_sub)
        perp  = min(30, X_sub.shape[0] // 4)
        import sklearn
        kw = dict(n_components=2, perplexity=perp, random_state=seed, verbose=0)
        sv = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        kw["max_iter" if sv >= (1, 5) else "n_iter"] = 1000
        results["t-SNE"] = (TSNE(**kw).fit_transform(X_pca), dl_sub)
        print(f"    t-SNE done (perplexity={perp})")
    if "umap" in methods:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=seed,
                                n_neighbors=15, min_dist=0.1)
            results["UMAP"] = (reducer.fit_transform(X_sub), dl_sub)
            print("    UMAP done")
        except ImportError:
            print("    UMAP not installed — skipping")

    n_plots = len(results)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6.5*n_plots, 6.5))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, (coords, lbls)) in zip(axes, results.items()):
        plot_scatter(coords, lbls, domains, name, ax,
                     colors, labels_map, alpha=0.55, size=14)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=colors.get(sp, "#333"),
               markeredgecolor="#333333", markeredgewidth=0.5,
               markersize=9, label=labels_map.get(sp, sp))
        for sp in domains
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(domains), 4), fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True)

    fig.suptitle(
        f"FFT Domain Separability — {dataset_name.upper()}\n"
        f"β={beta}  |  {n_total} images",
        fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_sample_grid(data, domains, colors, labels_map, img_side,
                     out_path, n_samples=5, seed=42):
    """Show sample images from each domain in a grid."""
    rng   = np.random.RandomState(seed)
    n_dom = len(domains)

    fig, axes = plt.subplots(n_dom, n_samples,
                              figsize=(2.2*n_samples, 2.5*n_dom))
    if n_dom == 1:
        axes = [axes]

    for row, sp in enumerate(domains):
        all_paths = [p for paths in data[sp].values() for p in paths]
        rng.shuffle(all_paths)
        selected = all_paths[:n_samples]
        for col in range(n_samples):
            ax = axes[row][col] if n_dom > 1 else axes[col]
            if col < len(selected):
                img = Image.open(selected[col]).convert("L").resize(
                    (img_side, img_side))
                ax.imshow(np.array(img), cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(labels_map.get(sp, sp), fontsize=9,
                              rotation=0, labelpad=80, va="center",
                              color=colors.get(sp, "#333"),
                              fontweight="bold")

    fig.suptitle("Sample Images per Domain", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Configuration ──────────────────────────────────────────
    DATASET        = "xpalm"       # casiams | xjtu | xpalm
    DATA_ROOT      = None          # None → auto from dataset
    DESC_MODE      = "radial"      # radial | raw | sensorprint
    BETA           = 0.4
    N_BINS         = 64
    ALPHA          = 1.0
    METHODS        = ["pca", "tsne", "umap"]
    USE_UMAP       = True
    CLASSIFIER     = "both"        # svm | nn | both
    MAX_PER_DOMAIN = None          # None → all
    IMG_SIDE       = 128
    SEED           = 42
    OUT_DIR        = "./Figs"
    N_SAMPLE_IMGS  = 6             # sample images per domain in grid

    # ── Resolve paths ──────────────────────────────────────────
    if DATA_ROOT is None:
        DATA_ROOT = {"casiams": CASIAMS_ROOT, "xjtu": XJTU_ROOT,
                     "xpalm": XPALM_ROOT}[DATASET]

    methods = [m for m in METHODS if m != "umap" or USE_UMAP]
    os.makedirs(OUT_DIR, exist_ok=True)
    colors, labels_map = get_palette(DATASET)

    print(f"\n{'='*56}")
    print(f"  FFT Domain Visualisation — {DATASET.upper()}")
    print(f"  data_root      : {DATA_ROOT}")
    print(f"  desc_mode      : {DESC_MODE}")
    print(f"  beta           : {BETA}   n_bins={N_BINS}")
    print(f"  classifier     : {CLASSIFIER}")
    print(f"  out_dir        : {OUT_DIR}")
    print(f"{'='*56}\n")

    # ── 1. Parse ───────────────────────────────────────────────
    print(f"Parsing {DATASET.upper()} …")
    if DATASET == "casiams":
        data = parse_casia_ms(DATA_ROOT)
    elif DATASET == "xjtu":
        data = parse_xjtu(DATA_ROOT)
    else:
        data = parse_xpalm(DATA_ROOT)

    if DATASET == "casiams":
        domains = sorted(data.keys())
    elif DATASET == "xjtu":
        domains = [f"{d}/{c}" for d, c in XJTU_VARIATIONS
                   if f"{d}/{c}" in data]
    else:
        domains = [d for d in XPALM_DOMAINS.keys() if d in data]

    for sp in domains:
        n_ids = len(data[sp])
        n_img = sum(len(v) for v in data[sp].values())
        print(f"  {sp:>18}  IDs={n_ids}  images={n_img}")

    tag = f"{DATASET}_{DESC_MODE}_beta{BETA}"

    # ── 2. Sample image grid ───────────────────────────────────
    print(f"\nPlotting sample images …")
    plot_sample_grid(data, domains, colors, labels_map, IMG_SIDE,
                     out_path=os.path.join(OUT_DIR, f"samples_{DATASET}.png"),
                     n_samples=N_SAMPLE_IMGS, seed=SEED)

    # ── 3. Extract descriptors ─────────────────────────────────
    print(f"\nExtracting FFT descriptors (β={BETA}) …")
    descriptors       = []
    domain_labels_arr = []
    mean_amps         = {}
    mean_descs        = {}

    for sp in domains:
        paths = [p for paths in data[sp].values() for p in paths]
        if MAX_PER_DOMAIN is not None:
            rng   = np.random.RandomState(SEED)
            paths = list(rng.choice(paths,
                                     min(MAX_PER_DOMAIN, len(paths)),
                                     replace=False))
        sp_descs, sp_amps = [], []
        for path in paths:
            try:
                amp, desc = extract_descriptor(
                    path, IMG_SIDE, BETA,
                    n_bins=N_BINS, mode=DESC_MODE, alpha=ALPHA)
                descriptors.append(desc)
                domain_labels_arr.append(sp)
                sp_descs.append(desc)
                sp_amps.append(amp)
            except Exception as e:
                print(f"    [WARN] {path}: {e}")

        if sp_descs:
            mean_descs[sp] = np.stack(sp_descs).mean(axis=0)
            mean_amps[sp]  = np.stack(sp_amps).mean(axis=0)
            print(f"  {sp:>18}  {len(sp_descs)} descriptors")

    print(f"\n  Total: {len(descriptors)} descriptors  "
          f"(dim={descriptors[0].shape[0]})")

    # ── 4. Scatter projections ─────────────────────────────────
    print("\nComputing projections …")
    plot_projections(
        descriptors, domain_labels_arr, domains, methods,
        colors, labels_map, DATASET, BETA, len(descriptors),
        out_path=os.path.join(OUT_DIR, f"fft_scatter_{tag}.png"),
        seed=SEED)

    # ── 5. Mean amplitude heatmaps ────────────────────────────
    print("\nPlotting mean amplitude heatmaps …")
    plot_mean_heatmaps(
        mean_amps, domains, IMG_SIDE, BETA,
        colors, labels_map,
        out_path=os.path.join(OUT_DIR, f"fft_heatmaps_{tag}.png"))

    # ── 6. Pairwise distance matrix ────────────────────────────
    print("Plotting pairwise distance matrix …")
    plot_distance_matrix(
        mean_descs, domains, labels_map,
        out_path=os.path.join(OUT_DIR, f"fft_distances_{tag}.png"))

    # ── 7. Radial profile curves ───────────────────────────────
    print("Plotting radial spectral profiles …")
    plot_radial_profiles(
        mean_descs, domains, colors, labels_map, BETA,
        out_path=os.path.join(OUT_DIR, f"fft_radial_{tag}.png"))

    # ── 8. Domain classifier ───────────────────────────────────
    clf_names = ["svm", "nn"] if CLASSIFIER == "both" else [CLASSIFIER]
    print(f"\nTraining domain classifier(s): {clf_names} …")
    clf_results = {}
    for clf in clf_names:
        cm_path = os.path.join(OUT_DIR, f"clf_confusion_{clf}_{tag}.png")
        acc, _ = classify_domains(
            descriptors, domain_labels_arr, domains,
            classifier=clf, desc_mode=DESC_MODE,
            out_path=cm_path, seed=SEED)
        clf_results[clf] = acc

    # ── 9. Console summary ─────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"  Summary — {DATASET.upper()}")
    print(f"  Descriptor: {DESC_MODE} ({descriptors[0].shape[0]}-d)")
    for clf, acc in clf_results.items():
        print(f"  {clf.upper():<6} accuracy: {acc:.2f}%")
    print(f"{'='*56}")

    print("\nPairwise cosine distances:")
    vecs = np.stack([mean_descs[sp] for sp in domains])
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    dist = 1.0 - (norm @ norm.T)
    w = max(len(sp) for sp in domains) + 2
    print(" " * w + "".join(f"{sp:>{w}}" for sp in domains))
    for i, sp_i in enumerate(domains):
        print(f"{sp_i:>{w}}" + "".join(f"{dist[i,j]:>{w}.4f}"
                                        for j in range(len(domains))))

    print(f"\nDone. Figures saved to: {OUT_DIR}/")
