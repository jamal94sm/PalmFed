# ==============================================================
# clustering.py — Domain-aware client clustering via Mahalanobis
# distance on low-frequency FFT style descriptors (paper §3.2,
# Eq. 7-10). Operates on an existing style_bank (see
# main.py::build_style_bank / utils.py::extract_style_template);
# does not touch raw images or paths.
# ==============================================================
import numpy as np


# ══════════════════════════════════════════════════════════════
# DESCRIPTOR: crop + optional log-scale (Eq. 3, 7)
# ══════════════════════════════════════════════════════════════
def crop_lowfreq(amp_template, alpha=0.15, scale="linear"):
    """
    amp_template : np.ndarray [H, W] (or [H, W, C])
        Full, fftshift'ed amplitude spectrum, as produced by
        utils.extract_style_template(). DC component assumed centered.
    alpha : float
        Half-width of the low-frequency crop, as a fraction of H/W
        (matches the FFT-augmentation beta family; paper uses 0.15).
    scale : "linear" | "log"
        "linear": use raw amplitude (paper Eq. 2 as written).
        "log":    apply log1p before flattening (numerically
                  stabilizes Mahalanobis distances; see fdd.py).

    Returns
    -------
    np.ndarray [r] flattened low-frequency descriptor,
    r = (2*floor(alpha*H)+1) * (2*floor(alpha*W)+1) [* C if multichannel]
    """
    if amp_template.ndim == 3:
        # average across channels to keep descriptor single-channel;
        # avoids r scaling by C (keeps dimensionality/sample-count
        # ratio tractable per the earlier reviewer discussion)
        amp_template = amp_template.mean(axis=-1)

    H, W = amp_template.shape
    cy, cx = H // 2, W // 2
    ly = int(alpha * H)
    lx = int(alpha * W)

    crop = amp_template[cy - ly: cy + ly + 1, cx - lx: cx + lx + 1]

    if scale == "log":
        crop = np.log1p(crop)
    elif scale != "linear":
        raise ValueError(f"Unknown scale '{scale}', expected 'linear' or 'log'")

    return crop.astype(np.float32).flatten()


# ══════════════════════════════════════════════════════════════
# PER-CLIENT STYLE VECTORS
# ══════════════════════════════════════════════════════════════
def build_client_style_vectors(style_bank, alpha=0.15, scale="linear"):
    """
    style_bank : dict {client_id: list[np.ndarray]}
        As produced by main.py::build_style_bank — each entry is a
        list of full amplitude templates for that client's images.

    Returns
    -------
    dict {client_id: np.ndarray [n_i, r]}
    """
    vectors = {}
    for ci, templates in style_bank.items():
        vectors[ci] = np.stack(
            [crop_lowfreq(t, alpha=alpha, scale=scale) for t in templates],
            axis=0,
        )
    return vectors


# ══════════════════════════════════════════════════════════════
# GAUSSIAN FIT (Eq. 7-8)
# ══════════════════════════════════════════════════════════════
def fit_client_gaussian(vectors, diagonal=True, epsilon=0.1):
    """
    vectors : np.ndarray [n_i, r]

    Returns
    -------
    mu : np.ndarray [r]
    sigma : np.ndarray [r] if diagonal else [r, r]
        Regularized per mahalanobis_distance's convention — this
        function returns the RAW (unregularized) estimate; shrinkage
        is applied at distance-computation time so epsilon can be
        swept without refitting.
    """
    mu = vectors.mean(axis=0)
    if diagonal:
        sigma = vectors.var(axis=0)
    else:
        centered = vectors - mu[None, :]
        n = vectors.shape[0]
        sigma = (centered.T @ centered) / max(n - 1, 1)
    return mu, sigma


# ══════════════════════════════════════════════════════════════
# SYMMETRIC MAHALANOBIS DISTANCE (Eq. 9)
# ══════════════════════════════════════════════════════════════
def mahalanobis_distance(mu_i, sigma_i, mu_j, sigma_j,
                          diagonal=True, epsilon=0.1):
    diff = mu_i - mu_j
    if diagonal:
        sigma_i_reg = (1 - epsilon) * sigma_i + epsilon
        sigma_j_reg = (1 - epsilon) * sigma_j + epsilon
        inv_sum = 1.0 / sigma_i_reg + 1.0 / sigma_j_reg
        return 0.5 * float((diff ** 2 * inv_sum).sum())
    else:
        r = sigma_i.shape[0]
        eye = np.eye(r, dtype=sigma_i.dtype)
        sigma_i_reg = (1 - epsilon) * sigma_i + epsilon * eye
        sigma_j_reg = (1 - epsilon) * sigma_j + epsilon * eye
        inv_sum = np.linalg.inv(sigma_i_reg) + np.linalg.inv(sigma_j_reg)
        return 0.5 * float(diff @ inv_sum @ diff)


def build_distance_matrix(style_vectors_by_client, diagonal=True, epsilon=0.1):
    """
    Returns
    -------
    client_ids : list[int] (sorted, defines row/col order)
    dist : np.ndarray [N, N]
    """
    client_ids = sorted(style_vectors_by_client.keys())
    n = len(client_ids)
    gaussians = {
        ci: fit_client_gaussian(style_vectors_by_client[ci], diagonal=diagonal)
        for ci in client_ids
    }
    dist = np.zeros((n, n), dtype=np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            mu_i, sigma_i = gaussians[client_ids[a]]
            mu_j, sigma_j = gaussians[client_ids[b]]
            d = mahalanobis_distance(mu_i, sigma_i, mu_j, sigma_j,
                                      diagonal=diagonal, epsilon=epsilon)
            dist[a, b] = dist[b, a] = d
    return client_ids, dist


# ══════════════════════════════════════════════════════════════
# PARTITION INTO TWO CLUSTERS — three interchangeable strategies
# ══════════════════════════════════════════════════════════════
def _partition_farthest_pair(client_ids, dist):
    n = len(client_ids)
    a, b = np.unravel_index(np.argmax(dist), dist.shape)
    cluster_a, cluster_b = {a}, {b}
    for k in range(n):
        if k in (a, b):
            continue
        cluster_a.add(k) if dist[k, a] <= dist[k, b] else cluster_b.add(k)
    return cluster_a, cluster_b


def _partition_agglomerative(client_ids, dist, linkage="average"):
    from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
    from scipy.spatial.distance import squareform
    condensed = squareform(dist, checks=False)
    Z = scipy_linkage(condensed, method=linkage)
    labels = fcluster(Z, t=2, criterion="maxclust")
    cluster_a = {i for i, lab in enumerate(labels) if lab == 1}
    cluster_b = {i for i, lab in enumerate(labels) if lab == 2}
    return cluster_a, cluster_b


def _partition_spectral(client_ids, dist):
    from sklearn.cluster import SpectralClustering
    sigma = dist[dist > 0].mean() if np.any(dist > 0) else 1.0
    affinity = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(affinity, 1.0)
    sc = SpectralClustering(n_clusters=2, affinity="precomputed",
                             random_state=0)
    labels = sc.fit_predict(affinity)
    cluster_a = {i for i, lab in enumerate(labels) if lab == 0}
    cluster_b = {i for i, lab in enumerate(labels) if lab == 1}
    return cluster_a, cluster_b


_PARTITION_STRATEGIES = {
    "farthest_pair": _partition_farthest_pair,
    "agglomerative": _partition_agglomerative,
    "spectral": _partition_spectral,
}


def partition_two_clusters(client_ids, dist, method="farthest_pair"):
    if method not in _PARTITION_STRATEGIES:
        raise ValueError(
            f"Unknown partition method '{method}', "
            f"expected one of {list(_PARTITION_STRATEGIES)}")
    cluster_a_idx, cluster_b_idx = _PARTITION_STRATEGIES[method](client_ids, dist)
    cluster_a = [client_ids[i] for i in sorted(cluster_a_idx)]
    cluster_b = [client_ids[i] for i in sorted(cluster_b_idx)]
    return cluster_a, cluster_b


# ══════════════════════════════════════════════════════════════
# TOP-LEVEL ENTRY POINT — main.py calls only this
# ══════════════════════════════════════════════════════════════
def cluster_clients_by_style(style_bank, alpha=0.15, scale="linear",
                              diagonal=True, epsilon=0.1,
                              partition_method="farthest_pair"):
    """
    Returns
    -------
    short_ids, long_ids : list[int]
        Naming kept consistent with main.py's manual-mode variables
        so this is a drop-in replacement downstream.
    """
    style_vectors = build_client_style_vectors(style_bank, alpha=alpha, scale=scale)
    client_ids, dist = build_distance_matrix(style_vectors, diagonal=diagonal,
                                              epsilon=epsilon)
    short_ids, long_ids = partition_two_clusters(client_ids, dist,
                                                  method=partition_method)
    return short_ids, long_ids