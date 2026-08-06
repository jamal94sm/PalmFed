# ==============================================================
# run_clustering.py — Standalone domain-aware clustering runner.
#
# Loads federated splits via datasets.py, builds a style_bank via
# utils.extract_style_template (same path as main.py::build_style_bank),
# then clusters clients using clustering.py under the "domain_aware"
# strategy for all three datasets. Independent of the FL training loop —
# no models are built, no training happens.
#
# Reads clustering params from configs.py::CONFIG (the "proposed"
# config), overridable via CLI flags below.





#python run_clustering.py --datasets casiams --cluster_scale log --cluster_partition spectral




# ==============================================================
import argparse
import json
import numpy as np
from PIL import Image

from configs import get_config
from datasets import get_federated_splits
from utils import extract_style_template
from clustering import cluster_clients_by_style, build_distance_matrix, \
    build_client_style_vectors


def build_style_bank(client_data, img_side):
    """
    Identical to main.py::build_style_bank — duplicated here so this
    script has zero import dependency on main.py (keeps it standalone
    and runnable without pulling in model/training code).
    """
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


def run_for_dataset(dataset_name, args):
    print(f"\n{'='*70}")
    print(f" Clustering — {dataset_name.upper()}")
    print(f"{'='*70}")

    cfg = get_config("proposed")
    cfg["dataset"] = dataset_name
    if dataset_name == "xjtu":
        cfg["n_ids"] = args.xjtu_n_ids

    # ── 1. Load federated splits (clients + train_samples) ──
    client_data, _, _, _, spectra = get_federated_splits(cfg, seed=cfg["random_seed"])
    n_clients = len(client_data)
    print(f" Clients: {n_clients} — {[cd['spectrum'] for cd in client_data]}")

    # ── 2. Build style_bank (full amplitude templates per client) ──
    style_bank = build_style_bank(client_data, cfg["img_side"])
    for ci, templates in style_bank.items():
        print(f"   Client {ci} [{client_data[ci]['spectrum']}]: "
              f"{len(templates)} templates")

    # ── 3. Cluster (domain-aware, Mahalanobis) ──
    style_vectors = build_client_style_vectors(
        style_bank, alpha=args.cluster_alpha, scale=args.cluster_scale)
    client_ids, dist_matrix = build_distance_matrix(
        style_vectors, diagonal=args.cluster_diagonal,
        epsilon=args.cluster_epsilon)

    short_ids, long_ids = cluster_clients_by_style(
        style_bank,
        alpha=args.cluster_alpha,
        scale=args.cluster_scale,
        diagonal=args.cluster_diagonal,
        epsilon=args.cluster_epsilon,
        partition_method=args.cluster_partition,
    )

    cluster_a_names = [client_data[i]["spectrum"] for i in short_ids]
    cluster_b_names = [client_data[i]["spectrum"] for i in long_ids]

    print(f"\n Distance matrix (client order: {client_ids}):")
    print(np.array2string(dist_matrix, precision=3, suppress_small=True))
    print(f"\n Cluster A ({args.cluster_partition}): {short_ids} -> {cluster_a_names}")
    print(f" Cluster B ({args.cluster_partition}): {long_ids} -> {cluster_b_names}")

    return {
        "dataset": dataset_name,
        "clustering_mode": "domain_aware",
        "cluster_alpha": args.cluster_alpha,
        "cluster_scale": args.cluster_scale,
        "cluster_diagonal": args.cluster_diagonal,
        "cluster_epsilon": args.cluster_epsilon,
        "cluster_partition": args.cluster_partition,
        "client_names": [cd["spectrum"] for cd in client_data],
        "distance_matrix": dist_matrix.tolist(),
        "cluster_a": {"ids": short_ids, "names": cluster_a_names},
        "cluster_b": {"ids": long_ids, "names": cluster_b_names},
    }


def parse_args():
    base_cfg = get_config("proposed")  # pull defaults straight from configs.py
    p = argparse.ArgumentParser(description="Standalone domain-aware clustering")
    p.add_argument("--datasets", nargs="+",
                    default=["casiams", "xjtu", "xpalm"],
                    choices=["casiams", "xjtu", "xpalm"])
    p.add_argument("--xjtu_n_ids", type=int, default=192)
    p.add_argument("--cluster_alpha", type=float,
                    default=base_cfg["cluster_alpha"])
    p.add_argument("--cluster_scale", choices=["linear", "log"],
                    default=base_cfg["cluster_scale"])
    p.add_argument("--cluster_diagonal", type=lambda v: v.lower() == "true",
                    default=base_cfg["cluster_diagonal"])
    p.add_argument("--cluster_epsilon", type=float,
                    default=base_cfg["cluster_epsilon"])
    p.add_argument("--cluster_partition",
                    choices=["farthest_pair", "agglomerative", "spectral"],
                    default=base_cfg["cluster_partition"])
    p.add_argument("--out", default="clustering_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    all_results = []
    for dataset_name in args.datasets:
        result = run_for_dataset(dataset_name, args)
        all_results.append(result)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
