"""
run_benchmark.py — Fair comparison across all methods.

Generates shared data splits ONCE, then runs each method with the same
splits, seeds, and evaluation protocol. Collects results and produces
a comparison table saved to a text file.

Usage:
  python run_benchmark.py --dataset casiams --eval_protocol open_set
  python run_benchmark.py --dataset casiams --eval_protocol closed_set --closed_set_mode cross_spectrum
  python run_benchmark.py --dataset xjtu --eval_protocol open_set --n_rounds 100
"""

import os
import sys
import json
import time
import pickle
import argparse
import subprocess
from datetime import datetime

from configs import get_config, _BASE
from datasets import get_federated_splits


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark all methods")
    p.add_argument("--dataset", default="casiams",
                   choices=["casiams", "xjtu"])
    p.add_argument("--eval_protocol", default="open_set",
                   choices=["open_set", "closed_set"])
    p.add_argument("--closed_set_mode", default="cross_spectrum",
                   choices=["holdout", "cross_spectrum"])
    p.add_argument("--n_rounds", type=int, default=50)
    p.add_argument("--n_ids", type=int, default=200)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--methods", nargs="*",
                   default=["proposed", "fedpalm", "psfed"],
                   help="Methods to benchmark")
    p.add_argument("--output_dir", default="./benchmark_results")
    p.add_argument("--dp_mode", default="ideal",
                   choices=["ideal", "predicted"])
    return p.parse_args()


def generate_shared_splits(args):
    """Generate splits once, save for all methods to reuse."""
    cfg = get_config("proposed")
    cfg["dataset"] = args.dataset
    cfg["eval_protocol"] = args.eval_protocol
    cfg["closed_set_mode"] = args.closed_set_mode
    cfg["n_ids"] = args.n_ids
    cfg["random_seed"] = args.random_seed

    print(f"\n{'='*70}")
    print(f"  Generating shared data splits")
    print(f"  Dataset: {args.dataset} | Protocol: {args.eval_protocol}")
    if args.eval_protocol == "closed_set":
        print(f"  Closed-set mode: {args.closed_set_mode}")
    print(f"{'='*70}\n")

    splits = get_federated_splits(cfg, seed=args.random_seed)
    splits_path = os.path.join(args.output_dir, "shared_splits.pkl")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"\n  Splits saved: {splits_path}")

    client_data, gallery, probe, _, spectra = splits
    print(f"  Clients: {len(client_data)} | "
          f"Gallery: {len(gallery)} | Probe: {len(probe)}")

    return splits_path


def get_method_script(method):
    """Map method name to script file."""
    return {
        "proposed": "main.py",
        "fedpalm": "fedpalm.py",
        "psfed": "psfed.py",
    }[method]


def run_method(method, args, splits_path, log_dir):
    """Run a single method via subprocess."""
    script = get_method_script(method)
    log_file = os.path.join(log_dir, f"{method}.log")

    cmd = [
        sys.executable, script,
        "--dataset", args.dataset,
        "--eval_protocol", args.eval_protocol,
        "--closed_set_mode", args.closed_set_mode,
        "--n_rounds", str(args.n_rounds),
        "--random_seed", str(args.random_seed),
        "--splits_path", splits_path,
        "--eval_every", str(args.eval_every),
    ]

    if method == "proposed":
        cmd.extend(["--method", "proposed", "--dp_mode", args.dp_mode])

    print(f"\n{'─'*70}")
    print(f"  Running: {method.upper()}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")
    print(f"{'─'*70}")

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    status = "OK" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
    print(f"  {method.upper()}: {status} ({elapsed:.0f}s)")

    return log_file, proc.returncode


def parse_proposed_results(args):
    """Parse proposed method JSON results."""
    cfg = get_config("proposed")
    results_dir = cfg["base_results_dir"].replace("{dataset}", args.dataset)
    protocol = args.eval_protocol
    dp_mode = args.dp_mode

    json_path = os.path.join(results_dir,
                              f"results_{protocol}_{dp_mode}.json")
    if not os.path.exists(json_path):
        return None

    with open(json_path) as f:
        data = json.load(f)

    if not data.get("history"):
        return None

    last = data["history"][-1]
    gt = last.get("global_test", {})
    return {
        "round": last.get("round", -1),
        "global_r1": gt.get("global", {}).get("rank1", -1),
        "global_eer": gt.get("global", {}).get("eer", -1),
        "local_r1": gt.get("avg_local", {}).get("rank1", -1),
        "local_eer": gt.get("avg_local", {}).get("eer", -1),
        "moe_r1": gt.get("moe", {}).get("rank1", -1),
        "moe_eer": gt.get("moe", {}).get("eer", -1),
    }


def parse_baseline_log(log_file, method):
    """Parse final metrics from baseline log file."""
    if not os.path.exists(log_file):
        return None

    lines = open(log_file).readlines()
    result = {}

    # Look for last evaluation lines
    for line in reversed(lines):
        line = line.strip()
        if "Global" in line and "EER=" in line and "Rank-1=" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p.startswith("EER="):
                    result["global_eer"] = float(
                        p.replace("EER=", "").replace("%", ""))
                if p.startswith("Rank-1="):
                    result["global_r1"] = float(
                        p.replace("Rank-1=", "").replace("%", ""))
            break

    if method == "fedpalm":
        for line in reversed(lines):
            line = line.strip()
            if "Full FedPalm" in line and "EER=" in line:
                parts = line.split()
                for p in parts:
                    if p.startswith("EER="):
                        result["moe_eer"] = float(
                            p.replace("EER=", "").replace("%", ""))
                    if p.startswith("Rank-1="):
                        result["moe_r1"] = float(
                            p.replace("Rank-1=", "").replace("%", ""))
                break
        for line in reversed(lines):
            line = line.strip()
            if "Local avg" in line and "EER=" in line:
                parts = line.split()
                for p in parts:
                    if p.startswith("EER="):
                        result["local_eer"] = float(
                            p.replace("EER=", "").replace("%", ""))
                    if p.startswith("Rank-1="):
                        result["local_r1"] = float(
                            p.replace("Rank-1=", "").replace("%", ""))
                break

    elif method == "psfed":
        for line in reversed(lines):
            line = line.strip()
            if "Local avg" in line and "EER=" in line:
                parts = line.split()
                for p in parts:
                    if p.startswith("EER="):
                        result["local_eer"] = float(
                            p.replace("EER=", "").replace("%", ""))
                    if p.startswith("Rank-1="):
                        result["local_r1"] = float(
                            p.replace("Rank-1=", "").replace("%", ""))
                break

    return result if result else None


def collect_results(args, log_files):
    """Collect results from all methods."""
    results = {}

    for method, (log_file, retcode) in log_files.items():
        if retcode != 0:
            results[method] = None
            continue

        if method == "proposed":
            results[method] = parse_proposed_results(args)
        else:
            results[method] = parse_baseline_log(log_file, method)

    return results


def print_comparison(args, results, report_file):
    """Print and save comparison table."""
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    protocol_str = args.eval_protocol
    if args.eval_protocol == "closed_set":
        protocol_str += f" ({args.closed_set_mode})"

    out(f"\n{'='*78}")
    out(f"  BENCHMARK RESULTS — {timestamp}")
    out(f"  Dataset: {args.dataset.upper()} | Protocol: {protocol_str}")
    out(f"  Rounds: {args.n_rounds} | Seed: {args.random_seed}")
    out(f"{'='*78}")

    # Header
    out(f"\n  {'Method':>12s} │ {'Global R1':>10s} {'Global EER':>11s} │ "
        f"{'Local R1':>9s} {'Local EER':>10s} │ "
        f"{'MoE R1':>8s} {'MoE EER':>9s}")
    out(f"  {'─'*74}")

    for method in ["proposed", "fedpalm", "psfed"]:
        if method not in results or results[method] is None:
            out(f"  {method:>12s} │ {'FAILED':>10s} {'':>11s} │ "
                f"{'':>9s} {'':>10s} │ {'':>8s} {'':>9s}")
            continue

        r = results[method]
        gr1 = f"{r.get('global_r1', -1):.2f}%" if r.get('global_r1', -1) >= 0 else "—"
        geer = f"{r.get('global_eer', -1):.3f}%" if r.get('global_eer', -1) >= 0 else "—"
        lr1 = f"{r.get('local_r1', -1):.2f}%" if r.get('local_r1', -1) >= 0 else "—"
        leer = f"{r.get('local_eer', -1):.3f}%" if r.get('local_eer', -1) >= 0 else "—"
        mr1 = f"{r.get('moe_r1', -1):.2f}%" if r.get('moe_r1', -1) >= 0 else "—"
        meer = f"{r.get('moe_eer', -1):.3f}%" if r.get('moe_eer', -1) >= 0 else "—"

        label = method.upper()
        if method == "proposed":
            label = f"PROPOSED ({args.dp_mode})"

        out(f"  {label:>12s} │ {gr1:>10s} {geer:>11s} │ "
            f"{lr1:>9s} {leer:>10s} │ {mr1:>8s} {meer:>9s}")

    out(f"  {'─'*74}")

    # Best highlights
    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        # Best global EER
        best_geer = min(valid.items(),
                        key=lambda x: x[1].get("global_eer", 999))
        out(f"\n  Best Global EER:  {best_geer[0].upper()} "
            f"({best_geer[1].get('global_eer', -1):.3f}%)")

        # Best MoE/routing EER
        moe_methods = {k: v for k, v in valid.items()
                       if v.get("moe_eer", -1) >= 0}
        if moe_methods:
            best_moe = min(moe_methods.items(),
                           key=lambda x: x[1].get("moe_eer", 999))
            out(f"  Best MoE EER:     {best_moe[0].upper()} "
                f"({best_moe[1].get('moe_eer', -1):.3f}%)")

    out(f"\n{'='*78}")

    # Save
    with open(report_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    out(f"\n  Report saved: {report_file}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir,
                            f"{args.dataset}_{args.eval_protocol}")
    os.makedirs(log_dir, exist_ok=True)

    # Step 1: Generate shared splits
    splits_path = generate_shared_splits(args)

    # Step 2: Run each method
    log_files = {}
    for method in args.methods:
        log_file, retcode = run_method(method, args, splits_path, log_dir)
        log_files[method] = (log_file, retcode)

    # Step 3: Collect results
    results = collect_results(args, log_files)

    # Step 4: Print and save comparison
    cs_suffix = f"_{args.closed_set_mode}" if args.eval_protocol == "closed_set" else ""
    report_file = os.path.join(
        args.output_dir,
        f"benchmark_{args.dataset}_{args.eval_protocol}{cs_suffix}_"
        f"seed{args.random_seed}.txt")
    print_comparison(args, results, report_file)


if __name__ == "__main__":
    main()
