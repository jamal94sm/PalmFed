"""
run_benchmark.py — Fair comparison across all methods.

1. Generates shared splits ONCE
2. Runs each method sequentially with same splits
3. Prints per-round eval for each method
4. Prints final comparison table
5. Saves report to text file
"""

import os, sys, json, time, pickle, argparse, re
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
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--methods", nargs="*",
                   default=["proposed", "fedpalm", "psfed"])
    p.add_argument("--output_dir", default="./benchmark_results")
    p.add_argument("--dp_mode", default="ideal",
                   choices=["ideal", "predicted"])
    return p.parse_args()


def generate_shared_splits(args):
    cfg = get_config("proposed")
    cfg["dataset"] = args.dataset
    cfg["eval_protocol"] = args.eval_protocol
    cfg["closed_set_mode"] = args.closed_set_mode
    cfg["n_ids"] = args.n_ids
    cfg["random_seed"] = args.random_seed

    splits = get_federated_splits(cfg, seed=args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    splits_path = os.path.join(args.output_dir, "shared_splits.pkl")
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)

    client_data, gallery, probe, _, spectra = splits
    print(f"  Splits saved: {splits_path}")
    print(f"  Clients: {len(client_data)} | "
          f"Gallery: {len(gallery)} | Probe: {len(probe)}")
    return splits_path


def run_method(method, args, splits_path, log_dir):
    import subprocess
    script = {"proposed": "main.py", "fedpalm": "fedpalm.py",
              "psfed": "psfed.py"}[method]
    log_file = os.path.join(log_dir, f"{method}.log")

    cmd = [sys.executable, script,
           "--dataset", args.dataset,
           "--eval_protocol", args.eval_protocol,
           "--closed_set_mode", args.closed_set_mode,
           "--n_rounds", str(args.n_rounds),
           "--random_seed", str(args.random_seed),
           "--splits_path", splits_path,
           "--eval_every", str(args.eval_every)]

    if method == "proposed":
        cmd.extend(["--method", "proposed", "--dp_mode", args.dp_mode])

    print(f"\n{'─'*70}")
    print(f"  Running: {method.upper()}")
    print(f"  Log: {log_file}")
    print(f"{'─'*70}")

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    status = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    print(f"  {method.upper()}: {status} ({elapsed:.0f}s)")
    return log_file, proc.returncode


# ══════════════════════════════════════════════════════════════
#  LOG PARSING
# ══════════════════════════════════════════════════════════════

def extract_metric(line, prefix):
    """Extract float after 'prefix=' in a line."""
    m = re.search(rf'{prefix}=\s*([\d.]+)%?', line)
    return float(m.group(1)) if m else None


def parse_proposed_log(log_file):
    """Parse proposed method log → per-round results."""
    rounds = []
    lines = open(log_file).readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "GLOBAL TEST" in line:
            # Scan ahead for results
            rnd_data = {}
            # Find round number from earlier "Eval round N" line
            for j in range(max(0, i-20), i):
                m = re.search(r'Eval round (\d+)', lines[j])
                if m:
                    rnd_data["round"] = int(m.group(1))
                    break

            # Parse subsequent lines for Global, Avg Loc, MoE
            for k in range(i+1, min(i+30, len(lines))):
                l = lines[k].strip()
                if "Global" in l and "│" in l and "%" in l:
                    rnd_data["global_r1"] = extract_metric(l, "R1") or \
                        extract_metric(l, r"│\s*([\d.]+)%")
                    # Parse formatted table line
                    nums = re.findall(r'([\d.]+)%', l)
                    if len(nums) >= 2:
                        rnd_data["global_r1"] = float(nums[0])
                        rnd_data["global_eer"] = float(nums[1])
                elif "Avg Loc" in l and "%" in l:
                    nums = re.findall(r'([\d.]+)%', l)
                    if len(nums) >= 2:
                        rnd_data["local_r1"] = float(nums[0])
                        rnd_data["local_eer"] = float(nums[1])
                elif l.startswith("MoE") and "%" in l:
                    nums = re.findall(r'([\d.]+)%', l)
                    if len(nums) >= 2:
                        rnd_data["moe_r1"] = float(nums[0])
                        rnd_data["moe_eer"] = float(nums[1])

            if "round" in rnd_data and "global_r1" in rnd_data:
                rounds.append(rnd_data)
        i += 1

    # Fallback: parse JSON results
    if not rounds:
        cfg = get_config("proposed")
        results_dir = cfg["base_results_dir"].replace("{dataset}",
                                                       cfg.get("dataset", "casiams"))
        for dp in ["ideal", "predicted"]:
            jp = os.path.join(results_dir, f"results_*_{dp}.json")
            import glob
            for jf in glob.glob(jp):
                with open(jf) as f:
                    data = json.load(f)
                for h in data.get("history", []):
                    gt = h.get("global_test", {})
                    rounds.append({
                        "round": h["round"],
                        "global_r1": gt.get("global", {}).get("rank1", -1),
                        "global_eer": gt.get("global", {}).get("eer", -1),
                        "local_r1": gt.get("avg_local", {}).get("rank1", -1),
                        "local_eer": gt.get("avg_local", {}).get("eer", -1),
                        "moe_r1": gt.get("moe", {}).get("rank1", -1),
                        "moe_eer": gt.get("moe", {}).get("eer", -1),
                    })
    return rounds


def parse_baseline_log(log_file, method):
    """Parse fedpalm/psfed log → per-round results."""
    rounds = []
    lines = open(log_file).readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not re.match(r'\[\d{2}:\d{2}:\d{2}\] Round \d+', line):
            continue

        m = re.search(r'Round (\d+)', line)
        if not m:
            continue
        rnd = int(m.group(1))
        rnd_data = {"round": rnd}

        # Look ahead for metrics
        for k in range(i+1, min(i+10, len(lines))):
            l = lines[k].strip()
            if l.startswith("Global") and "EER=" in l:
                rnd_data["global_eer"] = extract_metric(l, "EER")
                rnd_data["global_r1"] = extract_metric(l, "Rank-1")
            elif l.startswith("Local avg") and "EER=" in l:
                rnd_data["local_eer"] = extract_metric(l, "EER")
                rnd_data["local_r1"] = extract_metric(l, "Rank-1")
            elif method == "fedpalm" and "Full FedPalm" in l:
                rnd_data["moe_eer"] = extract_metric(l, "EER")
                rnd_data["moe_r1"] = extract_metric(l, "Rank-1")

        if "global_eer" in rnd_data:
            rounds.append(rnd_data)

    return rounds


def parse_results(log_file, method):
    if method == "proposed":
        return parse_proposed_log(log_file)
    else:
        return parse_baseline_log(log_file, method)


# ══════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════

def format_val(v, fmt=".2f"):
    return f"{v:{fmt}}%" if v is not None and v >= 0 else "—"


def print_method_table(method, rounds, out):
    """Print per-round table for one method."""
    has_moe = any(r.get("moe_r1") is not None and r.get("moe_r1", -1) >= 0
                  for r in rounds)

    out(f"\n  {method.upper()} — Per-Round Results")
    if has_moe:
        out(f"  {'Rnd':>5} │ {'Global R1':>10} {'Global EER':>11} │ "
            f"{'Local R1':>9} {'Local EER':>10} │ "
            )
    else:
        out(f"  {'Rnd':>5} │ {'Global R1':>10} {'Global EER':>11} │ "
            f"{'Local R1':>9} {'Local EER':>10}")
    out(f"  {'─'*68}")

    for r in rounds:
        line = (f"  {r['round']:>5} │ "
                f"{format_val(r.get('global_r1')):>10} "
                f"{format_val(r.get('global_eer'), '.3f'):>11} │ "
                f"{format_val(r.get('local_r1')):>9} "
                f"{format_val(r.get('local_eer'), '.3f'):>10}")
        if has_moe:
            line += (f" │ {format_val(r.get('moe_r1')):>8} "
                     f"{format_val(r.get('moe_eer'), '.3f'):>9}")
        out(line)


def print_comparison(args, all_results, out):
    """Print final comparison table across all methods."""
    out(f"\n{'='*78}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    protocol_str = args.eval_protocol
    if args.eval_protocol == "closed_set":
        protocol_str += f" ({args.closed_set_mode})"

    out(f"  BENCHMARK COMPARISON — {timestamp}")
    out(f"  Dataset: {args.dataset.upper()} | Protocol: {protocol_str}")
    out(f"  Rounds: {args.n_rounds} | Seed: {args.random_seed}")
    out(f"{'='*78}")

    # Get final round for each method
    final = {}
    for method, rounds in all_results.items():
        if rounds:
            final[method] = rounds[-1]

    # Header
    out(f"\n  {'Method':>15} │ {'Global R1':>10} {'Global EER':>11} │ "
        f"{'Local R1':>9} {'Local EER':>10} │ "
        )
    out(f"  {'─'*78}")

    for method in ["proposed", "fedpalm", "psfed"]:
        if method not in final:
            out(f"  {method.upper():>15} │ {'FAILED':>10} {'':>11} │ "
                f"{'':>9} {'':>10} │ {'':>8} {'':>9}")
            continue

        r = final[method]
        label = method.upper()
        if method == "proposed":
            label = f"PROPOSED({args.dp_mode})"

        # MoE only for proposed and fedpalm
        moe_r1 = format_val(r.get("moe_r1"), ".2f") if method in ["proposed", "fedpalm"] else "—"
        moe_eer = format_val(r.get("moe_eer"), ".3f") if method in ["proposed", "fedpalm"] else "—"

        out(f"  {label:>15} │ "
            f"{format_val(r.get('global_r1')):>10} "
            f"{format_val(r.get('global_eer'), '.3f'):>11} │ "
            f"{format_val(r.get('local_r1')):>9} "
            f"{format_val(r.get('local_eer'), '.3f'):>10} │ "
            f"{moe_r1:>8} {moe_eer:>9}")

    out(f"  {'─'*78}")

    # Best per category
    if final:
        for cat, key in [("Global EER", "global_eer"),
                         ("Global R1", "global_r1"),
                         ("Local EER", "local_eer")]:
            valid = [(m, r) for m, r in final.items()
                     if r.get(key) is not None and r.get(key, -1) >= 0]
            if valid:
                if "EER" in cat:
                    best_m, best_r = min(valid, key=lambda x: x[1][key])
                else:
                    best_m, best_r = max(valid, key=lambda x: x[1][key])
                out(f"  Best {cat:>12}: {best_m.upper()} "
                    f"({best_r[key]:.3f}%)")

    out(f"\n{'='*78}")


def main():
    args = parse_args()
    log_dir = os.path.join(args.output_dir,
                            f"{args.dataset}_{args.eval_protocol}")
    os.makedirs(log_dir, exist_ok=True)

    report_lines = []
    def out(s=""):
        report_lines.append(s)
        print(s)

    out(f"\n{'='*70}")
    out(f"  Generating shared data splits")
    out(f"  Dataset: {args.dataset} | Protocol: {args.eval_protocol}")
    out(f"{'='*70}")

    splits_path = generate_shared_splits(args)

    all_results = {}
    log_files = {}

    for method in args.methods:
        log_file, retcode = run_method(method, args, splits_path, log_dir)
        log_files[method] = (log_file, retcode)

        if retcode == 0:
            rounds = parse_results(log_file, method)
            all_results[method] = rounds

            # Print per-method table
            if rounds:
                print_method_table(method, rounds, out)
            else:
                out(f"\n  {method.upper()}: No results parsed from log")
        else:
            all_results[method] = []
            out(f"\n  {method.upper()}: FAILED")

    # Final comparison
    print_comparison(args, all_results, out)

    # Save report
    cs_suffix = f"_{args.closed_set_mode}" if args.eval_protocol == "closed_set" else ""
    report_file = os.path.join(
        args.output_dir,
        f"benchmark_{args.dataset}_{args.eval_protocol}{cs_suffix}_"
        f"seed{args.random_seed}.txt")
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n  Report saved: {report_file}")


if __name__ == "__main__":
    main()
