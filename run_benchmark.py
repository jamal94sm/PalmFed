"""
run_benchmark.py — Fair comparison across all methods.

1. Generates shared splits ONCE
2. Runs each method with live output streaming
3. After each method: prints summary table
4. After all methods: comparison table (avg of last 5 eval rounds)

All methods report every k rounds:
  - Global model EER/R1
  - Per-client local model EER/R1
  - Average local EER/R1
"""

import os, sys, json, time, pickle, argparse, re
from datetime import datetime

from configs import get_config
from datasets import get_federated_splits


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark all methods")
    p.add_argument("--dataset", default="casiams",
                   choices=["casiams", "xjtu", "xpalm"])
    p.add_argument("--eval_protocol", default="open_set",
                   choices=["open_set", "closed_set"])
    p.add_argument("--closed_set_mode", default="cross_spectrum",
                   choices=["cross_spectrum"])
    p.add_argument("--n_rounds", type=int, default=100)
    p.add_argument("--n_ids", type=int, default=200)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=10)

  
    p.add_argument("--methods", nargs="*",
                   default=["proposed", "fedpalm", "psfed",
                            "fedavg", "local", "centralized"])
    

  
    p.add_argument("--output_dir", default="./benchmark_results")
    p.add_argument("--dp_mode", default="ideal",
                   choices=["ideal", "predicted"])
    p.add_argument("--avg_last", type=int, default=5,
                   help="Average last N eval rounds for comparison")
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
    cs_mode = f"_{args.closed_set_mode}" if args.eval_protocol == "closed_set" else ""
    splits_path = os.path.join(
        args.output_dir,
        f"splits_{args.dataset}_{args.eval_protocol}{cs_mode}_seed{args.random_seed}.pkl")
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)

    client_data, gallery, probe, _, spectra = splits
    print(f"  Splits saved: {splits_path}")
    print(f"  Clients: {len(client_data)} | "
          f"Gallery: {len(gallery)} | Probe: {len(probe)}")
    return splits_path, [cd["spectrum"] for cd in client_data]


def run_method(method, args, splits_path, log_dir):
    import subprocess
    script = {"proposed": "main.py", "fedpalm": "fedpalm.py",
              "psfed": "psfed.py", "fedavg": "fedavg.py",
              "local": "local_only.py", "centralized": "centralized.py"}[method]
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

    print(f"\n{'━'*70}")
    print(f"  Running: {method.upper()}")
    print(f"{'━'*70}\n")

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        proc.wait()
    elapsed = time.time() - t0

    status = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    print(f"\n  {method.upper()}: {status} ({elapsed:.0f}s)")
    return log_file, proc.returncode


# ══════════════════════════════════════════════════════════════
#  LOG PARSING
# ══════════════════════════════════════════════════════════════

def extract_metric(line, prefix):
    m = re.search(rf'{prefix}=\s*([\d.]+)%?', line)
    return float(m.group(1)) if m else None


def parse_proposed_log(log_file):
    """Parse proposed method log for per-round global + per-client local."""
    rounds = []
    lines = open(log_file).readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "Eval round" in line:
            m = re.search(r'Eval round (\d+)', line)
            if not m:
                i += 1; continue
            rnd = int(m.group(1))
            entry = {"round": rnd, "per_client": []}

            # Scan ahead for eval section
            in_eval = False
            for k in range(i+1, min(i+60, len(lines))):
                l = lines[k].strip()

                # Start of eval block (matches both old and new format)
                if l.startswith("LOCAL EVAL") or l == "EVALUATION":
                    in_eval = True
                    continue

                if not in_eval:
                    continue

                # Per-client line: "iPhone/Flash │   99.54%     1.624%"
                if "│" in l and "%" in l and not any(
                        x in l for x in ["Client", "─", "Global",
                                          "Avg", "MoE", "Rnd",
                                          "LOCAL"]):
                    nums = re.findall(r'([\d.]+)%', l)
                    name = l.split("│")[0].strip()
                    if len(nums) >= 2 and name:
                        entry["per_client"].append({
                            "name": name, "r1": float(nums[0]),
                            "eer": float(nums[1]),
                        })

                # Avg Loc line
                if "Avg Loc" in l and "%" in l:
                    nums = re.findall(r'([\d.]+)%', l)
                    if len(nums) >= 2:
                        entry["local_r1"] = float(nums[0])
                        entry["local_eer"] = float(nums[1])

                # Global line
                if "Global" in l and "│" in l and "%" in l:
                    nums = re.findall(r'([\d.]+)%', l)
                    if len(nums) >= 2:
                        entry["global_r1"] = float(nums[0])
                        entry["global_eer"] = float(nums[1])
                    break  # Global is last, stop scanning

            if "global_r1" in entry:
                rounds.append(entry)
        i += 1
    return rounds


def parse_baseline_log(log_file, method):
    """Parse fedpalm/psfed log for per-round results."""
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
        entry = {"round": rnd, "per_client": []}

        for k in range(i+1, min(i+20, len(lines))):
            l = lines[k].strip()
            if l.startswith("Global") and "EER=" in l:
                entry["global_eer"] = extract_metric(l, "EER")
                entry["global_r1"] = extract_metric(l, "Rank-1")
            elif l.startswith("Local avg") and "EER=" in l:
                entry["local_eer"] = extract_metric(l, "EER")
                entry["local_r1"] = extract_metric(l, "Rank-1")
            elif "EER=" in l and "Rank-1=" in l:
                # Per-client line: "    460  EER=15.200%  Rank-1=85.00%"
                name = l.split("EER=")[0].strip()
                if name and not any(x in name for x in
                                     ["Global", "Local", "Full",
                                      "Short", "Long"]):
                    entry["per_client"].append({
                        "name": name,
                        "eer": extract_metric(l, "EER"),
                        "r1": extract_metric(l, "Rank-1"),
                    })

        if "global_eer" in entry:
            rounds.append(entry)
    return rounds


def parse_results(log_file, method):
    if method in ("proposed", "fedavg"):
        return parse_proposed_log(log_file)
    elif method == "local":
        return parse_local_log(log_file)
    elif method == "centralized":
        return parse_centralized_log(log_file)
    else:
        return parse_baseline_log(log_file, method)


def parse_local_log(log_file):
    """Parse local baseline — no global model, only per-client local."""
    rounds = []
    lines = open(log_file).readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "Eval round" in line:
            m = re.search(r'Eval round (\d+)', line)
            if not m:
                i += 1; continue
            rnd = int(m.group(1))
            entry = {"round": rnd, "per_client": []}

            for k in range(i+1, min(i+40, len(lines))):
                l = lines[k].strip()
                if "│" in l and "%" in l and not any(
                        x in l for x in ["Client", "─", "Global",
                                          "Avg", "LOCAL"]):
                    nums = re.findall(r'([\d.]+)%', l)
                    name = l.split("│")[0].strip()
                    if len(nums) >= 2 and name:
                        entry["per_client"].append({
                            "name": name, "r1": float(nums[0]),
                            "eer": float(nums[1]),
                        })
                if "Avg Loc" in l and "%" in l:
                    nums = re.findall(r'([\d.]+)%', l)
                    if len(nums) >= 2:
                        entry["local_r1"] = float(nums[0])
                        entry["local_eer"] = float(nums[1])
                    break

            if "local_r1" in entry:
                rounds.append(entry)
        i += 1
    return rounds


def parse_centralized_log(log_file):
    """Parse centralized baseline — only global model."""
    rounds = []
    lines = open(log_file).readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "Eval epoch" in line:
            m = re.search(r'Eval epoch (\d+)', line)
            if not m:
                i += 1; continue
            rnd = int(m.group(1))
            entry = {"round": rnd, "per_client": []}

            for k in range(i+1, min(i+10, len(lines))):
                l = lines[k].strip()
                if "Global" in l and "│" in l and "%" in l:
                    nums = re.findall(r'([\d.]+)%', l)
                    if len(nums) >= 2:
                        entry["global_r1"] = float(nums[0])
                        entry["global_eer"] = float(nums[1])
                    break

            if "global_r1" in entry:
                rounds.append(entry)
        i += 1
    return rounds


# ══════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════

def fv(v, fmt=".2f"):
    return f"{v:{fmt}}%" if v is not None and v >= 0 else "—"


def print_method_summary(method, rounds, client_names, out):
    """Print per-round summary table for one method."""
    out(f"\n  {'─'*70}")
    out(f"  {method.upper()} — Summary")
    out(f"  {'─'*70}")

    # Header
    hdr = f"  {'Rnd':>5} │ {'Global R1':>10} {'EER':>9} │"
    for cn in client_names:
        hdr += f" {cn[:5]:>5}"
    hdr += f" │ {'Avg R1':>7} {'EER':>9}"
    out(hdr)
    out(f"  {'─'*len(hdr)}")

    for r in rounds:
        line = (f"  {r['round']:>5} │ "
                f"{fv(r.get('global_r1')):>10} "
                f"{fv(r.get('global_eer'), '.3f'):>9} │")

        # Per-client R1 (compact)
        for j, cn in enumerate(client_names):
            if j < len(r.get("per_client", [])):
                pc = r["per_client"][j]
                line += f" {fv(pc.get('r1')):>5}"
            else:
                line += f" {'—':>5}"

        line += (f" │ {fv(r.get('local_r1')):>7} "
                 f"{fv(r.get('local_eer'), '.3f'):>9}")
        out(line)


def print_comparison(args, all_results, client_names, out):
    """Final comparison: avg of last N eval rounds."""
    n_avg = args.avg_last
    protocol_str = args.eval_protocol
    if args.eval_protocol == "closed_set":
        protocol_str += f" ({args.closed_set_mode})"

    out(f"\n{'═'*78}")
    out(f"  BENCHMARK COMPARISON — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"  Dataset: {args.dataset.upper()} | Protocol: {protocol_str}")
    out(f"  Rounds: {args.n_rounds} | Seed: {args.random_seed}")
    out(f"  Averaged over last {n_avg} eval rounds")
    out(f"{'═'*78}")

    # Compute averages
    method_avgs = {}
    for method, rounds in all_results.items():
        if not rounds:
            continue
        last_n = rounds[-n_avg:] if len(rounds) >= n_avg else rounds

        avg = {
            "global_r1": sum(r.get("global_r1", 0) for r in last_n) / len(last_n),
            "global_eer": sum(r.get("global_eer", 0) for r in last_n) / len(last_n),
            "local_r1": sum(r.get("local_r1", 0) for r in last_n) / len(last_n),
            "local_eer": sum(r.get("local_eer", 0) for r in last_n) / len(last_n),
            "per_client": [],
        }

        # Per-client averages
        n_clients = len(client_names)
        for ci in range(n_clients):
            pc_r1s = [r["per_client"][ci]["r1"]
                      for r in last_n
                      if ci < len(r.get("per_client", []))
                      and r["per_client"][ci].get("r1", -1) >= 0]
            pc_eers = [r["per_client"][ci]["eer"]
                       for r in last_n
                       if ci < len(r.get("per_client", []))
                       and r["per_client"][ci].get("eer", -1) >= 0]
            avg["per_client"].append({
                "r1": sum(pc_r1s) / len(pc_r1s) if pc_r1s else -1,
                "eer": sum(pc_eers) / len(pc_eers) if pc_eers else -1,
            })

        method_avgs[method] = avg

    # ── Global + Avg Local comparison ──
    out(f"\n  {'Method':>15} │ {'Global R1':>10} {'Global EER':>11} │ "
        f"{'Avg Loc R1':>11} {'Avg Loc EER':>12}")
    out(f"  {'─'*64}")

    for method in ["proposed", "fedpalm", "psfed", "fedavg", "local", "centralized"]:
        if method not in method_avgs:
            out(f"  {method.upper():>15} │ {'FAILED':>10} {'':>11} │ "
                f"{'':>11} {'':>12}")
            continue
        a = method_avgs[method]
        label = f"PROPOSED" if method == "proposed" else method.upper()
        out(f"  {label:>15} │ "
            f"{fv(a['global_r1']):>10} {fv(a['global_eer'], '.3f'):>11} │ "
            f"{fv(a['local_r1']):>11} {fv(a['local_eer'], '.3f'):>12}")

    # ── Per-client local comparison ──
    out(f"\n  Per-client Local R1 (avg last {n_avg} rounds):")
    hdr = f"  {'Method':>15} │"
    for cn in client_names:
        hdr += f" {cn:>8}"
    hdr += f" │ {'Avg':>8}"
    out(hdr)
    out(f"  {'─'*len(hdr)}")

    for method in ["proposed", "fedpalm", "psfed", "fedavg", "local", "centralized"]:
        if method not in method_avgs:
            continue
        a = method_avgs[method]
        label = f"PROPOSED" if method == "proposed" else method.upper()
        line = f"  {label:>15} │"
        for ci in range(len(client_names)):
            if ci < len(a["per_client"]):
                line += f" {fv(a['per_client'][ci]['r1']):>8}"
            else:
                line += f" {'—':>8}"
        line += f" │ {fv(a['local_r1']):>8}"
        out(line)

    out(f"\n  Per-client Local EER (avg last {n_avg} rounds):")
    hdr = f"  {'Method':>15} │"
    for cn in client_names:
        hdr += f" {cn:>8}"
    hdr += f" │ {'Avg':>8}"
    out(hdr)
    out(f"  {'─'*len(hdr)}")

    for method in ["proposed", "fedpalm", "psfed", "fedavg", "local", "centralized"]:
        if method not in method_avgs:
            continue
        a = method_avgs[method]
        label = f"PROPOSED" if method == "proposed" else method.upper()
        line = f"  {label:>15} │"
        for ci in range(len(client_names)):
            if ci < len(a["per_client"]):
                line += f" {fv(a['per_client'][ci]['eer'], '.3f'):>8}"
            else:
                line += f" {'—':>8}"
        line += f" │ {fv(a['local_eer'], '.3f'):>8}"
        out(line)

    # Best per category
    out(f"\n  Winners:")
    if method_avgs:
        valid = [(m, a) for m, a in method_avgs.items()]
        best_geer = min(valid, key=lambda x: x[1]["global_eer"])
        best_gr1 = max(valid, key=lambda x: x[1]["global_r1"])
        best_leer = min(valid, key=lambda x: x[1]["local_eer"])
        best_lr1 = max(valid, key=lambda x: x[1]["local_r1"])
        out(f"    Best Global EER:    {best_geer[0].upper()} "
            f"({best_geer[1]['global_eer']:.3f}%)")
        out(f"    Best Global R1:     {best_gr1[0].upper()} "
            f"({best_gr1[1]['global_r1']:.2f}%)")
        out(f"    Best Avg Local EER: {best_leer[0].upper()} "
            f"({best_leer[1]['local_eer']:.3f}%)")
        out(f"    Best Avg Local R1:  {best_lr1[0].upper()} "
            f"({best_lr1[1]['local_r1']:.2f}%)")

    out(f"\n{'═'*78}")


def main():
    args = parse_args()
    log_dir = os.path.join(args.output_dir,
                            f"{args.dataset}_{args.eval_protocol}")
    os.makedirs(log_dir, exist_ok=True)

    report_lines = []
    def out(s=""):
        report_lines.append(s)
        print(s)

    out(f"\n{'═'*70}")
    out(f"  Generating shared data splits")
    out(f"  Dataset: {args.dataset} | Protocol: {args.eval_protocol}")
    out(f"{'═'*70}")

    splits_path, client_names = generate_shared_splits(args)

    all_results = {}

    for method in args.methods:
        log_file, retcode = run_method(method, args, splits_path, log_dir)

        if retcode == 0:
            rounds = parse_results(log_file, method)
            all_results[method] = rounds
            if rounds:
                print_method_summary(method, rounds, client_names, out)
            else:
                out(f"\n  {method.upper()}: No results parsed")
        else:
            all_results[method] = []
            out(f"\n  {method.upper()}: FAILED")

    # Final comparison
    print_comparison(args, all_results, client_names, out)

    # Save report
    cs_suffix = f"_{args.closed_set_mode}" if args.eval_protocol == "closed_set" else ""
    report_file = os.path.join(
        args.output_dir,
        f"benchmark_{args.dataset}_{args.eval_protocol}{cs_suffix}_"
        f"seed{args.random_seed}.txt")
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines) + "\n")
        # Append full logs
        for method in args.methods:
            lf = os.path.join(log_dir, f"{method}.log")
            if os.path.exists(lf):
                f.write(f"\n{'═'*70}\n  {method.upper()} — Full Log\n{'═'*70}\n")
                f.write(open(lf).read())
    out(f"\n  Report saved: {report_file}")


if __name__ == "__main__":
    main()
