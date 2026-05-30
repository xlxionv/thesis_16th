#!/usr/bin/env python3
"""
Run RH2 for comparison entries where rh2 is missing/null, or for all entries
when --rerun_all is set.

Examples:
  python3 run_missing_rh2.py
  python3 run_missing_rh2.py --benchmarks 23_8_4 24_8_4_gated_pm
  python3 run_missing_rh2.py --benchmarks 24_8_4 --rerun_all
  python3 run_missing_rh2.py --dry_run
"""

import argparse
import json
import os
import re
import time

from run_benchmark_pipeline import PROJECT_ROOT, approach_rh2, instance_path


BENCHMARK_RE = re.compile(r"^(\d+)_(\d+)_(\d+)(?:_.+)?$")


def parse_benchmark_key(name):
    match = BENCHMARK_RE.match(name)
    if not match:
        return None
    return tuple(int(x) for x in match.groups())


def comparison_files(selected=None):
    root = os.path.join(PROJECT_ROOT, "benchmark_results")
    selected = set(selected or [])

    for name in sorted(os.listdir(root)):
        if selected and name not in selected:
            continue

        parsed = parse_benchmark_key(name)
        if parsed is None:
            continue

        path = os.path.join(root, name, "comparison_100instances.json")
        if os.path.exists(path):
            yield name, parsed, path


def instance_number(key):
    match = re.match(r"^instance_(\d+)$", key)
    return int(match.group(1)) if match else None


def run_file(name, dims, path, dry_run=False, limit=None, rerun_all=False):
    p, l, t = dims
    with open(path) as f:
        results = json.load(f)

    pending = []
    for key, row in sorted(results.items(), key=lambda item: instance_number(item[0]) or 10**9):
        idx = instance_number(key)
        if idx is None:
            continue
        if not isinstance(row, dict):
            continue
        if rerun_all or row.get("rh2") is None:
            pending.append((idx, key, row))

    if limit is not None:
        pending = pending[:limit]

    if not pending:
        print(f"{name}: no RH2 entries to run")
        return 0

    mode = "RH2 entries" if rerun_all else "missing RH2"
    print(f"{name}: {len(pending)} {mode}")
    if dry_run:
        for idx, _, _ in pending:
            print(f"  would run instance {idx}")
        return 0

    completed = 0
    for idx, key, row in pending:
        cfg = instance_path(p, l, t, idx)
        if not os.path.exists(cfg):
            print(f"  instance {idx}: missing config {cfg}")
            continue

        print(f"  instance {idx}: running RH2")
        t0 = time.time()
        try:
            cost = approach_rh2(cfg)
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  instance {idx}: RH2 failed after {elapsed:.1f}s: {exc}")
            row["rh2"] = None
            row["rh2_time"] = elapsed
        else:
            elapsed = time.time() - t0
            row["rh2"] = cost
            row["rh2_time"] = elapsed
            if cost is None:
                print(f"  instance {idx}: RH2 returned None ({elapsed:.1f}s)")
            else:
                print(f"  instance {idx}: RH2 {cost:.2f} ({elapsed:.1f}s)")
                completed += 1

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Benchmark result directories to update, e.g. 23_8_4 24_8_4_gated_pm.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--rerun_all",
        action="store_true",
        help="Run RH2 for every comparison entry, overwriting existing rh2/rh2_time values.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum missing RH2 instances to run per benchmark.",
    )
    args = parser.parse_args()

    total = 0
    seen = False
    for name, dims, path in comparison_files(args.benchmarks):
        seen = True
        total += run_file(
            name,
            dims,
            path,
            dry_run=args.dry_run,
            limit=args.limit,
            rerun_all=args.rerun_all,
        )

    if args.benchmarks and not seen:
        print("No matching comparison files found.")
    elif not args.dry_run:
        print(f"Completed RH2 for {total} instances.")


if __name__ == "__main__":
    main()
