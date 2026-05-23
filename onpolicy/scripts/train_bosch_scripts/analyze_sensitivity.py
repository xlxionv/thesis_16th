#!/usr/bin/env python3
"""
Reads eval_per_instance.json from sensitivity training runs and prints
the "% of instances with lower total cost" table.

Usage (from project root):
    python onpolicy/scripts/train_bosch_scripts/analyze_sensitivity.py
    python onpolicy/scripts/train_bosch_scripts/analyze_sensitivity.py \
        --results_dir onpolicy/scripts/train_bosch_scripts/results/BOSCH/rmappo \
        --modes average mean_std p75 p90 worst
"""
import argparse
import json
import os


def load_costs(results_dir, mode):
    path = os.path.join(results_dir, f"sensitivity_{mode}", "run1", "models", "eval_per_instance.json")
    with open(path) as f:
        return json.load(f)  # {instance_filename: total_cost}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="results/BOSCH/rmappo",
        help="Path to the rmappo results directory",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["average", "mean_std", "p75", "p90", "worst"],
        help="Setup time modes to compare (must match experiment_name suffixes)",
    )
    args = parser.parse_args()

    all_costs = {}
    for mode in args.modes:
        try:
            all_costs[mode] = load_costs(args.results_dir, mode)
        except FileNotFoundError as e:
            print(f"Warning: could not load results for mode '{mode}': {e}")

    if not all_costs:
        print("No results loaded. Check --results_dir and run sensitivity_setup_time.sh first.")
        return

    # Use only instances present in all modes
    common = set.intersection(*(set(c.keys()) for c in all_costs.values()))
    if not common:
        print("No common instances across modes. Check that all runs used the same eval_configs.")
        return

    instances = sorted(common)
    wins = {mode: 0 for mode in all_costs}

    for inst in instances:
        costs = {mode: all_costs[mode][inst] for mode in all_costs}
        best_mode = min(costs, key=costs.get)
        wins[best_mode] += 1

    n = len(instances)
    print(f"\n{'Setting':<20}  {'% instances with lowest cost':>28}")
    print("-" * 52)
    for mode in args.modes:
        if mode not in wins:
            continue
        pct = 100.0 * wins[mode] / n
        print(f"{mode:<20}  {pct:>27.0f}%")
    print(f"\n(n = {n} instances evaluated per mode)")


if __name__ == "__main__":
    main()
