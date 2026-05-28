"""
run_all_large_benchmarks.py — Run full 3-step pipeline for all 9 large benchmarks.

Benchmarks:
  23_8_4, 24_8_4, 25_8_4, 26_9_4, 27_9_4, 28_9_4, 29_10_4, 30_10_4, 31_10_4

All require training (no pre-trained models).
kill_switch and entropy_coef scale via get_kill_switch(P) / get_entropy_coef(P)
in run_benchmark_pipeline.py (P>=10: flat 20 per unit, entropy 0.01).

Usage:
  python run_all_large_benchmarks.py               # run everything
  python run_all_large_benchmarks.py --start 26_9_4  # resume from a specific size
"""

import argparse
import subprocess
import sys
import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

BENCHMARKS = [
    (23, 8, 4),
    (24, 8, 4),
    (25, 8, 4),
    (26, 9, 4),
    (27, 9, 4),
    (28, 9, 4),
    (29, 10, 4),
    (30, 10, 4),
    (31, 10, 4),
]


def run_benchmark(
    P,
    L,
    T,
    skip_train=False,
    skip_rh2_compare=False,
    pm_action_mode="normal",
    pm_gate_risk_threshold=1.0,
    result_tag=None,
    num_env_steps=None,
):
    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "run_benchmark_pipeline.py"),
        "--num_products", str(P),
        "--num_lines",    str(L),
        "--num_periods",  str(T),
    ]
    if skip_train:
        cmd.append("--skip_train")
    if skip_rh2_compare:
        cmd.append("--skip_rh2_compare")
    if pm_action_mode != "normal":
        cmd.extend(["--pm_action_mode", pm_action_mode])
        cmd.extend(["--pm_gate_risk_threshold", str(pm_gate_risk_threshold)])
    if result_tag:
        cmd.extend(["--result_tag", result_tag])
    if num_env_steps is not None:
        cmd.extend(["--num_env_steps", str(num_env_steps)])

    print(f"\n{'#'*60}")
    print(
        f"# Running benchmark P={P} L={L} T={T}  skip_train={skip_train}  "
        f"skip_rh2_compare={skip_rh2_compare}  pm_action_mode={pm_action_mode}  "
        f"result_tag={result_tag or 'none'}"
    )
    print(f"{'#'*60}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def print_summary(result_tag=None):
    print(f"\n{'='*100}")
    print(f"{'OVERALL SUMMARY — ALL LARGE BENCHMARKS':^100}")
    print(f"{'='*100}")
    print(f"{'Benchmark':<12} {'n':>4} {'RH2 Mean':>10} {'S1+2+3':>10} {'S1':>10} "
          f"{'S1+2+3/RH2':>12} {'S1/RH2':>8} {'T_RH2':>8} {'T_S123':>8} {'T_S1':>7}")
    print(f"{'-'*100}")

    for P, L, T in BENCHMARKS:
        # Note: If your output directory names include the prefix 'test_benchmark_', 
        # you can adjust the path below to: f"test_benchmark_{P}_{L}_{T}"
        key = f"{P}_{L}_{T}"
        if result_tag:
            key = f"{key}_{result_tag}"
        out = os.path.join(PROJECT_ROOT, "benchmark_results", key,
                           "comparison_100instances.json")
        if not os.path.exists(out):
            print(f"{P}_{L}_{T:<9} {'—':>4}")
            continue

        with open(out) as f:
            results = json.load(f)

        valid = {k: v for k, v in results.items()
                 if all(v.get(a) is not None for a in ["step1_only", "step1_2_3", "rh2"])}
        rl_valid = {k: v for k, v in results.items()
                    if all(v.get(a) is not None for a in ["step1_only", "step1_2_3"])}
        if not valid:
            if rl_valid:
                s1 = np.array([v["step1_only"] for v in rl_valid.values()])
                s123 = np.array([v["step1_2_3"] for v in rl_valid.values()])
                print(f"{str(P)+'_'+str(L)+'_'+str(T):<12} {len(rl_valid):>4} {'n/a':>10} "
                      f"{np.mean(s123):>10.1f} {np.mean(s1):>10.1f} "
                      f"{'n/a':>12} {'n/a':>8} {'n/a':>8} {'n/a':>8} {'n/a':>7}")
            else:
                print(f"{P}_{L}_{T:<9} no valid results")
            continue

        s1   = np.array([v["step1_only"] for v in valid.values()])
        s123 = np.array([v["step1_2_3"]  for v in valid.values()])
        rh2  = np.array([v["rh2"]        for v in valid.values()])

        t1   = np.array([v["step1_only_time"] for v in valid.values() if v.get("step1_only_time")])
        t123 = np.array([v["step1_2_3_time"]  for v in valid.values() if v.get("step1_2_3_time")])
        trh2 = np.array([v["rh2_time"]        for v in valid.values() if v.get("rh2_time")])

        gap123 = np.mean((s123 - rh2) / rh2) * 100
        gap1   = np.mean((s1   - rh2) / rh2) * 100

        fmt_t = lambda a: f"{np.mean(a):.0f}s" if len(a) else "n/a"
        print(f"{str(P)+'_'+str(L)+'_'+str(T):<12} {len(valid):>4} {np.mean(rh2):>10.1f} "
              f"{np.mean(s123):>10.1f} {np.mean(s1):>10.1f} "
              f"{gap123:>+11.1f}% {gap1:>+7.1f}% "
              f"{fmt_t(trh2):>8} {fmt_t(t123):>8} {fmt_t(t1):>7}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None,
                        help="Start from this benchmark (e.g. '26_9_4'). Skips earlier ones.")
    parser.add_argument("--skip_rh2_compare", action="store_true",
                        help="Skip RH2 comparison for each benchmark.")
    parser.add_argument("--pm_action_mode", type=str, default="normal",
                        choices=["normal", "no_pm", "gated"],
                        help="Controls PM action availability for machine agents.")
    parser.add_argument("--pm_gate_risk_threshold", type=float, default=1.0,
                        help="For pm_action_mode=gated, allow PM after work when hazard_rate * age reaches this threshold.")
    parser.add_argument("--result_tag", type=str, default=None,
                        help="Optional suffix for benchmark_results and W&B experiment names.")
    parser.add_argument("--num_env_steps", type=int, default=None,
                        help="Override the default training budget for each benchmark.")
    args = parser.parse_args()

    start_idx = 0
    if args.start:
        keys = [f"{P}_{L}_{T}" for P, L, T in BENCHMARKS]
        if args.start in keys:
            start_idx = keys.index(args.start)
        else:
            print(f"Unknown benchmark '{args.start}'. Valid: {keys}")
            sys.exit(1)

    for i, (P, L, T) in enumerate(BENCHMARKS):
        if i < start_idx:
            print(f"Skipping {P}_{L}_{T} (before --start)")
            continue

        success = run_benchmark(
            P,
            L,
            T,
            skip_train=False,
            skip_rh2_compare=args.skip_rh2_compare,
            pm_action_mode=args.pm_action_mode,
            pm_gate_risk_threshold=args.pm_gate_risk_threshold,
            result_tag=args.result_tag,
            num_env_steps=args.num_env_steps,
        )
        if not success:
            print(f"\nERROR: benchmark {P}_{L}_{T} failed. Stopping.")
            break

    print_summary(result_tag=args.result_tag)


if __name__ == "__main__":
    main()
