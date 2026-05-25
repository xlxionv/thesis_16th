"""
run_all_medium_benchmarks.py — Run full 3-step pipeline for all 9 medium benchmarks.

Benchmarks:
  14_5_4, 15_5_4, 16_5_4, 17_6_4, 18_6_4, 19_6_4, 20_7_4, 21_7_4, 22_7_4

All require training (no pre-trained models).
kill_switch and entropy_coef scale via get_kill_switch(P) / get_entropy_coef(P)
in run_benchmark_pipeline.py (P>=10: flat 20 per unit, entropy 0.01).

Usage:
  python run_all_medium_benchmarks.py               # run everything
  python run_all_medium_benchmarks.py --start 17_6_4  # resume from a specific size
"""

import argparse
import subprocess
import sys
import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

BENCHMARKS = [
    (14, 5, 4),
    (15, 5, 4),
    (16, 5, 4),
    (17, 6, 4),
    (18, 6, 4),
    (19, 6, 4),
    (20, 7, 4),
    (21, 7, 4),
    (22, 7, 4),
]


def run_benchmark(P, L, T, skip_train=False):
    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "run_benchmark_pipeline.py"),
        "--num_products", str(P),
        "--num_lines",    str(L),
        "--num_periods",  str(T),
    ]
    if skip_train:
        cmd.append("--skip_train")

    print(f"\n{'#'*60}")
    print(f"# Running benchmark P={P} L={L} T={T}  skip_train={skip_train}")
    print(f"{'#'*60}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def print_summary():
    print(f"\n{'='*100}")
    print(f"{'OVERALL SUMMARY — ALL MEDIUM BENCHMARKS':^100}")
    print(f"{'='*100}")
    print(f"{'Benchmark':<12} {'n':>4} {'RH2 Mean':>10} {'S1+2+3':>10} {'S1':>10} "
          f"{'S1+2+3/RH2':>12} {'S1/RH2':>8} {'T_RH2':>8} {'T_S123':>8} {'T_S1':>7}")
    print(f"{'-'*100}")

    for P, L, T in BENCHMARKS:
        out = os.path.join(PROJECT_ROOT, "benchmark_results", f"{P}_{L}_{T}",
                           "comparison_100instances.json")
        if not os.path.exists(out):
            print(f"{P}_{L}_{T:<9} {'—':>4}")
            continue

        with open(out) as f:
            results = json.load(f)

        valid = {k: v for k, v in results.items()
                 if all(v.get(a) is not None for a in ["step1_only", "step1_2_3", "rh2"])}
        if not valid:
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
                        help="Start from this benchmark (e.g. '17_6_4'). Skips earlier ones.")
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

        success = run_benchmark(P, L, T, skip_train=False)
        if not success:
            print(f"\nERROR: benchmark {P}_{L}_{T} failed. Stopping.")
            break

    print_summary()


if __name__ == "__main__":
    main()
