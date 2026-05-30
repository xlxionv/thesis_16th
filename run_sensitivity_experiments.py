"""
Run one-factor-at-a-time sensitivity experiments.

Default target is the representative large benchmark 24_8_4. Pass additional
benchmarks as P_L_T strings if you want to repeat the same grid elsewhere.

Examples:
  python run_sensitivity_experiments.py --dry_run
  python run_sensitivity_experiments.py --groups B D entropy lr clip gamma --skip_rh2_compare
  python run_sensitivity_experiments.py --benchmarks 24_8_4 28_9_4 --groups B entropy

The D/PPO grids omit the default value because the untagged benchmark run is
the paired baseline. Experiment B keeps tau=1.0 because all B settings use the
gated PM policy, so tau=1.0 is the center point inside that factor.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_BENCHMARKS = [(24, 8, 4)]

def parse_benchmark(value):
    try:
        p, l, t = value.split("_")
        return int(p), int(l), int(t)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected benchmark as P_L_T, got {value!r}"
        ) from exc


def fmt_float_tag(value):
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    return text


def experiment_grid(groups):
    experiments = []

    if "B" in groups:
        for tau in [0.5, 0.75, 1.0]:
            experiments.append({
                "group": "B",
                "tag": f"sensB_gated_tau{fmt_float_tag(tau)}",
                "args": [
                    "--pm_gate_risk_threshold", str(tau),
                ],
            })

    if "D" in groups:
        for penalty in [10.0, 40.0]:
            experiments.append({
                "group": "D",
                "tag": f"sensD_kill{fmt_float_tag(penalty)}",
                "args": ["--kill_switch_penalty", str(penalty)],
            })

    if "entropy" in groups:
        for entropy in [0.001, 0.005, 0.02]:
            experiments.append({
                "group": "entropy",
                "tag": f"sensEnt_{fmt_float_tag(entropy)}",
                "args": ["--entropy_coef", str(entropy)],
            })

    if "lr" in groups:
        for scale in [0.5, 2.0]:
            experiments.append({
                "group": "lr",
                "tag": f"sensLR_x{fmt_float_tag(scale)}",
                "args": ["--lr_scale", str(scale)],
            })

    if "clip" in groups:
        for clip in [0.2, 0.5]:
            experiments.append({
                "group": "clip",
                "tag": f"sensClip_{fmt_float_tag(clip)}",
                "args": ["--clip_param", str(clip)],
            })

    if "gamma" in groups:
        for gamma in [0.95, 0.99]:
            experiments.append({
                "group": "gamma",
                "tag": f"sensGamma_{fmt_float_tag(gamma)}",
                "args": ["--gamma", str(gamma)],
            })

    return experiments


def build_command(benchmark, experiment, args):
    p, l, t = benchmark
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "run_benchmark_pipeline.py"),
        "--num_products", str(p),
        "--num_lines", str(l),
        "--num_periods", str(t),
        "--pm_action_mode", "gated",
        "--result_tag", experiment["tag"],
    ]
    if "--pm_gate_risk_threshold" not in experiment["args"]:
        cmd.extend(["--pm_gate_risk_threshold", "1.0"])
    cmd.extend(experiment["args"])

    if args.force_train:
        cmd.append("--force_train")
    if args.skip_train:
        cmd.append("--skip_train")
    if args.skip_compare:
        cmd.append("--skip_compare")
    if args.skip_rh2_compare:
        cmd.append("--skip_rh2_compare")
    if args.num_env_steps is not None:
        cmd.extend(["--num_env_steps", str(args.num_env_steps)])

    return cmd


def wait_for_result_tag(benchmarks, tag, poll_seconds):
    pending = set(benchmarks)
    while pending:
        complete = []
        for benchmark in pending:
            path = os.path.join(
                PROJECT_ROOT,
                "benchmark_results",
                result_dir_name(benchmark, tag),
                "comparison_100instances.json",
            )
            if os.path.exists(path):
                complete.append(benchmark)

        for benchmark in complete:
            pending.remove(benchmark)
            print(f"Found completed prerequisite: {result_dir_name(benchmark, tag)}")

        if pending:
            waiting = ", ".join(result_dir_name(b, tag) for b in sorted(pending))
            print(f"Waiting for prerequisite result(s): {waiting}")
            time.sleep(poll_seconds)


def result_dir_name(benchmark, tag=None):
    p, l, t = benchmark
    name = f"{p}_{l}_{t}"
    if tag:
        name = f"{name}_{tag}"
    return name


def load_results(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def paired_values(baseline, candidate, field):
    base_values = []
    cand_values = []
    for key in sorted(baseline):
        base = baseline.get(key, {})
        cand = candidate.get(key, {})
        if base.get(field) is None or cand.get(field) is None:
            continue
        base_values.append(float(base[field]))
        cand_values.append(float(cand[field]))
    return np.array(base_values), np.array(cand_values)


def summarize_sensitivity(benchmarks, experiments, output_name):
    for benchmark in benchmarks:
        base_dir = os.path.join(
            PROJECT_ROOT,
            "benchmark_results",
            result_dir_name(benchmark),
        )
        baseline_path = os.path.join(base_dir, "comparison_100instances.json")
        baseline = load_results(baseline_path)
        if baseline is None:
            print(f"Missing baseline result: {baseline_path}")
            continue

        rows = []
        for experiment in experiments:
            tag = experiment["tag"]
            result_dir = os.path.join(
                PROJECT_ROOT,
                "benchmark_results",
                result_dir_name(benchmark, tag),
            )
            path = os.path.join(result_dir, "comparison_100instances.json")
            candidate = load_results(path)
            if candidate is None:
                rows.append({
                    "benchmark": result_dir_name(benchmark),
                    "group": experiment["group"],
                    "tag": tag,
                    "status": "missing",
                    "result_file": path,
                })
                continue

            row = {
                "benchmark": result_dir_name(benchmark),
                "group": experiment["group"],
                "tag": tag,
                "status": "ok",
                "result_file": path,
            }

            for field in ["step1_2_3", "step1_only"]:
                base, cand = paired_values(baseline, candidate, field)
                prefix = field
                row[f"{prefix}_n_paired"] = int(len(base))
                if len(base):
                    diff = cand - base
                    pct = diff / base * 100.0
                    row[f"{prefix}_baseline_mean"] = float(np.mean(base))
                    row[f"{prefix}_candidate_mean"] = float(np.mean(cand))
                    row[f"{prefix}_mean_diff"] = float(np.mean(diff))
                    row[f"{prefix}_mean_pct_diff"] = float(np.mean(pct))
                    row[f"{prefix}_median_pct_diff"] = float(np.median(pct))
                    row[f"{prefix}_win_rate"] = float(np.mean(cand < base))
                else:
                    row[f"{prefix}_baseline_mean"] = None
                    row[f"{prefix}_candidate_mean"] = None
                    row[f"{prefix}_mean_diff"] = None
                    row[f"{prefix}_mean_pct_diff"] = None
                    row[f"{prefix}_median_pct_diff"] = None
                    row[f"{prefix}_win_rate"] = None

            rows.append(row)

        rows_ok = [
            row for row in rows
            if row.get("status") == "ok" and row.get("step1_2_3_n_paired", 0) > 0
        ]
        best_by_group = {}
        for group in sorted({row["group"] for row in rows_ok}):
            group_rows = [row for row in rows_ok if row["group"] == group]
            best_by_group[group] = min(
                group_rows,
                key=lambda row: row["step1_2_3_candidate_mean"],
            )

        summary = {
            "benchmark": result_dir_name(benchmark),
            "baseline_file": baseline_path,
            "selection_metric": "lowest step1_2_3_candidate_mean",
            "best_by_group": best_by_group,
            "rows": rows,
        }

        p, l, t = benchmark
        out_base = os.path.join(
            PROJECT_ROOT,
            "benchmark_results",
            f"{p}_{l}_{t}_{output_name}",
        )
        json_path = f"{out_base}.json"
        csv_path = f"{out_base}.csv"

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        columns = [
            "benchmark", "group", "tag", "status",
            "step1_2_3_n_paired",
            "step1_2_3_baseline_mean",
            "step1_2_3_candidate_mean",
            "step1_2_3_mean_diff",
            "step1_2_3_mean_pct_diff",
            "step1_2_3_median_pct_diff",
            "step1_2_3_win_rate",
            "step1_only_n_paired",
            "step1_only_baseline_mean",
            "step1_only_candidate_mean",
            "step1_only_mean_diff",
            "step1_only_mean_pct_diff",
            "step1_only_median_pct_diff",
            "step1_only_win_rate",
            "result_file",
        ]
        with open(csv_path, "w") as f:
            f.write(",".join(columns) + "\n")
            for row in rows:
                values = []
                for col in columns:
                    value = row.get(col)
                    if value is None:
                        value = ""
                    values.append(str(value))
                f.write(",".join(values) + "\n")

        print(f"Wrote sensitivity summary: {json_path}")
        print(f"Wrote sensitivity summary: {csv_path}")

        if best_by_group:
            print("\nBest settings by group:")
            for group, row in best_by_group.items():
                print(
                    f"  {group:<8} {row['tag']:<24} "
                    f"step1_2_3_mean={row['step1_2_3_candidate_mean']:.2f} "
                    f"diff={row['step1_2_3_mean_pct_diff']:+.2f}%"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=parse_benchmark,
        default=DEFAULT_BENCHMARKS,
        help="Benchmarks as P_L_T. Default: 24_8_4.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["B", "D", "entropy", "lr", "clip", "gamma"],
        default=["B", "D", "entropy", "lr", "clip", "gamma"],
        help="Sensitivity groups to run.",
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them.")
    parser.add_argument("--force_train", action="store_true",
                        help="Retrain even if a tagged model already exists.")
    parser.add_argument("--skip_train", action="store_true",
                        help="Use existing tagged models only.")
    parser.add_argument("--skip_compare", action="store_true",
                        help="Train only; do not run comparison.")
    parser.add_argument("--skip_rh2_compare", action="store_true",
                        help="Skip RH2 comparison for expensive sensitivity runs.")
    parser.add_argument("--num_env_steps", type=int, default=None,
                        help="Override training budget for every run.")
    parser.add_argument("--summarize_only", action="store_true",
                        help="Only write the sensitivity comparison summary.")
    parser.add_argument("--summary_output_name", type=str,
                        default="sensitivity_summary",
                        help="Output suffix for benchmark_results/{P}_{L}_{T}_*.json/csv.")
    parser.add_argument("--wait_for_result_tag", type=str, default=None,
                        help="Wait until this tagged comparison file exists before running.")
    parser.add_argument("--wait_poll_seconds", type=int, default=300,
                        help="Polling interval for --wait_for_result_tag.")
    args = parser.parse_args()

    experiments = experiment_grid(args.groups)
    if not experiments:
        print("No experiments selected.")
        return

    total = len(args.benchmarks) * len(experiments)
    print(f"Prepared {total} sensitivity runs.")

    if args.summarize_only:
        summarize_sensitivity(args.benchmarks, experiments, args.summary_output_name)
        return

    if args.wait_for_result_tag and not args.dry_run:
        wait_for_result_tag(
            args.benchmarks,
            args.wait_for_result_tag,
            args.wait_poll_seconds,
        )

    for benchmark in args.benchmarks:
        p, l, t = benchmark
        for idx, experiment in enumerate(experiments, 1):
            cmd = build_command(benchmark, experiment, args)
            print("\n" + "#" * 80)
            print(
                f"{p}_{l}_{t} [{idx}/{len(experiments)}] "
                f"group={experiment['group']} tag={experiment['tag']}"
            )
            print(" ".join(cmd))
            if args.dry_run:
                continue
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            if result.returncode != 0:
                raise SystemExit(result.returncode)

    if not args.dry_run:
        summarize_sensitivity(args.benchmarks, experiments, args.summary_output_name)


if __name__ == "__main__":
    main()
