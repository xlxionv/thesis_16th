"""
Clears broken step1_2_3 entries and re-runs them with the fixed milp_posthoc.py.
Run after the main pipeline has finished.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from run_benchmark_pipeline import approach_step123, instance_path

BROKEN = [
    # (P, L, T, [instance indices to re-run])
    (12, 4, 4, [10]),
    (13, 4, 4, [1, 2, 3]),       # 4,5,6 were fine; 7-10 used fixed code
]

for P, L, T, indices in BROKEN:
    result_file = os.path.join("benchmark_results", f"{P}_{L}_{T}", "comparison_100instances.json")
    if not os.path.exists(result_file):
        print(f"{P}_{L}_{T}: result file not found, skipping")
        continue

    with open(result_file) as f:
        results = json.load(f)

    for i in indices:
        key = f"instance_{i}"
        cfg = instance_path(P, L, T, i)
        if not os.path.exists(cfg):
            print(f"{P}_{L}_{T} instance {i}: config not found, skipping")
            continue

        old_val = results.get(key, {}).get("step1_2_3")
        step1_val = results.get(key, {}).get("step1_only")
        print(f"\n[{P}_{L}_{T} instance {i}]  old step1_2_3={old_val:.1f}  step1_only={step1_val:.1f}" if old_val else f"\n[{P}_{L}_{T} instance {i}]  no existing value")

        # Clear and recompute
        if key not in results:
            results[key] = {}
        results[key]["step1_2_3"] = None
        results[key]["step1_2_3_time"] = None

        try:
            t0 = time.time()
            c = approach_step123(P, L, T, cfg, i)
            elapsed = time.time() - t0
            results[key]["step1_2_3"] = c
            results[key]["step1_2_3_time"] = elapsed
            print(f"  new step1_2_3={c:.1f}  ({elapsed:.1f}s)  diff vs step1_only={c - step1_val:+.1f}" if c and step1_val else f"  result={c}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[key]["step1_2_3"] = None

        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

print("\nDone.")
