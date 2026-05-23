#!/bin/bash
# benchmark_pipeline.sh — RH2 baseline vs 3-Step RL+MILP Pipeline
#
# For each instance, RH2 is launched in the background while Steps 2a→2b→3
# run in the foreground. Both finish before moving to the next instance, so
# the two methods run at the same time without requiring extra CPU cores.
#
# Usage:
#   bash onpolicy/scripts/train_bosch_scripts/benchmark_pipeline.sh \
#        <model_dir> [num_instances]
#
# Example:
#   bash onpolicy/scripts/train_bosch_scripts/benchmark_pipeline.sh \
#        results/BOSCH/rmappo/step1_rl/run1/models 10
#
# Output files (in benchmark_results/):
#   rh2_<i>.log             raw RH2 solver output
#   rl_allocation_<i>.json  RL allocation (Step 2a)
#   milp_lot_sizes_<i>.json MILP lot sizes (Step 2b)
#   step3_<i>.log           Step 3 inference log
#   rl_summary.csv          RL+MILP costs per instance
#   rh2_summary.csv         RH2 costs per instance
#   comparison.txt          printed side-by-side summary

set -e

MODEL_DIR="${1:?Usage: $0 <model_dir> [num_instances]}"
N="${2:-10}"

# Auto-detect hidden_size from wandb config.yaml in the model directory
HIDDEN_SIZE=$(python3 -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('$MODEL_DIR/config.yaml'))
    v = cfg.get('hidden_size', {})
    print(v.get('value', 64) if isinstance(v, dict) else v)
except:
    print(64)
" 2>/dev/null)

NUM_PRODUCTS=5
NUM_LINES=2
NUM_PERIODS=4
INSTANCE_DIR="dataset/Small/test_benchmark_5_2_4"
OUT_DIR="benchmark_results"
RL_CSV="$OUT_DIR/rl_summary.csv"
RH2_CSV="$OUT_DIR/rh2_summary.csv"

mkdir -p "$OUT_DIR"
echo "instance,backlog,inventory,maintenance,production,setup,total,time_s"  > "$RL_CSV"
echo "instance,backlog,inventory,setup,maintenance,production,total,time_s"  > "$RH2_CSV"

echo ""
echo "============================================================"
echo "  Benchmark: RH2 baseline  vs  3-Step RL+MILP Pipeline"
echo "  Problem : ${NUM_PRODUCTS}P / ${NUM_LINES}L / ${NUM_PERIODS}T"
echo "  Model   : $MODEL_DIR"
echo "  Instances: 1 – $N  (running both methods simultaneously)"
echo "============================================================"
printf "  %-6s %12s %12s %10s\n" "Inst" "RH2 Total" "3-Step Total" "Gap %"
echo "  ------------------------------------------------------------"

for i in $(seq 1 "$N"); do
    CONFIG="$INSTANCE_DIR/test_P${NUM_PRODUCTS}_L${NUM_LINES}_T${NUM_PERIODS}_${i}.json"
    if [ ! -f "$CONFIG" ]; then
        echo "  [SKIP] Instance $i not found: $CONFIG"
        continue
    fi

    ALLOC="$OUT_DIR/rl_allocation_${i}.json"
    LOTS="$OUT_DIR/milp_lot_sizes_${i}.json"
    STEP3_LOG="$OUT_DIR/step3_${i}.log"
    RH2_LOG="$OUT_DIR/rh2_${i}.log"

    # ── Launch RH2 in the background ──────────────────────────────────────
    RH2_START=$(python3 -c "import time; print(time.time())")
    python configs/bosch/rh2_baseline.py \
        --config "$CONFIG" \
        --lookahead 3 \
        --time_limit 60 \
        > "$RH2_LOG" 2>&1 &
    RH2_PID=$!

    # ── Step 2a → 2b → 3 (foreground, runs while RH2 is solving) ─────────
    RL_START=$(python3 -c "import time; print(time.time())")

    python configs/bosch/rl_export_allocation.py \
        --config "$CONFIG" \
        --model_dir "$MODEL_DIR" \
        --output "$ALLOC" \
        --num_products $NUM_PRODUCTS \
        --num_lines $NUM_LINES \
        --num_periods $NUM_PERIODS \
        2>/dev/null

    python configs/bosch/milp_posthoc.py \
        --config "$CONFIG" \
        --rl_allocation "$ALLOC" \
        --output "$LOTS" \
        --lookahead 3 \
        --time_limit 60 \
        2>/dev/null

    python onpolicy/scripts/train/train_bosch.py \
        --algorithm_name rmappo \
        --experiment_name benchmark_eval \
        --seed 1 \
        --num_products $NUM_PRODUCTS \
        --num_lines $NUM_LINES \
        --num_periods $NUM_PERIODS \
        --hidden_size $HIDDEN_SIZE \
        --obs_mode binary \
        --milp_lot_sizes_path "$LOTS" \
        --model_dir "$MODEL_DIR" \
        --use_eval \
        --n_eval_rollout_threads 1 \
        --eval_interval 1 \
        --eval_configs "$CONFIG" \
        --n_rollout_threads 1 \
        --num_env_steps 200 \
        >"$STEP3_LOG" 2>/dev/null

    RL_TIME=$(python3 -c "import time; print(f'{time.time()-${RL_START}:.1f}')")

    # ── Wait for RH2 to finish (may already be done) ──────────────────────
    wait $RH2_PID || true   # don't abort if RH2 failed on this instance
    RH2_TIME=$(python3 -c "import time; print(f'{time.time()-${RH2_START}:.1f}')")

    # ── Parse Step 3 results ───────────────────────────────────────────────
    LAST=$(grep "eval average episode costs:" "$STEP3_LOG" | tail -1)
    S3_BACK=$(  echo "$LAST" | perl -ne 'print "$1" if /Backlog:\s*([0-9.]+)/')
    S3_INV=$(   echo "$LAST" | perl -ne 'print "$1" if /Inventory:\s*([0-9.]+)/')
    S3_MAINT=$( echo "$LAST" | perl -ne 'print "$1" if /Maintenance:\s*([0-9.]+)/')
    S3_PROD=$(  echo "$LAST" | perl -ne 'print "$1" if /Production:\s*([0-9.]+)/')
    S3_SETUP=$( echo "$LAST" | perl -ne 'print "$1" if /Setup:\s*([0-9.]+)/')
    S3_TOTAL=$( echo "$LAST" | perl -ne 'print "$1" if /TOTAL:\s*([0-9.]+)/')
    echo "$i,${S3_BACK:-0},${S3_INV:-0},${S3_MAINT:-0},${S3_PROD:-0},${S3_SETUP:-0},${S3_TOTAL:-0},$RL_TIME" >> "$RL_CSV"

    # ── Parse RH2 results ──────────────────────────────────────────────────
    RH2_TOTAL=$( grep 'TOTAL COST'     "$RH2_LOG" | perl -ne 'print "$1" if /\$\s*([0-9.]+)/' || echo "0")
    RH2_INV=$(   grep 'Inventory Cost' "$RH2_LOG" | perl -ne 'print "$1" if /\$\s*([0-9.]+)/' || echo "0")
    RH2_BACK=$(  grep 'Backlog Cost'   "$RH2_LOG" | perl -ne 'print "$1" if /\$\s*([0-9.]+)/' || echo "0")
    RH2_SETUP=$( grep 'Setup Cost'     "$RH2_LOG" | perl -ne 'print "$1" if /\$\s*([0-9.]+)/' || echo "0")
    RH2_MAINT=$( grep 'Maintenance Cost' "$RH2_LOG" | perl -ne 'print "$1" if /\$\s*([0-9.]+)/' || echo "0")
    RH2_PROD=$(  grep 'Production Cost'  "$RH2_LOG" | perl -ne 'print "$1" if /\$\s*([0-9.]+)/' || echo "0")
    echo "$i,${RH2_BACK:-0},${RH2_INV:-0},${RH2_SETUP:-0},${RH2_MAINT:-0},${RH2_PROD:-0},${RH2_TOTAL:-0},$RH2_TIME" >> "$RH2_CSV"

    GAP=$(python3 -c "
r=float('${RH2_TOTAL:-0}')
s=float('${S3_TOTAL:-0}')
print(f'{(s-r)/r*100:+.1f}' if r>0 else 'N/A')
")
    printf "  %-6s %12s %12s   %s%%\n" "$i" "${RH2_TOTAL:-N/A}" "${S3_TOTAL:-N/A}" "$GAP"
done

# ── Side-by-side summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  FINAL COMPARISON (avg over $N instances)"
echo "============================================================"
python3 - "$RL_CSV" "$RH2_CSV" <<'PYEOF'
import csv, sys, os

def load(path):
    if not os.path.exists(path):
        return [], []
    rows = list(csv.DictReader(open(path)))
    return rows, [r for r in rows if r.get("total","0") not in ("","0")]

rl_rows,  rl_valid  = load(sys.argv[1])
rh2_rows, rh2_valid = load(sys.argv[2])

def avg(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key,"") not in ("","N/A")]
    return sum(vals)/len(vals) if vals else 0.0

print(f"\n  {'Component':<20} {'RL+MILP':>12} {'RH2':>12} {'Gap':>10}")
print("  " + "─"*58)
for label, rk in [("Backlog","backlog"),("Inventory","inventory"),
                  ("Maintenance","maintenance"),("Production","production"),
                  ("Setup","setup"),("TOTAL","total")]:
    rv = avg(rh2_valid, rk)
    sv = avg(rl_valid,  rk)
    gap = f"{(sv-rv)/rv*100:+.1f}%" if rv > 0 else "N/A"
    marker = "  ◀ better" if label=="TOTAL" and sv < rv else ""
    print(f"  {label:<20} {sv:>12.2f} {rv:>12.2f} {gap:>10}{marker}")

rl_t  = avg(rl_valid,  "time_s")
rh2_t = avg(rh2_valid, "time_s")
spd   = rh2_t/rl_t if rl_t > 0 else float("inf")
print(f"\n  {'Avg wall time':<20} {rl_t:>11.1f}s {rh2_t:>11.1f}s  {spd:>8.1f}x faster")

wins  = sum(1 for r,s in zip(rh2_valid, rl_valid)
            if float(s.get("total",0)) < float(r.get("total",0)))
print(f"\n  3-Step wins on {wins}/{len(rl_valid)} instances evaluated")
PYEOF

echo "============================================================"
echo "  Detailed results: $RL_CSV  |  $RH2_CSV"
echo "  Per-instance logs: $OUT_DIR/rh2_<i>.log  $OUT_DIR/step3_<i>.log"
