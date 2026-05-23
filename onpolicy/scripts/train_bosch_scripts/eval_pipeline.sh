#!/bin/bash
# eval_pipeline.sh — Run Steps 2a, 2b, 3 across all 100 instances.
#
# Usage (after Step 1 training is done):
#   bash onpolicy/scripts/train_bosch_scripts/eval_pipeline.sh \
#        results/BOSCH/rmappo/step1_rl_full/run1/models
#
# Output:
#   eval_results/rl_allocation_<i>.json   — RL allocation per instance
#   eval_results/milp_lot_sizes_<i>.json  — MILP lot sizes per instance
#   eval_results/summary.csv              — final costs across all 100 instances

set -e
MODEL_DIR="${1:?Usage: $0 <model_dir>}"

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
OUT_DIR="eval_results"
SUMMARY="$OUT_DIR/summary.csv"

mkdir -p "$OUT_DIR"
echo "instance,backlog,inventory,maintenance,production,setup,total" > "$SUMMARY"

echo "=================================================="
echo "  3-Step Eval Pipeline — $NUM_PRODUCTS P / $NUM_LINES L / $NUM_PERIODS T"
echo "  Model: $MODEL_DIR"
echo "  Instances: $INSTANCE_DIR"
echo "=================================================="

for i in $(seq 1 100); do
    CONFIG="$INSTANCE_DIR/test_P${NUM_PRODUCTS}_L${NUM_LINES}_T${NUM_PERIODS}_${i}.json"
    ALLOC="$OUT_DIR/rl_allocation_${i}.json"
    LOTS="$OUT_DIR/milp_lot_sizes_${i}.json"
    LOG="$OUT_DIR/step3_instance_${i}.log"

    if [ ! -f "$CONFIG" ]; then
        echo "[SKIP] Instance $i not found: $CONFIG"
        continue
    fi

    echo -n "[Instance $i] Step 2a export..."
    python configs/bosch/rl_export_allocation.py \
        --config "$CONFIG" \
        --model_dir "$MODEL_DIR" \
        --output "$ALLOC" \
        --num_products $NUM_PRODUCTS \
        --num_lines $NUM_LINES \
        --num_periods $NUM_PERIODS \
        2>/dev/null
    echo -n " Step 2b MILP..."
    python configs/bosch/milp_posthoc.py \
        --config "$CONFIG" \
        --rl_allocation "$ALLOC" \
        --output "$LOTS" \
        --lookahead 3 \
        --time_limit 60 \
        2>/dev/null
    echo -n " Step 3 inference..."
    python onpolicy/scripts/train/train_bosch.py \
        --algorithm_name rmappo \
        --experiment_name step3_eval \
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
        >"$LOG" 2>/dev/null

    # Parse final eval costs from log
    LAST=$(grep "eval average episode costs:" "$LOG" | tail -1)
    BACKLOG=$(echo "$LAST" | grep -oP 'Backlog: \K[0-9.]+')
    INV=$(echo    "$LAST" | grep -oP 'Inventory: \K[0-9.]+')
    MAINT=$(echo  "$LAST" | grep -oP 'Maintenance: \K[0-9.]+')
    PROD=$(echo   "$LAST" | grep -oP 'Production: \K[0-9.]+')
    SETUP=$(echo  "$LAST" | grep -oP 'Setup: \K[0-9.]+')
    TOTAL=$(echo  "$LAST" | grep -oP 'TOTAL: \K[0-9.]+')
    echo "$i,$BACKLOG,$INV,$MAINT,$PROD,$SETUP,$TOTAL" >> "$SUMMARY"
    echo " TOTAL=$TOTAL"
done

echo ""
echo "=================================================="
echo "  SUMMARY (averages across 100 instances)"
python3 - <<'EOF'
import csv, sys
rows = list(csv.DictReader(open("eval_results/summary.csv")))
if not rows:
    print("No results found.")
    sys.exit()
cols = ["backlog","inventory","maintenance","production","setup","total"]
avgs = {c: sum(float(r[c]) for r in rows if r[c])/len(rows) for c in cols}
for c, v in avgs.items():
    print(f"  Avg {c.capitalize():15s}: {v:>10.2f}")
print(f"\n  Instances evaluated: {len(rows)}")
EOF
echo "=================================================="
echo "  Full results saved to: $SUMMARY"
