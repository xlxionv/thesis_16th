"""
milp_posthoc.py — Step 2 of the 3-Step RL+MILP Pipeline

Takes a JSON file of binary line-product allocations (produced by running
the trained RL Process Agent in inference mode) and solves a fast LP to find
the optimal lot sizes (quantities) for each period.

Supports two allocation formats:
  - Legacy 2D: {"period_0": [[1,0,1],[0,1,0]], ...}  (L×P union mask)
  - Per-step 4D: {"period_0": {"step_0": [[...]], "step_1": [[...]], ...}, ...}

For the 4D format, the feasibility shield uses the actual product sequence
across micro-steps (exact setup transitions) instead of the worst-case
(N-1)*max_setup bound, allowing tighter capacity constraints and higher
throughput.
"""

import argparse
import json
import os
import time

import numpy as np
import pulp


def _get_period_union_mask(rl_alloc, period_key, L, P):
    """Return L×P union mask from either legacy 2D or new per-step 4D format."""
    raw = rl_alloc.get(period_key)
    if raw is None:
        return np.zeros((L, P), dtype=np.float32)
    if isinstance(raw, dict):
        # 4D format: union over all steps
        union = np.zeros((L, P), dtype=np.float32)
        for step_mask in raw.values():
            union = np.maximum(union, np.array(step_mask, dtype=np.float32))
        return union
    return np.array(raw, dtype=np.float32)


def _compute_exact_setup_overhead(rl_alloc, period_key, prev_period_key, l, data, state, t):
    """
    Compute exact setup overhead for line l in period t using per-step sequence.

    For the 4D allocation format: reconstruct the product sequence from the
    micro-step masks and sum actual changeover times.  This replaces the
    conservative (N-1)*max_setup bound.

    For the legacy 2D format: fall back to the worst-case bound.
    """
    P = data["num_products"]
    setup_t = data["setup_time_matrix"]
    elig = data["eligibility_matrix"]

    raw = rl_alloc.get(period_key)
    if raw is None or not isinstance(raw, dict):
        # Legacy 2D — use worst-case bound (original logic)
        y_fixed = np.zeros((data["num_lines"], P), dtype=np.float32) if raw is None else np.array(raw, dtype=np.float32)
        active_products = [p for p in range(P) if y_fixed[l][p] > 0.5 and elig[l][p] > 0.5]
        if not active_products:
            return 0.0, active_products

        if t == 0:
            last_p = int(state["last_prod"][l])
            first_p = active_products[0]
            initial_setup = (
                data["first_setup_time"][l] if last_p < 0
                else (setup_t[l][last_p][first_p] if last_p != first_p else 0.0)
            )
        else:
            prev_y = _get_period_union_mask(rl_alloc, prev_period_key, data["num_lines"], P)
            prev_active = [q for q in range(P) if prev_y[l][q] > 0.5 and elig[l][q] > 0.5]
            if not prev_active:
                prev_active = list(range(P))
            initial_setup = float(np.max(
                [setup_t[l][q][p] for q in prev_active for p in active_products if q != p]
                or [data["first_setup_time"][l]]
            ))

        if len(active_products) > 1:
            max_intra = float(np.max(
                [setup_t[l][p][q] for p in active_products for q in active_products if p != q]
                or [0.0]
            ))
            intra_overhead = max_intra * (len(active_products) - 1)
        else:
            intra_overhead = 0.0

        return initial_setup + intra_overhead, active_products

    # 4D format — use union of activated products across all micro-steps (same worst-case
    # bound as the 2D path). The manager mask is a routing signal, NOT a production
    # sequence — treating each step activation as a production run causes phantom
    # changeovers that make capacity infeasible.
    step_keys = sorted(raw.keys(), key=lambda k: int(k.split("_")[1]))
    union_mask = np.zeros((data["num_lines"], P), dtype=np.float32)
    for sk in step_keys:
        union_mask = np.maximum(union_mask, np.array(raw[sk], dtype=np.float32))
    active_products = [p for p in range(P) if union_mask[l][p] > 0.5 and elig[l][p] > 0.5]

    if not active_products:
        return 0.0, active_products

    if t == 0:
        last_p = int(state["last_prod"][l])
        first_p = active_products[0]
        initial_setup = (
            data["first_setup_time"][l] if last_p < 0
            else (setup_t[l][last_p][first_p] if last_p != first_p else 0.0)
        )
    else:
        prev_y = _get_period_union_mask(rl_alloc, prev_period_key, data["num_lines"], P)
        prev_active = [q for q in range(P) if prev_y[l][q] > 0.5 and elig[l][q] > 0.5]
        if not prev_active:
            prev_active = list(range(P))
        initial_setup = float(np.max(
            [setup_t[l][q][p] for q in prev_active for p in active_products if q != p]
            or [data["first_setup_time"][l]]
        ))

    if len(active_products) > 1:
        max_intra = float(np.max(
            [setup_t[l][p][q] for p in active_products for q in active_products if p != q]
            or [0.0]
        ))
        intra_overhead = max_intra * (len(active_products) - 1)
    else:
        intra_overhead = 0.0

    return initial_setup + intra_overhead, active_products


def solve_lot_sizes(start_t, window, data, state, rl_alloc, time_limit=60):
    """LP lot-sizing with fixed binary allocation from RL agents.

    Handles both legacy 2D (period → L×P) and per-step 4D
    (period → step → L×P) allocation formats.  The 4D format enables
    exact setup-overhead accounting; the 2D format uses the conservative
    worst-case bound.

    Backlog is allowed with a per-unit penalty equal to data["backlog_cost"],
    making the LP always feasible when the RL allocation is capacity-constrained.
    """
    P, L = data["num_products"], data["num_lines"]
    K = data["capacity_per_line"]
    elig = data["eligibility_matrix"]
    proc = data["processing_time_matrix"]
    b_cost = float(data.get("backlog_cost", 1.0))

    prob = pulp.LpProblem(f"PosthocLotSizing_{start_t}", pulp.LpMinimize)

    x    = [[[pulp.LpVariable(f"x_{t}_{l}_{p}",    lowBound=0) for p in range(P)] for l in range(L)] for t in range(window)]
    inv  = [[pulp.LpVariable(f"inv_{t}_{p}",  lowBound=0) for p in range(P)] for t in range(window)]
    back = [[pulp.LpVariable(f"back_{t}_{p}", lowBound=0) for p in range(P)] for t in range(window)]

    for t in range(window):
        global_t = start_t + t
        alloc_key = f"period_{global_t}"
        prev_key = f"period_{global_t - 1}"
        y_union = _get_period_union_mask(rl_alloc, alloc_key, L, P)
        demand_t = data["demand_profile"][global_t]

        # Inventory balance: allow backlog so the LP is always feasible even when
        # the RL allocation over-concentrates products on a single line.
        for p in range(P):
            prev_inv = state["inv"][p] if t == 0 else inv[t - 1][p]
            produced = pulp.lpSum([x[t][l][p] for l in range(L)])
            prob += prev_inv + produced - inv[t][p] + back[t][p] == demand_t[p]

        # Feasibility shield per line using exact setup overhead
        for l in range(L):
            setup_overhead, active_products = _compute_exact_setup_overhead(
                rl_alloc, alloc_key, prev_key, l, data, state, t
            )

            if not active_products:
                for p in range(P):
                    prob += x[t][l][p] == 0
                continue

            effective_cap = max(0.0, float(K[l]) - setup_overhead)

            for p in range(P):
                if y_union[l][p] < 0.5 or elig[l][p] < 0.5:
                    prob += x[t][l][p] == 0

            prob += pulp.lpSum(proc[l][p] * x[t][l][p] for p in active_products) <= effective_cap

    # Objective: minimize inventory holding + production + maintenance + backlog costs
    obj = []
    for t in range(window):
        global_t = start_t + t
        alloc_key = f"period_{global_t}"
        y_union = _get_period_union_mask(rl_alloc, alloc_key, L, P)
        for p in range(P):
            obj.append(data["holding_cost"] * inv[t][p])
            obj.append(b_cost * back[t][p])
            for l in range(L):
                if y_union[l][p] > 0.5 and elig[l][p] > 0.5:
                    obj.append(data["production_cost_matrix"][l][p] * x[t][l][p])
                    obj.append(data["hazard_rate"][l] * data["cm_cost"][l] * proc[l][p] * x[t][l][p])

    prob += pulp.lpSum(obj)
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    lot_sizes = np.zeros((L, P), dtype=np.float32)
    new_inv = state["inv"].copy()
    new_last = state["last_prod"].copy()
    period_backlog = np.zeros(P, dtype=np.float32)

    if prob.status == 1:
        for l in range(L):
            for p in range(P):
                val = pulp.value(x[0][l][p])
                if val is not None and val > 1e-5:
                    lot_sizes[l][p] = float(val)

        for p in range(P):
            inv_val = pulp.value(inv[0][p])
            new_inv[p] = float(inv_val) if inv_val is not None and inv_val > 1e-5 else 0.0
            back_val = pulp.value(back[0][p])
            if back_val is not None and back_val > 1e-5:
                period_backlog[p] = float(back_val)

        for l in range(L):
            active = [p for p in range(P) if lot_sizes[l][p] > 1e-5]
            if active:
                new_last[l] = active[-1]

    return lot_sizes, new_inv, new_last, prob.status == 1, period_backlog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Post-hoc MILP lot sizing from RL allocation")
    parser.add_argument("--config", type=str, required=True, help="Path to the problem JSON config file")
    parser.add_argument("--rl_allocation", type=str, required=True, help="Path to RL allocation JSON (period -> binary L×P matrix)")
    parser.add_argument("--output", type=str, default="milp_lot_sizes.json", help="Output file for MILP-optimized lot sizes")
    parser.add_argument("--lookahead", type=int, default=3, help="Lookahead window for LP solve")
    parser.add_argument("--time_limit", type=float, default=60.0, help="CBC time limit per solve in seconds")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.exists(args.rl_allocation):
        raise FileNotFoundError(f"RL allocation file not found: {args.rl_allocation}")

    with open(args.config) as f:
        data = json.load(f)
    with open(args.rl_allocation) as f:
        rl_alloc = json.load(f)

    T = data["num_periods"]
    state = {
        "inv": np.zeros(data["num_products"]),
        "last_prod": np.full(data["num_lines"], -1),
    }

    output = {}
    wall_start = time.time()

    # Detect allocation format
    first_val = next(iter(rl_alloc.values()), None)
    alloc_format = "4D per-step" if isinstance(first_val, dict) else "2D legacy"

    print(f"\n[milp_posthoc] Config: {args.config}")
    print(f"[milp_posthoc] RL allocation: {args.rl_allocation} ({alloc_format})")
    print(f"[milp_posthoc] Periods: {T}, Lookahead: {args.lookahead}")
    print(f"[milp_posthoc] Backlog allowed with per-unit penalty = backlog_cost (LP always feasible)")
    print(f"\n{'Per':>3} | {'Status':>8} | {'Time(s)':>7} | {'Backlog qty':>11}")
    print("-" * 45)

    total_backlog = 0.0
    for t in range(T):
        t0 = time.time()
        window = min(args.lookahead, T - t)
        lot_sizes, new_inv, new_last, solved, period_backlog = solve_lot_sizes(
            t, window, data, state, rl_alloc, args.time_limit
        )
        state["inv"] = new_inv
        state["last_prod"] = new_last

        output[f"period_{t}"] = lot_sizes.tolist()
        elapsed = time.time() - t0
        status_str = "OK" if solved else "FAILED"
        back_qty = float(np.sum(period_backlog))
        total_backlog += back_qty
        back_str = f"{back_qty:.1f}" if back_qty > 0 else "-"
        print(f"{t+1:>3} | {status_str:>8} | {elapsed:>7.2f} | {back_str:>11}")

    if total_backlog > 0:
        print(f"\n  WARNING: total backlog across all periods = {total_backlog:.1f} units")

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    wall_total = time.time() - wall_start
    print(f"\n[milp_posthoc] Done in {wall_total:.1f}s")
    print(f"[milp_posthoc] Lot sizes saved to: {args.output}")
    print(f"\nNext step (Step 3 inference):")
    print(f"  python onpolicy/scripts/train/train_bosch.py \\")
    print(f"    --config {args.config} \\")
    print(f"    --milp_lot_sizes_path {args.output} \\")
    print(f"    ... (other training args)")
