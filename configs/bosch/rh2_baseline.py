"""
rh2_baseline.py – RH2 Rolling Horizon MILP Baseline for Bosch SD-CLSP

This script implements the Rolling Horizon 2 (RH2) heuristic with full
intra-period sequencing (MTZ routing), a Detailed Schedule Analyzer,
and exports Expert Data for RL Behavioral Cloning.
"""

import json
import sys
import time
import numpy as np
import pulp
import os

# ==============================================================================
# 1. SHARED COST EVALUATOR (Matches BoschEnv._end_period exactly)
# ==============================================================================
def compute_period_cost(
    produced, inventory_in, backlog_in, demand, setup_costs, pm_costs, 
    cm_costs, prod_cost_mat, h_cost, b_cost, backlog_penalty
):
    P = len(demand)
    total_produced = np.sum(produced, axis=0)

    new_inventory = np.zeros(P)
    new_backlog = np.zeros(P)
    inv_cost = 0.0
    backlog_cost = 0.0

    for p in range(P):
        supply = inventory_in[p] + total_produced[p]
        required = demand[p] + backlog_in[p]
        
        if supply >= required:
            new_inventory[p] = supply - required
            new_backlog[p] = 0.0
        else:
            new_inventory[p] = 0.0
            new_backlog[p] = required - supply
            
        inv_cost += h_cost * new_inventory[p]
        backlog_cost += b_cost * new_backlog[p]
        if new_backlog[p] > 0.0:
            pen = backlog_penalty[p] if isinstance(backlog_penalty, list) else backlog_penalty
            if new_backlog[p] > 0.001: backlog_cost += pen

    prod_cost = np.sum(produced * prod_cost_mat)
    setup_cost = np.sum(setup_costs)
    pm_cost = np.sum(pm_costs)
    cm_cost = np.sum(cm_costs)
    
    total = inv_cost + backlog_cost + prod_cost + setup_cost + pm_cost + cm_cost

    return {
        "inv_cost": inv_cost, "backlog_cost": backlog_cost, 
        "prod_cost": prod_cost, "setup_cost": setup_cost, 
        "pm_cost": pm_cost, "cm_cost": cm_cost, "total": total,
        "new_inventory": new_inventory, "new_backlog": new_backlog
    }

# ==============================================================================
# 2. THE MILP WINDOW SOLVER (with MTZ Intra-period Sequencing)
# ==============================================================================
def solve_lookahead_window(start_t, window, data, state, time_limit=60):
    P, L = data["num_products"], data["num_lines"]
    K = data["capacity_per_line"]
    elig = data["eligibility_matrix"]
    proc = data["processing_time_matrix"]
    setup_t = data["setup_time_matrix"]
    setup_c = data["setup_cost_matrix"]
    
    M_qty = sum(sum(d) for d in data["demand_profile"]) + sum(state["inv"]) + 1000
    prob = pulp.LpProblem(f"RH2_Window_{start_t}", pulp.LpMinimize)

    # Variables
    x = [[[pulp.LpVariable(f"x_{t}_{l}_{p}", lowBound=0) for p in range(P)] for l in range(L)] for t in range(window)]
    y = [[[pulp.LpVariable(f"y_{t}_{l}_{p}", cat="Binary") for p in range(P)] for l in range(L)] for t in range(window)]
    is_active = [[pulp.LpVariable(f"active_{t}_{l}", cat="Binary") for l in range(L)] for t in range(window)]
    first = [[[pulp.LpVariable(f"first_{t}_{l}_{p}", cat="Binary") for p in range(P)] for l in range(L)] for t in range(window)]
    last = [[[pulp.LpVariable(f"last_{t}_{l}_{p}", cat="Binary") for p in range(P)] for l in range(L)] for t in range(window)]
    z = [[[[pulp.LpVariable(f"z_{t}_{l}_{p}_{q}", cat="Binary") for q in range(P)] for p in range(P)] for l in range(L)] for t in range(window)]
    SO = [[[pulp.LpVariable(f"SO_{t}_{l}_{p}", lowBound=0) for p in range(P)] for l in range(L)] for t in range(window)]
    CO = [[[pulp.LpVariable(f"CO_{t}_{l}_{p}", lowBound=0) for p in range(P)] for l in range(L)] for t in range(window)]
    w = [[pulp.LpVariable(f"w_{t}_{l}", cat="Binary") for l in range(L)] for t in range(window)]
    inv = [[pulp.LpVariable(f"inv_{t}_{p}", lowBound=0) for p in range(P)] for t in range(window)]
    back = [[pulp.LpVariable(f"back_{t}_{p}", lowBound=0) for p in range(P)] for t in range(window)]
    is_backlog = [[pulp.LpVariable(f"isb_{t}_{p}", cat="Binary") for p in range(P)] for t in range(window)]

    # Constraints
    for t in range(window):
        global_t = start_t + t
        
        # 1. Inventory Balance
        for p in range(P):
            prev_inv = state["inv"][p] if t == 0 else inv[t-1][p]
            prev_back = state["back"][p] if t == 0 else back[t-1][p]
            produced = pulp.lpSum([x[t][l][p] for l in range(L)])
            prob += (prev_inv + produced - inv[t][p] + back[t][p] == data["demand_profile"][global_t][p] + prev_back)
            prob += back[t][p] <= M_qty * is_backlog[t][p]

        # 2. Line Logic & Routing
        for l in range(L):
            prob += pulp.lpSum(y[t][l][p] for p in range(P)) <= P * is_active[t][l]
            prob += pulp.lpSum(first[t][l][p] for p in range(P)) == is_active[t][l]
            prob += pulp.lpSum(last[t][l][p] for p in range(P)) == is_active[t][l]

            for p in range(P):
                if elig[l][p] < 0.5:
                    prob += y[t][l][p] == 0
                    prob += x[t][l][p] == 0
                else:
                    prob += x[t][l][p] <= M_qty * y[t][l][p]
                
                prob += z[t][l][p][p] == 0
                prob += first[t][l][p] + pulp.lpSum(z[t][l][q][p] for q in range(P) if q != p) == y[t][l][p]
                prob += last[t][l][p] + pulp.lpSum(z[t][l][p][q] for q in range(P) if q != p) == y[t][l][p]

                runtime = proc[l][p] * x[t][l][p]
                expected_cm = data["hazard_rate"][l] * data["cm_time"][l] * runtime
                
                prob += CO[t][l][p] == SO[t][l][p] + runtime + expected_cm
                prob += CO[t][l][p] <= K[l]
                prob += SO[t][l][p] <= K[l] * y[t][l][p]

                for q in range(P):
                    if p != q and elig[l][p] > 0.5 and elig[l][q] > 0.5:
                        prob += SO[t][l][q] >= CO[t][l][p] + setup_t[l][p][q] - K[l] * (1 - z[t][l][p][q])

            for p in range(P):
                if elig[l][p] > 0.5:
                    if t == 0:
                        last_p = int(state["last_prod"][l])
                        s_in = data["first_setup_time"][l] if last_p < 0 else (setup_t[l][last_p][p] if last_p != p else 0)
                    else:
                        s_in = sum(setup_t[l][q][p] for q in range(P))/P
                    prob += SO[t][l][p] >= (s_in + data["pm_time"][l] * w[t][l]) - K[l] * (1 - first[t][l][p])

    # Objective Function
    obj = []
    for t in range(window):
        for p in range(P):
            obj.append(data["holding_cost"] * inv[t][p])
            pen = data["per_product_backlog_penalty"][p] if isinstance(data["per_product_backlog_penalty"], list) else data["per_product_backlog_penalty"]
            obj.append(data["backlog_cost"] * back[t][p] + pen * is_backlog[t][p])
            for l in range(L):
                if elig[l][p] > 0.5:
                    obj.append(data["production_cost_matrix"][l][p] * x[t][l][p])
                    obj.append(data["hazard_rate"][l] * data["cm_cost"][l] * proc[l][p] * x[t][l][p])
                    
                    if t == 0:
                        last_p = int(state["last_prod"][l])
                        c_in = data["first_setup_cost"][l] if last_p < 0 else (setup_c[l][last_p][p] if last_p != p else 0)
                    else:
                        c_in = sum(setup_c[l][q][p] for q in range(P))/P
                    obj.append(c_in * first[t][l][p])
                    
                    for q in range(P):
                        if p != q and elig[l][q] > 0.5:
                            obj.append(setup_c[l][p][q] * z[t][l][p][q])
        for l in range(L):
            obj.append(data["pm_cost"][l] * w[t][l])

    prob += pulp.lpSum(obj)
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    # Extract T=0 Results
    produced = np.zeros((L, P))
    pm_done = np.zeros(L, dtype=bool)
    line_last = state["last_prod"].copy()
    setup_cost_actual = np.zeros(L)
    pm_cost_actual = np.zeros(L)
    cm_cost_actual = np.zeros(L)

    if prob.status == 1:
        for l in range(L):
            if pulp.value(w[0][l]) is not None and pulp.value(w[0][l]) > 0.5:
                pm_done[l] = True
                pm_cost_actual[l] = data["pm_cost"][l]
                
            active_p = []
            for p in range(P):
                val = pulp.value(x[0][l][p])
                if val is not None and val > 1e-5:
                    produced[l][p] = val
                    active_p.append(p)
                    cm_cost_actual[l] += data["hazard_rate"][l] * data["cm_cost"][l] * (proc[l][p] * val)
            
            if active_p:
                for p in range(P):
                    if pulp.value(first[0][l][p]) is not None and pulp.value(first[0][l][p]) > 0.5:
                        last_p = int(state["last_prod"][l])
                        if last_p < 0: setup_cost_actual[l] += data["first_setup_cost"][l]
                        elif last_p != p: setup_cost_actual[l] += setup_c[l][last_p][p]
                        
                    if pulp.value(last[0][l][p]) is not None and pulp.value(last[0][l][p]) > 0.5:
                        line_last[l] = p

                    for q in range(P):
                        if p != q and pulp.value(z[0][l][p][q]) is not None and pulp.value(z[0][l][p][q]) > 0.5:
                            setup_cost_actual[l] += setup_c[l][p][q]

    return produced, pm_done, line_last, setup_cost_actual, pm_cost_actual, cm_cost_actual

# ==============================================================================
# 3. SCHEDULE ANALYZER (Prints detailed human-readable decisions)
# ==============================================================================
def print_detailed_schedule(summary, data):
    print("\n" + "="*85)
    print(" 🏭 DETAILED RH2 SCHEDULE ANALYSIS (WHAT THE SOLVER DECIDED)")
    print("="*85)
    
    for res in summary:
        t = res["period"]
        print(f"\n[ PERIOD {t+1} ] - Demand for today: {data['demand_profile'][t]}")
        print("-" * 50)
        
        # 1. Machine Actions
        print("MACHINE ACTIONS:")
        for l in range(data["num_lines"]):
            produced = res["produced"][l]
            pm_done = res["pm_done"][l]
            active_products = np.where(produced > 0)[0]
            
            actions = []
            if pm_done:
                actions.append("🔧 Performed PM")
            
            if len(active_products) > 0:
                for p in active_products:
                    actions.append(f"📦 Prod {p} (Qty: {produced[p]:.0f})")
            else:
                actions.append("💤 IDLE")
                
            print(f"  Line {l}: {', '.join(actions)}")
            
        # 2. End of Day Status (Inventory & Backlog)
        print("\nEND OF DAY STATUS:")
        inv = res["costs"]["new_inventory"]
        back = res["costs"]["new_backlog"]
        
        has_activity = False
        for p in range(data["num_products"]):
            if inv[p] > 0 or back[p] > 0 or data["demand_profile"][t][p] > 0:
                print(f"  Prod {p} | Demand: {data['demand_profile'][t][p]:>4.0f} | "
                      f"Inventory: {inv[p]:>4.0f} | Backlog: {back[p]:>4.0f}")
                has_activity = True
                
        if not has_activity:
            print("  No active inventory or backlog.")
        print("="*85)

# ==============================================================================
# 4. MAIN EXECUTION (Rolling Horizon Loop)
# ==============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RH2 Baseline for SD-CLSP")
    parser.add_argument("--config", type=str, default="dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_1.json", help="Path to the JSON config file")
    parser.add_argument("--lookahead", type=int, default=3, help="Lookahead window in periods")
    parser.add_argument("--time_limit", type=float, default=1000.0, help="Time limit per solve in seconds")
    parser.add_argument("--output", type=str, default=None, help="CSV file to append a result row (instance,backlog,inventory,maintenance,production,setup,total)")
    parser.add_argument("--instance_id", type=int, default=None, help="Instance number to write in the CSV output row")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed schedule printout")

    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        # Try relative to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, args.config)
        if os.path.exists(alt_path):
            config_path = alt_path

    print(f"\n[RH2] Loading {config_path}...")
    with open(config_path, "r") as f:
        data = json.load(f)

    state = {
        "inv": np.zeros(data["num_products"]),
        "back": np.zeros(data["num_products"]),
        "last_prod": np.full(data["num_lines"], -1)
    }

    T = data["num_periods"]
    total_integrated_cost = 0.0
    history = {"inv":0, "back":0, "prod":0, "setup":0, "pm":0, "cm":0}
    period_summaries = []
    
    wall_start = time.time()
    
    print(f"\nStarting RH2 MILP Baseline (Periods: {T}, Lookahead: {args.lookahead})")
    print(f"{'Per':>3} | {'Total ($)':>9} | {'Inv':>7} | {'Backlog':>7} | {'Prod':>7} | {'Setup':>7} | {'PM/CM':>7} | {'Time(s)':>6}")
    print("-" * 75)

    for current_t in range(T):
        t0 = time.time()
        window = min(args.lookahead, T - current_t)
        
        produced, pm_done, new_last, setup_c, pm_c, cm_c = solve_lookahead_window(
            current_t, window, data, state, args.time_limit
        )
        
        costs = compute_period_cost(
            produced, state["inv"], state["back"], data["demand_profile"][current_t],
            setup_c, pm_c, cm_c, np.array(data["production_cost_matrix"]),
            data["holding_cost"], data["backlog_cost"], data["per_product_backlog_penalty"]
        )
        
        state["inv"] = costs["new_inventory"]
        state["back"] = costs["new_backlog"]
        state["last_prod"] = new_last
        total_integrated_cost += costs["total"]
        
        history["inv"] += costs["inv_cost"]
        history["back"] += costs["backlog_cost"]
        history["prod"] += costs["prod_cost"]
        history["setup"] += costs["setup_cost"]
        history["pm"] += costs["pm_cost"]
        history["cm"] += costs["cm_cost"]

        elapsed = time.time() - t0
        
        # Save for schedule printout and export
        period_summaries.append({
            "period": current_t,
            "produced": produced,
            "pm_done": pm_done,
            "costs": costs
        })
        
        print(f"{current_t+1:>3} | {costs['total']:>9.2f} | {costs['inv_cost']:>7.2f} | {costs['backlog_cost']:>7.2f} | {costs['prod_cost']:>7.2f} | {costs['setup_cost']:>7.2f} | {costs['pm_cost']+costs['cm_cost']:>7.2f} | {elapsed:>6.1f}")

    wall_total = time.time() - wall_start
    
    # 5. PRINT RESULTS
    if not args.quiet:
        print_detailed_schedule(period_summaries, data)

    print("\n" + "=" * 55)
    print(f" RH2 BASELINE FINAL RESULTS (Total Time: {wall_total:.1f}s)")
    print("=" * 55)
    print(f"  Inventory Cost    :  ${history['inv']:>10.2f}")
    print(f"  Backlog Cost      :  ${history['back']:>10.2f}")
    print(f"  Production Cost   :  ${history['prod']:>10.2f}")
    print(f"  Setup Cost        :  ${history['setup']:>10.2f}")
    print(f"  Maintenance Cost  :  ${history['pm']+history['cm']:>10.2f}")
    print("-" * 55)
    print(f"  TOTAL COST        :  ${total_integrated_cost:>10.2f}")
    print("=" * 55)

    # 6. WRITE CSV ROW (for batch benchmark comparison)
    if args.output:
        import csv, os as _os
        write_header = not _os.path.exists(args.output)
        with open(args.output, "a", newline="") as csvf:
            writer = csv.writer(csvf)
            if write_header:
                writer.writerow(["instance", "backlog", "inventory", "maintenance", "production", "setup", "total", "time_s"])
            writer.writerow([
                args.instance_id if args.instance_id is not None else _os.path.basename(args.config),
                f"{history['back']:.4f}",
                f"{history['inv']:.4f}",
                f"{history['pm'] + history['cm']:.4f}",
                f"{history['prod']:.4f}",
                f"{history['setup']:.4f}",
                f"{total_integrated_cost:.4f}",
                f"{wall_total:.2f}",
            ])

    # ==========================================================
    # 6. EXPORT EXPERT DATA FOR RL BEHAVIORAL CLONING
    # ==========================================================
    expert_dict = {}
    
    for res in period_summaries:
        t = res["period"]
        produced_matrix = res["produced"] # Shape: [L, P]
        
        # Convert continuous quantities into binary 1s and 0s
        # If RH2 produced more than 0.5 units, consider it "Activated" (1)
        binary_allocation = (produced_matrix > 0.5).astype(int).tolist()
        
        expert_dict[f"period_{t}"] = binary_allocation
        
    export_filename = "rh2_expert_data.json"
    with open(export_filename, "w") as f:
        json.dump(expert_dict, f, indent=2)
        
    print(f"\n✅ Exported Expert Action Data to {export_filename}")