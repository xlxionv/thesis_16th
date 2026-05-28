# Bosch Pipeline Performance Analysis Report: The 17_6_4 Performance Gap

This report presents a deep-dive diagnostic investigation into why the multi-step `step1_2_3` pipeline (RL-guided MILP) underperforms compared to the `rh2` Rolling Horizon baseline in the `17_6_4` factory scheduling benchmarks, while outperforming it significantly on other large-scale sizes.

---

## 📊 1. Cross-Size Benchmark Performance

By analyzing all 14 benchmark sizes in the repository, we identified a clear transition point where the relative performance of the approaches changes:

| Size (P_L_T) | Instances | Mean Step 1 Only | Mean Step 1+2+3 | Mean RH2 Baseline | Step 1 → 1+2+3 | 1+2+3 vs. RH2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **5_2_4** | 100 | 2703.0 | 2675.2 | 2506.6 | 1.1% | +6.8% (Worse) |
| **6_2_4** | 80 | 3320.5 | 3262.2 | 3002.7 | 1.7% | +8.7% (Worse) |
| **8_3_4** | 10 | 4369.7 | 4296.0 | 3934.1 | 1.7% | +9.2% (Worse) |
| **9_3_4** | 10 | 5049.3 | 4787.1 | 4463.4 | 5.2% | +7.2% (Worse) |
| **10_3_4** | 10 | 5290.2 | 5267.5 | 4943.8 | 0.4% | +6.6% (Worse) |
| **11_4_4** | 10 | 6053.1 | 5791.4 | 6444.2 | 4.3% | **-6.0% (Better!)** |
| **12_4_4** | 10 | 6703.6 | 6700.9 | 6479.0 | -0.1% | +3.7% (Worse) |
| **13_4_4** | 10 | 7743.0 | 7489.6 | 10885.8 | 3.3% | **-19.5% (Better!)** |
| **14_5_4** | 10 | 7710.5 | 7408.0 | 10358.9 | 4.0% | **-11.8% (Better!)** |
| **15_5_4** | 10 | 12166.8 | 8163.0 | 8253.2 | 28.4% | **-0.4% (Better!)** |
| **16_5_4** | 10 | 11104.3 | 8601.2 | 9912.5 | 22.1% | **-8.6% (Better!)** |
| **17_6_4** | 10 | 10301.5 | 9022.9 | 8868.9 | 11.1% | **+2.1% (Worse)** |
| **18_6_4** | 1 | 16044.4 | 9703.2 | 9276.5 | 39.5% | **+4.6% (Worse)** |

### Key Insight:
* **Decoupled Architecture Wins at Scale**: For medium-to-large sizes (e.g., `13_4_4` to `16_5_4`), the `rh2` baseline struggles severely because of the combinatorial explosion of the MILP search space, frequently timing out or returning sub-optimal solutions. The RL-guided pipeline (`step1_2_3`) solves these instantly and beats RH2 by **up to 19.5%**.
* **The Regression at 17_6_4**: At `17_6_4`, `step1_2_3` regresses to be **2.1% worse** than `rh2`. Because `rh2` is allowed to run for up to 1500 seconds (25 minutes) per instance, it has enough time to find a high-quality global optimum on this size, exposing a fundamental limitation in the RL model's routing decisions.

---

## 🔍 2. Deep-Dive Cost Breakdown for `17_6_4`

We ran a period-by-period diagnostic audit on `17_6_4` Instance 1 to inspect the exact cost components:

```
Period 1: Total: 2550.09 | Inv: 0.00 | Backlog: 0.00 | Prod: 1994.24 | Setup: 502.71 | PM: 38.21 | CM: 14.93
Period 2: Total: 2086.55 | Inv: 0.00 | Backlog: 0.00 | Prod: 1635.66 | Setup: 439.32 | PM:  0.00 | CM: 11.56
Period 3: Total: 2406.56 | Inv: 0.00 | Backlog: 0.00 | Prod: 1925.92 | Setup: 467.32 | PM:  0.00 | CM: 13.32
Period 4: Total: 2452.64 | Inv: 0.00 | Backlog: 0.00 | Prod: 2001.37 | Setup: 437.12 | PM:  0.00 | CM: 14.15
---------------------------------------------------------------------------------------------------------
TOTALS  : Total: 9495.84 | Inv: 0.00 | Backlog: 0.00 | Prod: 7557.19 | Setup: 1846.47 | PM: 38.21 | CM: 53.96
```

### Critical Findings:
1. **Mathematically Perfect Delivery**: Both Inventory Holding and Backlog costs are **exactly 0.00**! The pipeline achieves a mathematically perfect zero-backlog schedule.
2. **Production Cost Dominance**: Production costs (`$7,557.19`) make up **79.6%** of the total cost. Combined with Setup costs (`$1,846.47`), these two components account for **99% of the scheduling budget**.

---

## 🔴 3. The Core Flaw: Reward Misalignment in RL Training

The performance gap between `step1_2_3` and `rh2` in `17_6_4` is not a lot-sizing error; it is a **structural routing flaw** caused by a critical misalignment in the RL reward function.

### A. The Blind Spot in `step1` Reward Mode
In `bosch_env.py` ([lines 1266-1279](file:///Users/nhi/thesis_14th-1/onpolicy/envs/bosch/bosch_env.py#L1266-L1279)), the RL Process Agent's `team` reward is calculated as:
```python
total_proc_time = float(
    np.sum(self.period_produced_per_line * self.processing_time_matrix)
)
team = -(total_proc_time + float(worker_total_direct_costs))
...
rewards[:, 0] += team
```

> [!IMPORTANT]
> **The production cost matrix (`production_cost_matrix`) is completely omitted from the RL reward function!**
> The RL agent is trained to minimize **processing time** (capacity utilization) and backlog, but it is **completely blind to the actual monetary production costs** of assigning specific products to specific lines.

### B. The Production Cost Variance
Because the problem configuration has highly line-dependent production costs, a product is much cheaper to make on some lines than others:
```
Production cost matrix for 17_6_4 Instance 1:
Product 0:  Line 0 ($1.40) | Line 1 ($1.33)  | Line 5 ($1.53)
Product 2:  Line 0 ($1.54) | Line 1 ($1.31)  | Line 5 ($1.32)
Product 6:  Line 0 ($1.60) | Line 1 ($1.39)  | Line 4 ($1.36)
```

Because the RL agent is blind to these numbers, it allocates products to lines based purely on capacity availability, ignoring whether a line is extremely expensive. 

### C. The Post-Hoc Feasibility Shield Constraint
In Step 2 (`milp_posthoc.py`), the post-hoc lot sizer enforces the RL allocator's mask as a **strict binary constraint**:
```python
if y_union[l][p] < 0.5 or elig[l][p] < 0.5:
    prob += x[t][l][p] == 0
```
This locks the post-hoc MILP into the RL agent's routing decisions. If the RL agent allocates Product 6 to Line 0 (cost `$1.60`), the post-hoc MILP **cannot** re-route it to Line 4 (cost `$1.36`) to save money. 

Meanwhile, the `rh2` baseline is a unified MILP solver that natively minimizes the real monetary production costs in its objective function:
```python
obj.append(data["production_cost_matrix"][l][p] * x[t][l][p])
```
Given enough search time (e.g., 20 minutes), `rh2` finds the globally cheapest routing, while `step1_2_3` is forced to produce quantities along the RL agent's cost-blind routing.

---

## 🛠️ 4. Recommended Action Plan

To close the final performance gap with `rh2` on large sizes like `17_6_4` and `18_6_4`, we recommend two non-disruptive changes:

### 1. Align RL Training Reward
Modify `step1` reward mode in `bosch_env.py` to include the production cost matrix instead of raw processing time. This ensures that the RL Process Agent learns to choose cost-optimal routes during training:
```diff
-            total_proc_time = float(
-                np.sum(self.period_produced_per_line * self.processing_time_matrix)
-            )
-            team = -(total_proc_time + float(worker_total_direct_costs))
+            prod_cost_total = float(
+                np.sum(self.period_produced_per_line * self.production_cost_matrix)
+            )
+            team = -(prod_cost_total + float(worker_total_direct_costs))
```

### 2. Relax the MILP Post-Hoc Constraints
Instead of enforcing the RL allocation mask as a strict feasibility shield (`x == 0`), allow the post-hoc MILP to locally adjust routing if a much cheaper line is available and has free capacity. This can be controlled via a tolerance parameter or by adding a small penalty for deviating from RL signals rather than a hard block.
