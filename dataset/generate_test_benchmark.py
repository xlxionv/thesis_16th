# generate_test_benchmark.py
import json
import random
import os

def generate_test_instance(P, L, T):
    capacity_hours = 120.0
    data = {
        "num_products": P,
        "num_lines": L,
        "num_periods": T,
        "capacity_per_line": [capacity_hours] * L,
        "eligibility_matrix": [[1 for _ in range(P)] for _ in range(L)],
        "processing_time_matrix": [[random.uniform(11/60, 15/60) for _ in range(P)] for _ in range(L)],
        "production_cost_matrix": [[random.uniform(1.3, 1.6) for _ in range(P)] for _ in range(L)],
        "demand_profile": [[random.randint(0, 150) for _ in range(P)] for _ in range(T)],
        "hazard_rate": [random.choice([2.0, 3.0]) / capacity_hours for _ in range(L)],
        "cm_time": [random.uniform(7/60, 11/60) for _ in range(L)],
        "pm_time": [random.uniform(2.5, 3.5) for _ in range(L)],
        "pm_cost": [random.uniform(10.0, 20.0) for _ in range(L)],
        "cm_cost": [random.uniform(2.0, 3.0) for _ in range(L)],
        "holding_cost": 1.75,
        "backlog_cost": 5.25,
        "per_product_backlog_penalty": 12.25,
        "first_setup_time": [random.uniform(1.2, 2.4) for _ in range(L)],
        "first_setup_cost": [random.uniform(25.0, 30.0) for _ in range(L)],
        "setup_time_matrix": [],
        "setup_cost_matrix": []
    }
    for l in range(L):
        t_mat, c_mat = [], []
        for i in range(P):
            t_row, c_row = [], []
            for j in range(P):
                if i == j:
                    t_row.append(0.0)
                    c_row.append(0.0)
                else:
                    t_row.append(random.uniform(1.2, 2.4))
                    c_row.append(random.uniform(25.0, 30.0))
            t_mat.append(t_row)
            c_mat.append(c_row)
        data["setup_time_matrix"].append(t_mat)
        data["setup_cost_matrix"].append(c_mat)
    return data

if __name__ == "__main__":
    # Choose your locked-in size for the whole training experiment
    P = 13
    L = 4
    T = 4
    NUM_FILES = 100
    
    target_dir = "dataset/test_benchmark_13_4_4"
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Generating {NUM_FILES} fixed evaluation sets for P={P}, L={L}, T={T}...")
    
    for i in range(1, NUM_FILES + 1):
        filename = os.path.join(target_dir, f"test_P{P}_L{L}_T{T}_{i}.json")
        data = generate_test_instance(P, L, T)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
            
    print("Done! You can now pass these to --eval_configs")