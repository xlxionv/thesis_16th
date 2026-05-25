"""
run_benchmark_pipeline.py — Full 3-step pipeline for a single benchmark size.

Steps:
  1. Train RL model with best P=5 hyperparameters (scaled kill_switch)
  2. Export RL allocation (4D per-micro-step)
  3. MILP lot sizing (post-hoc)
  4. RL inference with MILP lot sizes
  5. RH2 baseline
  6. Save comparison results to benchmark_results/{P}_{L}_{T}/

Usage:
  # Train + compare
  python run_benchmark_pipeline.py --num_products 6 --num_lines 2 --num_periods 4

  # Skip training (model already exists)
  python run_benchmark_pipeline.py --num_products 6 --num_lines 2 --num_periods 4 --skip_train
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Best hyperparameters from P=5 sweep
BEST_HPARAMS = {
    "hidden_size":             128,
    "recurrent_N":             1,
    "lr":                      0.00026918605816868973,
    "critic_lr":               9.359933007344967e-05,
    "dense_production_reward": 2.1194776779870743,
    "entropy_coef":            0.0017196365215399027,
    "clip_param":              0.34949444139736674,
    "ppo_epoch":               6,
    "gae_lambda":              0.9624779421611346,
    "gamma":                   0.9626502158677256,
}
BASE_KILL_SWITCH = 5000   # for P=5; scaled proportionally for larger P
BASE_P = 5

# P>=10 uses proportional kill_switch (penalty × unmet_qty per period).
# Value of 20 per unit means 600 unmet units → -12,000 penalty, which clearly
# dominates the efficiency reward (~2,000–4,000/period) without being unstable.
LARGE_P_KILL_SWITCH  = 20     # per-unit backlog penalty for proportional reward mode
LARGE_P_ENTROPY_COEF = 0.01   # ~6× the P=5 optimized value; prevents premature collapse


def get_kill_switch(P):
    if P >= 10:
        return LARGE_P_KILL_SWITCH
    return int(BASE_KILL_SWITCH * P / BASE_P)


def get_entropy_coef(P):
    if P >= 10:
        return LARGE_P_ENTROPY_COEF
    return BEST_HPARAMS["entropy_coef"]


def get_num_steps(P):
    if P <= 13:   # Small
        return 500_000
    elif P <= 22: # Medium
        return 1_000_000
    else:         # Large
        return 3_000_000


def results_dir(P, L, T):
    return os.path.join(PROJECT_ROOT, "benchmark_results", f"{P}_{L}_{T}")


def model_dir(P, L, T):
    return os.path.join(results_dir(P, L, T), "model")


def dataset_size(P):
    if P <= 13:
        return "Small"
    elif P <= 22:
        return "Medium"
    else:
        return "Large"


def dataset_dir(P, L, T):
    return os.path.join(PROJECT_ROOT, "dataset", dataset_size(P), f"test_benchmark_{P}_{L}_{T}")


def instance_path(P, L, T, i):
    return os.path.join(dataset_dir(P, L, T), f"test_P{P}_L{L}_T{T}_{i}.json")


# ---------------------------------------------------------------------------
# Step 1: Train
# ---------------------------------------------------------------------------

def train(P, L, T):
    mdir = model_dir(P, L, T)
    os.makedirs(mdir, exist_ok=True)

    kill_switch = get_kill_switch(P)
    experiment_name = f"benchmark_{P}_{L}_{T}"
    print(f"\n{'='*60}")
    print(f"STEP 1: Training P={P} L={L} T={T}  kill_switch={kill_switch}")
    print(f"{'='*60}")

    eval_configs = [
        f"dataset/{dataset_size(P)}/test_benchmark_{P}_{L}_{T}/test_P{P}_L{L}_T{T}_{i}.json"
        for i in range(1, 101)
    ]

    train_args = [
        "--algorithm_name",         "rmappo",
        "--experiment_name",        experiment_name,
        "--seed",                   "1",
        "--num_products",           str(P),
        "--num_lines",              str(L),
        "--num_periods",            str(T),
        "--lookahead_days",         "4",
        "--reward_mode",            "step1",
        "--allocator_mode",         "jit",
        "--obs_mode",               "binary",
        "--dense_setup_penalty",    "0.0",
        "--max_actions_per_period", "8",
        "--n_rollout_threads",      "8",
        "--n_training_threads",     "1",
        "--num_env_steps",          str(get_num_steps(P)),
        "--log_interval",           "5",
        "--num_mini_batch",         "1",
        "--use_linear_lr_decay",
        "--use_eval",
        "--n_eval_rollout_threads", "5",
        "--eval_interval",          "50",
        "--eval_configs",           *eval_configs,
        "--hidden_size",            str(BEST_HPARAMS["hidden_size"]),
        "--recurrent_N",            str(BEST_HPARAMS["recurrent_N"]),
        "--lr",                     str(BEST_HPARAMS["lr"]),
        "--critic_lr",              str(BEST_HPARAMS["critic_lr"]),
        "--dense_production_reward",str(BEST_HPARAMS["dense_production_reward"]),
        "--entropy_coef",           str(get_entropy_coef(P)),
        "--clip_param",             str(BEST_HPARAMS["clip_param"]),
        "--ppo_epoch",              str(BEST_HPARAMS["ppo_epoch"]),
        "--gae_lambda",             str(BEST_HPARAMS["gae_lambda"]),
        "--gamma",                  str(BEST_HPARAMS["gamma"]),
        "--kill_switch_penalty",    str(kill_switch),
        "--use_wandb",
        "--user_name",              "tntvan-iac-instagram",
    ]

    from onpolicy.scripts.train.train_bosch import main as train_main
    train_main(train_args)

    # Copy saved model → benchmark_results/{P}_{L}_{T}/model/
    # When W&B is enabled, model saves to wandb run dir; otherwise to run1/models/.
    import shutil, glob

    base = os.path.join(PROJECT_ROOT, "results", "BOSCH", "rmappo", experiment_name)
    candidates = (
        glob.glob(os.path.join(base, "wandb", "run-*", "files", "actor_agent0.pt")) +
        glob.glob(os.path.join(base, "run*", "models", "actor_agent0.pt"))
    )
    if candidates:
        # Use the most recently modified one
        src_agent0 = max(candidates, key=os.path.getmtime)
        src_dir = os.path.dirname(src_agent0)
        for fname in ["actor_agent0.pt", "actor_agent1.pt"]:
            src = os.path.join(src_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(mdir, fname))

        # Copy config.yaml if present; otherwise write one from known hyperparams
        src_cfg = os.path.join(src_dir, "config.yaml")
        dst_cfg = os.path.join(mdir, "config.yaml")
        if os.path.exists(src_cfg):
            shutil.copy2(src_cfg, dst_cfg)
        else:
            import yaml
            cfg = {
                "hidden_size":             {"value": BEST_HPARAMS["hidden_size"]},
                "recurrent_N":             {"value": BEST_HPARAMS["recurrent_N"]},
                "reward_mode":             {"value": "step1"},
                "allocator_mode":          {"value": "jit"},
                "obs_mode":                {"value": "binary"},
                "max_actions_per_period":  {"value": 8},
            }
            with open(dst_cfg, "w") as f:
                yaml.dump(cfg, f)

        print(f"\n[Step 1] Training complete. Model copied to: {mdir}")
    else:
        print(f"\n[Step 1] WARNING: no model found under {base}")
        print(f"         Check the directory manually.")


# ---------------------------------------------------------------------------
# Step 2a: Export RL allocation
# ---------------------------------------------------------------------------

def export_allocation(P, L, T, instance_idx, config_path, alloc_path):
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "configs/bosch/rl_export_allocation.py"),
        "--config",        config_path,
        "--model_dir",     model_dir(P, L, T),
        "--output",        alloc_path,
        "--num_products",  str(P),
        "--num_lines",     str(L),
        "--num_periods",   str(T),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"  [Step2a EXPORT] FAILED instance {instance_idx}: {result.stderr[-300:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# Step 2b: MILP lot sizing
# ---------------------------------------------------------------------------

def milp_lot_sizes(P, L, T, instance_idx, config_path, alloc_path, milp_path):
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "configs/bosch/milp_posthoc.py"),
        "--config",       config_path,
        "--rl_allocation", alloc_path,
        "--output",       milp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"  [Step2b MILP] FAILED instance {instance_idx}: {result.stderr[-300:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# Step 3: RL inference with MILP lot sizes
# ---------------------------------------------------------------------------

def build_env_and_policies(P, L, T, config_path, milp_lot_sizes_path=None):
    import torch
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    from onpolicy.envs.bosch.bosch_env import BoschEnv
    from onpolicy.scripts.train.train_bosch import parse_args
    from onpolicy.config import get_config

    hs = BEST_HPARAMS["hidden_size"]
    rN = BEST_HPARAMS["recurrent_N"]

    dummy_argv = [
        "--num_products",          str(P),
        "--num_lines",             str(L),
        "--num_periods",           str(T),
        "--max_actions_per_period","8",
        "--lookahead_days",        "4",
        "--hidden_size",           str(hs),
        "--recurrent_N",           str(rN),
        "--reward_mode",           "step1",
        "--allocator_mode",        "jit",
        "--obs_mode",              "binary",
        "--algorithm_name",        "rmappo",
        "--experiment_name",       "inference",
        "--seed",                  "1",
    ]
    env_args = parse_args(dummy_argv, get_config())
    env_args.eval_config_dicts = [json.load(open(config_path))]
    env_args.milp_lot_sizes_path = milp_lot_sizes_path

    device = torch.device("cpu")
    env = BoschEnv(env_args, rank=0, is_eval=True)
    num_agents = env.num_agents

    manager_policy = R_MAPPOPolicy(
        env_args, env.observation_space[0], env.observation_space[0],
        env.action_space[0], device=device,
    )
    machine_policy = R_MAPPOPolicy(
        env_args, env.observation_space[1], env.observation_space[1],
        env.action_space[1], device=device,
    )

    mdir = model_dir(P, L, T)
    import torch
    manager_policy.actor.load_state_dict(
        torch.load(os.path.join(mdir, "actor_agent0.pt"), map_location=device))
    machine_policy.actor.load_state_dict(
        torch.load(os.path.join(mdir, "actor_agent1.pt"), map_location=device))
    manager_policy.actor.eval()
    machine_policy.actor.eval()

    return env, env_args, manager_policy, machine_policy, num_agents


def run_rl_episode(env, env_args, manager_policy, machine_policy, num_agents):
    import torch
    import numpy as np

    rN = env_args.recurrent_N
    hs = env_args.hidden_size
    rnn_states = [np.zeros((1, rN, hs), dtype=np.float32) for _ in range(num_agents)]
    masks = np.ones((1, 1), dtype=np.float32)

    obs_list = env.reset()
    available_actions = env._build_available_actions()
    done = False
    episode_total_cost = None

    while not done:
        actions_env = []
        for agent_id in range(num_agents):
            obs_np = np.array([obs_list[agent_id]], dtype=np.float32)
            avail_np = np.array([available_actions[agent_id]], dtype=np.float32)
            policy = manager_policy if agent_id == 0 else machine_policy
            with torch.no_grad():
                action, rnn_out = policy.act(obs_np, rnn_states[agent_id], masks,
                                             avail_np, deterministic=True)
            action_np = action.detach().cpu().numpy()
            rnn_states[agent_id] = rnn_out.detach().cpu().numpy()

            act_space = env.action_space[agent_id]
            if act_space.__class__.__name__ == "MultiDiscrete":
                action_env = None
                for i in range(act_space.shape):
                    oh = np.eye(act_space.high[i] + 1)[action_np[:, i]]
                    action_env = oh if action_env is None else np.concatenate([action_env, oh], axis=1)
                actions_env.append(action_env[0])
            else:
                actions_env.append(np.squeeze(np.eye(act_space.n)[action_np], 1)[0])

        obs_list, _, dones, infos = env.step(actions_env)
        available_actions = env._build_available_actions()
        done = all(dones)
        if "episode_total_cost" in infos[0]:
            episode_total_cost = infos[0]["episode_total_cost"]

    return episode_total_cost


def approach_step1_only(P, L, T, config_path):
    env, env_args, mgr, mch, n = build_env_and_policies(P, L, T, config_path)
    return run_rl_episode(env, env_args, mgr, mch, n)


def approach_step123(P, L, T, config_path, instance_idx):
    alloc_tmp = f"/tmp/rl_alloc_{P}_{L}_{T}_{instance_idx}.json"
    milp_tmp  = f"/tmp/milp_lots_{P}_{L}_{T}_{instance_idx}.json"

    if not export_allocation(P, L, T, instance_idx, config_path, alloc_tmp):
        return None
    if not milp_lot_sizes(P, L, T, instance_idx, config_path, alloc_tmp, milp_tmp):
        return None
    if not os.path.exists(milp_tmp):
        return None

    env, env_args, mgr, mch, n = build_env_and_policies(P, L, T, config_path, milp_tmp)
    cost = run_rl_episode(env, env_args, mgr, mch, n)

    for f in [alloc_tmp, milp_tmp]:
        try: os.remove(f)
        except: pass

    return cost


def approach_rh2(config_path):
    cmd = [sys.executable,
           os.path.join(PROJECT_ROOT, "configs/bosch/rh2_baseline.py"),
           "--config", config_path, "--quiet",
           "--time_limit", "300"]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=PROJECT_ROOT, timeout=1500)
    output = result.stdout + result.stderr
    match = re.search(r'TOTAL COST\s*:\s*\$\s*([\d,]+\.?\d*)', output)
    if match:
        return float(match.group(1).replace(',', ''))
    return None


# ---------------------------------------------------------------------------
# Compare: run all 3 approaches on 100 instances
# ---------------------------------------------------------------------------

N_COMPARE = 10  # number of instances to compare (RH2 is expensive for large P)


def run_comparison(P, L, T):
    rdir = results_dir(P, L, T)
    os.makedirs(rdir, exist_ok=True)
    output_file = os.path.join(rdir, "comparison_100instances.json")

    results = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results for {P}_{L}_{T}")

    for i in range(1, N_COMPARE + 1):
        key = f"instance_{i}"
        cfg = instance_path(P, L, T, i)
        if not os.path.exists(cfg):
            print(f"[{i}/100] SKIP — not found: {cfg}")
            continue

        if key in results and all(results[key].get(k) is not None
                                  for k in ["step1_only", "step1_2_3", "rh2"]):
            print(f"[{i}/100] Already complete, skipping")
            continue

        print(f"\n[{i}/100] {P}_{L}_{T} instance {i}")
        if key not in results:
            results[key] = {}

        if results[key].get("step1_only") is None:
            try:
                t0 = time.time()
                c = approach_step1_only(P, L, T, cfg)
                elapsed = time.time() - t0
                results[key]["step1_only"] = c
                results[key]["step1_only_time"] = elapsed
                print(f"  Step1 only:  {c:.2f}  ({elapsed:.1f}s)" if c else "  Step1 only: None")
            except Exception as e:
                print(f"  Step1 only FAILED: {e}")
                results[key]["step1_only"] = None

        if results[key].get("step1_2_3") is None:
            try:
                t0 = time.time()
                c = approach_step123(P, L, T, cfg, i)
                elapsed = time.time() - t0
                results[key]["step1_2_3"] = c
                results[key]["step1_2_3_time"] = elapsed
                print(f"  Step1+2+3:   {c:.2f}  ({elapsed:.1f}s)" if c else "  Step1+2+3: None")
            except Exception as e:
                print(f"  Step1+2+3 FAILED: {e}")
                results[key]["step1_2_3"] = None

        if results[key].get("rh2") is None:
            try:
                t0 = time.time()
                c = approach_rh2(cfg)
                elapsed = time.time() - t0
                results[key]["rh2"] = c
                results[key]["rh2_time"] = elapsed
                print(f"  RH2:         {c:.2f}  ({elapsed:.1f}s)" if c else "  RH2: None")
            except Exception as e:
                print(f"  RH2 FAILED: {e}")
                results[key]["rh2"] = None

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    valid = {k: v for k, v in results.items()
             if all(v.get(a) is not None for a in ["step1_only", "step1_2_3", "rh2"])}
    if not valid:
        print("No valid results to summarize.")
        return

    s1   = np.array([v["step1_only"] for v in valid.values()])
    s123 = np.array([v["step1_2_3"]  for v in valid.values()])
    rh2  = np.array([v["rh2"]        for v in valid.values()])

    # Timing (may be absent for instances run before timing was added)
    t1   = np.array([v["step1_only_time"] for v in valid.values() if v.get("step1_only_time")])
    t123 = np.array([v["step1_2_3_time"]  for v in valid.values() if v.get("step1_2_3_time")])
    trh2 = np.array([v["rh2_time"]        for v in valid.values() if v.get("rh2_time")])

    def fmt_time(arr):
        return f"{np.mean(arr):.1f}s" if len(arr) else "n/a"

    print(f"\n{'='*70}")
    print(f"SUMMARY: P={P} L={L} T={T}  ({len(valid)} instances)")
    print(f"{'='*70}")
    print(f"{'Approach':<20} {'Mean':>8} {'Std':>8} {'vs RH2':>8} {'AvgTime':>10}")
    print(f"{'-'*58}")
    print(f"{'RH2 Baseline':<20} {np.mean(rh2):>8.1f} {np.std(rh2):>8.1f} {'—':>8} {fmt_time(trh2):>10}")
    print(f"{'Step 1+2+3':<20} {np.mean(s123):>8.1f} {np.std(s123):>8.1f} {np.mean((s123-rh2)/rh2)*100:>7.1f}% {fmt_time(t123):>10}")
    print(f"{'Step 1 only':<20} {np.mean(s1):>8.1f} {np.std(s1):>8.1f} {np.mean((s1-rh2)/rh2)*100:>7.1f}% {fmt_time(t1):>10}")
    print(f"\nResults saved to: {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_products", type=int, required=True)
    parser.add_argument("--num_lines",    type=int, required=True)
    parser.add_argument("--num_periods",  type=int, required=True)
    parser.add_argument("--skip_train",   action="store_true",
                        help="Skip Step 1 training (use existing model in benchmark_results/)")
    parser.add_argument("--skip_compare", action="store_true",
                        help="Run training only, skip comparison")
    args = parser.parse_args()

    P, L, T = args.num_products, args.num_lines, args.num_periods

    mdir = model_dir(P, L, T)
    model_exists = (os.path.exists(os.path.join(mdir, "actor_agent0.pt")) and
                    os.path.exists(os.path.join(mdir, "actor_agent1.pt")))

    if not args.skip_train:
        if model_exists:
            print(f"[Step 1] Model already exists at {mdir}, skipping training.")
        else:
            train(P, L, T)
    else:
        if not model_exists:
            print(f"ERROR: --skip_train set but no model found at {mdir}")
            sys.exit(1)

    if not args.skip_compare:
        run_comparison(P, L, T)


if __name__ == "__main__":
    main()
