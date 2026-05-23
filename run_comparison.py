"""
run_comparison.py - Compare three approaches on 100 test instances

Approach 1: Step 1 RL only (deterministic inference)
Approach 2: Step 1 + 2 + 3 pipeline (RL export + MILP + RL with MILP lots)
Approach 3: RH2 baseline (rolling horizon MILP)
"""

import sys
import os
import json
import re
import subprocess
import tempfile
import numpy as np

sys.path.insert(0, '/Users/nhi/thesis_14th-1')

import torch

MODEL_DIR = "/Users/nhi/thesis_14th-1/results/BOSCH/rmappo/sweep_step1_team_5_2_4/wandb/run-20260520_203418-0kt3xljv/files"
DATASET_DIR = "/Users/nhi/thesis_14th-1/dataset/Small/test_benchmark_5_2_4"
OUTPUT_FILE = "/Users/nhi/thesis_14th-1/results/comparison_100instances.json"

NUM_PRODUCTS = 5
NUM_LINES = 2
NUM_PERIODS = 4


def build_env_and_policies(config_path, milp_lot_sizes_path=None):
    """Build environment and policies for a single instance."""
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    from onpolicy.envs.bosch.bosch_env import BoschEnv
    from onpolicy.scripts.train.train_bosch import parse_args
    from onpolicy.config import get_config

    # Read wandb config
    import yaml
    config_yaml = os.path.join(MODEL_DIR, "config.yaml")
    wandb_cfg = {}
    if os.path.exists(config_yaml):
        with open(config_yaml) as f:
            cfg = yaml.safe_load(f)
        for key in ("hidden_size", "recurrent_N", "obs_mode", "reward_mode", "allocator_mode", "max_actions_per_period"):
            v = cfg.get(key)
            if isinstance(v, dict):
                v = v.get("value")
            if v is not None:
                wandb_cfg[key] = v

    dummy_argv = [
        "--num_products", str(NUM_PRODUCTS),
        "--num_lines", str(NUM_LINES),
        "--num_periods", str(NUM_PERIODS),
        "--max_actions_per_period", str(wandb_cfg.get("max_actions_per_period", 8)),
        "--lookahead_days", "4",
        "--hidden_size", str(wandb_cfg.get("hidden_size", 128)),
        "--recurrent_N", str(wandb_cfg.get("recurrent_N", 1)),
        "--reward_mode", str(wandb_cfg.get("reward_mode", "step1")),
        "--allocator_mode", str(wandb_cfg.get("allocator_mode", "jit")),
        "--obs_mode", str(wandb_cfg.get("obs_mode", "binary")),
        "--algorithm_name", "rmappo",
        "--experiment_name", "comparison_inference",
        "--seed", "1",
    ]
    env_args = parse_args(dummy_argv, get_config())
    env_args.eval_config_dicts = [json.load(open(config_path))]
    env_args.milp_lot_sizes_path = milp_lot_sizes_path

    device = torch.device("cpu")
    env = BoschEnv(env_args, rank=0, is_eval=True)
    num_agents = env.num_agents

    manager_policy = R_MAPPOPolicy(
        env_args,
        env.observation_space[0],
        env.observation_space[0],
        env.action_space[0],
        device=device,
    )
    machine_policy = R_MAPPOPolicy(
        env_args,
        env.observation_space[1],
        env.observation_space[1],
        env.action_space[1],
        device=device,
    )

    manager_ckpt = os.path.join(MODEL_DIR, "actor_agent0.pt")
    machine_ckpt = os.path.join(MODEL_DIR, "actor_agent1.pt")

    manager_policy.actor.load_state_dict(torch.load(manager_ckpt, map_location=device))
    machine_policy.actor.load_state_dict(torch.load(machine_ckpt, map_location=device))
    manager_policy.actor.eval()
    machine_policy.actor.eval()

    return env, env_args, manager_policy, machine_policy, num_agents


def run_rl_episode(env, env_args, manager_policy, machine_policy, num_agents):
    """Run one deterministic episode; return episode_total_cost."""
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
                action, rnn_out = policy.act(
                    obs_np,
                    rnn_states[agent_id],
                    masks,
                    avail_np,
                    deterministic=True,
                )

            action_np = action.detach().cpu().numpy()
            rnn_states[agent_id] = rnn_out.detach().cpu().numpy()

            act_space = env.action_space[agent_id]
            if act_space.__class__.__name__ == "MultiDiscrete":
                action_env = None
                for i in range(act_space.shape):
                    one_hot = np.eye(act_space.high[i] + 1)[action_np[:, i]]
                    action_env = one_hot if action_env is None else np.concatenate([action_env, one_hot], axis=1)
                actions_env.append(action_env[0])
            else:
                actions_env.append(np.squeeze(np.eye(act_space.n)[action_np], 1)[0])

        obs_list, _, dones, infos = env.step(actions_env)
        available_actions = env._build_available_actions()
        done = all(dones)

        # Update episode_total_cost whenever it's available
        if "episode_total_cost" in infos[0]:
            episode_total_cost = infos[0]["episode_total_cost"]

    return episode_total_cost


def approach1_step1_only(config_path):
    """Run Step 1 RL only inference."""
    env, env_args, manager_policy, machine_policy, num_agents = build_env_and_policies(config_path)
    cost = run_rl_episode(env, env_args, manager_policy, machine_policy, num_agents)
    return cost


def approach2_step123_pipeline(config_path, instance_idx):
    """Run Step 1+2+3 pipeline."""
    alloc_tmp = f"/tmp/rl_alloc_{instance_idx}.json"
    milp_tmp = f"/tmp/milp_lots_{instance_idx}.json"

    # Step 1: Export RL allocation
    cmd1 = [
        sys.executable,
        "/Users/nhi/thesis_14th-1/configs/bosch/rl_export_allocation.py",
        "--config", config_path,
        "--model_dir", MODEL_DIR,
        "--output", alloc_tmp,
        "--num_products", str(NUM_PRODUCTS),
        "--num_lines", str(NUM_LINES),
        "--num_periods", str(NUM_PERIODS),
    ]
    result1 = subprocess.run(cmd1, capture_output=True, text=True,
                             cwd="/Users/nhi/thesis_14th-1")
    if result1.returncode != 0:
        print(f"  [Step1 export] FAILED for instance {instance_idx}: {result1.stderr[-500:]}")
        return None

    # Step 2: Run MILP
    cmd2 = [
        sys.executable,
        "/Users/nhi/thesis_14th-1/configs/bosch/milp_posthoc.py",
        "--config", config_path,
        "--rl_allocation", alloc_tmp,
        "--output", milp_tmp,
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True,
                             cwd="/Users/nhi/thesis_14th-1")
    if result2.returncode != 0:
        print(f"  [Step2 MILP] FAILED for instance {instance_idx}: {result2.stderr[-500:]}")
        return None

    if not os.path.exists(milp_tmp):
        print(f"  [Step2] MILP output file not found: {milp_tmp}")
        return None

    # Step 3: Run RL inference with MILP lot sizes
    env, env_args, manager_policy, machine_policy, num_agents = build_env_and_policies(
        config_path, milp_lot_sizes_path=milp_tmp
    )
    cost = run_rl_episode(env, env_args, manager_policy, machine_policy, num_agents)

    # Cleanup
    for f in [alloc_tmp, milp_tmp]:
        try:
            os.remove(f)
        except:
            pass

    return cost


def approach3_rh2_baseline(config_path):
    """Run RH2 baseline and parse TOTAL COST from output."""
    cmd = [
        sys.executable,
        "/Users/nhi/thesis_14th-1/configs/bosch/rh2_baseline.py",
        "--config", config_path,
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd="/Users/nhi/thesis_14th-1", timeout=300)

    output = result.stdout + result.stderr
    # Parse: "  TOTAL COST        :  $   XXXX.XX"
    match = re.search(r'TOTAL COST\s*:\s*\$\s*([\d,]+\.?\d*)', output)
    if match:
        cost_str = match.group(1).replace(',', '')
        return float(cost_str)
    else:
        print(f"  [RH2] Could not parse TOTAL COST from output. Last 300 chars: {output[-300:]}")
        return None


def main():
    results = {}

    # Load existing results if any (for resuming)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")

    for i in range(1, 101):
        instance_key = f"instance_{i}"
        config_path = os.path.join(DATASET_DIR, f"test_P5_L2_T4_{i}.json")

        if not os.path.exists(config_path):
            print(f"[{i}/100] SKIP - file not found: {config_path}")
            continue

        # Check if already fully computed
        if instance_key in results:
            r = results[instance_key]
            if all(k in r and r[k] is not None for k in ["step1_only", "step1_2_3", "rh2"]):
                print(f"[{i}/100] Already complete, skipping")
                continue

        print(f"\n[{i}/100] Processing {config_path}")
        if instance_key not in results:
            results[instance_key] = {}

        # Approach 1
        if results[instance_key].get("step1_only") is None:
            try:
                c1 = approach1_step1_only(config_path)
                results[instance_key]["step1_only"] = c1
                print(f"  Approach 1 (Step1 only):    {c1:.2f}" if c1 is not None else "  Approach 1: None")
            except Exception as e:
                print(f"  Approach 1 FAILED: {e}")
                results[instance_key]["step1_only"] = None

        # Approach 2
        if results[instance_key].get("step1_2_3") is None:
            try:
                c2 = approach2_step123_pipeline(config_path, i)
                results[instance_key]["step1_2_3"] = c2
                print(f"  Approach 2 (Step1+2+3):     {c2:.2f}" if c2 is not None else "  Approach 2: None")
            except Exception as e:
                print(f"  Approach 2 FAILED: {e}")
                results[instance_key]["step1_2_3"] = None

        # Approach 3
        if results[instance_key].get("rh2") is None:
            try:
                c3 = approach3_rh2_baseline(config_path)
                results[instance_key]["rh2"] = c3
                print(f"  Approach 3 (RH2):           {c3:.2f}" if c3 is not None else "  Approach 3: None")
            except Exception as e:
                print(f"  Approach 3 FAILED: {e}")
                results[instance_key]["rh2"] = None

        # Save after each instance
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS SAVED TO: {OUTPUT_FILE}")
    print(f"{'='*60}")

    # Summary statistics
    approaches = {
        "step1_only": "Step 1 RL Only",
        "step1_2_3": "Step 1+2+3 Pipeline",
        "rh2": "RH2 Baseline",
    }

    print(f"\n{'SUMMARY STATISTICS':^60}")
    print(f"{'='*60}")

    for key, name in approaches.items():
        vals = [v[key] for v in results.values()
                if key in v and v[key] is not None]
        n = len(vals)
        if n == 0:
            print(f"\n{name}: No valid results")
            continue
        arr = np.array(vals)
        print(f"\n{name} (n={n}):")
        print(f"  Mean  : {np.mean(arr):.2f}")
        print(f"  Std   : {np.std(arr):.2f}")
        print(f"  Min   : {np.min(arr):.2f}")
        print(f"  Max   : {np.max(arr):.2f}")
        print(f"  Median: {np.median(arr):.2f}")

    print(f"\n{'='*60}")

    # How often does each approach win?
    valid_instances = [
        k for k, v in results.items()
        if all(v.get(a) is not None for a in approaches)
    ]
    print(f"\nPairwise comparisons on {len(valid_instances)} instances with all 3 results:")

    if valid_instances:
        s1 = np.array([results[k]["step1_only"] for k in valid_instances])
        s123 = np.array([results[k]["step1_2_3"] for k in valid_instances])
        rh2 = np.array([results[k]["rh2"] for k in valid_instances])

        print(f"\n  Step1+2+3 vs Step1 only:")
        print(f"    Step1+2+3 better: {np.sum(s123 < s1)} / {len(valid_instances)}")
        print(f"    Step1 only better: {np.sum(s1 < s123)} / {len(valid_instances)}")
        print(f"    Avg improvement (Step1+2+3 - Step1): {np.mean(s123 - s1):.2f}")

        print(f"\n  RH2 vs Step1 only:")
        print(f"    RH2 better: {np.sum(rh2 < s1)} / {len(valid_instances)}")
        print(f"    Step1 only better: {np.sum(s1 < rh2)} / {len(valid_instances)}")
        print(f"    Avg improvement (RH2 - Step1): {np.mean(rh2 - s1):.2f}")

        print(f"\n  RH2 vs Step1+2+3:")
        print(f"    RH2 better: {np.sum(rh2 < s123)} / {len(valid_instances)}")
        print(f"    Step1+2+3 better: {np.sum(s123 < rh2)} / {len(valid_instances)}")
        print(f"    Avg improvement (RH2 - Step1+2+3): {np.mean(rh2 - s123):.2f}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
