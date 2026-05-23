"""
rl_export_allocation.py — Bridge between Step 1 and Step 2

Loads a trained RL model (from Step 1), runs one deterministic episode on a
given problem instance, and saves the Process Agent's binary allocation
decisions per period AND per micro-step to a JSON file.

That JSON is then fed into milp_posthoc.py (Step 2) so the MILP receives the
RL agents' own allocation with exact per-step sequencing information.

Usage:
    python rl_export_allocation.py \
        --config   dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_1.json \
        --model_dir  results/BOSCH/rmappo/step1_rl_training/run1/models \
        --output   rl_allocation.json \
        --num_products 5 --num_lines 2 --num_periods 4

Output format (4D: period → step → L×P binary):
    {
      "period_0": {
        "step_0": [[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]],
        "step_1": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
        ...
      },
      "period_1": { ... }
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.envs.bosch.bosch_env import BoschEnv


def _decode_manager_action(action_vec, num_lines, num_products):
    """Convert one-hot policy output → binary L×P allocation mask."""
    one_hot = np.asarray(action_vec, dtype=np.float32).ravel()
    masks = np.zeros((num_lines, num_products), dtype=np.int32)
    for l in range(num_lines):
        for p in range(num_products):
            idx = (l * num_products + p) * 2
            if idx + 1 < len(one_hot) and one_hot[idx + 1] > 0.5:
                masks[l, p] = 1
    return masks


def _read_wandb_config(model_dir):
    """Read hidden_size and recurrent_N from wandb config.yaml if present."""
    config_path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        out = {}
        for key in ("hidden_size", "recurrent_N", "obs_mode", "reward_mode",
                    "allocator_mode", "max_actions_per_period"):
            v = cfg.get(key)
            if isinstance(v, dict):
                v = v.get("value")
            if v is not None:
                out[key] = v
        return out
    except Exception:
        return {}


def run_inference(args):
    device = torch.device("cpu")
    L, P = args.num_lines, args.num_products

    # Auto-detect architecture from wandb config.yaml in the model dir.
    wandb_cfg = _read_wandb_config(args.model_dir)

    # Parse full defaults from the training argument parser so the policy and env
    # receive every flag they expect, then override the fields specific to inference.
    from onpolicy.scripts.train.train_bosch import parse_args
    from onpolicy.config import get_config
    dummy_argv = [
        "--num_products", str(P),
        "--num_lines", str(L),
        "--num_periods", str(args.num_periods),
        "--max_actions_per_period", str(wandb_cfg.get("max_actions_per_period", getattr(args, "max_actions_per_period", 8))),
        "--lookahead_days", str(getattr(args, "lookahead_days", 4)),
        "--hidden_size", str(wandb_cfg.get("hidden_size", getattr(args, "hidden_size", 64))),
        "--recurrent_N", str(wandb_cfg.get("recurrent_N", getattr(args, "recurrent_N", 1))),
        "--reward_mode", str(wandb_cfg.get("reward_mode", "step1")),
        "--allocator_mode", str(wandb_cfg.get("allocator_mode", "jit")),
        "--obs_mode", str(wandb_cfg.get("obs_mode", "binary")),
        "--algorithm_name", "rmappo",
        "--experiment_name", "export_inference",
        "--seed", "1",
    ]
    env_args = parse_args(dummy_argv, get_config())
    env_args.eval_config_dicts = [json.load(open(args.config))]
    env_args.milp_lot_sizes_path = None

    env = BoschEnv(env_args, rank=0, is_eval=True)
    num_agents = env.num_agents  # 1 + L

    # Build policies (manager + shared machine)
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

    # Load actor weights
    model_dir = args.model_dir
    manager_ckpt = os.path.join(model_dir, "actor_agent0.pt")
    machine_ckpt = os.path.join(model_dir, "actor_agent1.pt")

    if not os.path.exists(manager_ckpt):
        raise FileNotFoundError(f"Manager actor not found: {manager_ckpt}")
    if not os.path.exists(machine_ckpt):
        raise FileNotFoundError(f"Machine actor not found: {machine_ckpt}")

    manager_policy.actor.load_state_dict(
        torch.load(manager_ckpt, map_location=device)
    )
    machine_policy.actor.load_state_dict(
        torch.load(machine_ckpt, map_location=device)
    )
    manager_policy.actor.eval()
    machine_policy.actor.eval()

    # RNN states: shape (1, recurrent_N, hidden_size)
    rN = env_args.recurrent_N
    hs = env_args.hidden_size
    rnn_states = [
        np.zeros((1, rN, hs), dtype=np.float32)
        for _ in range(num_agents)
    ]
    masks = np.ones((1, 1), dtype=np.float32)

    obs_list = env.reset()
    available_actions = env._build_available_actions()

    allocation = {}
    done = False

    print(f"[rl_export] Running inference over {args.num_periods} periods...")

    while not done:
        period_before = env.period_index

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

            # One-hot encode
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

        # Capture per-step masks at period boundaries.
        # _end_period() saves micro_step_masks → last_period_micro_step_masks
        # before _start_new_period() clears micro_step_masks.
        if env.period_index != period_before or done:
            period_key = f"period_{period_before}"
            allocation[period_key] = {
                f"step_{f}": (mask > 0.5).astype(int).tolist()
                for f, mask in sorted(env.last_period_micro_step_masks.items())
            }
            total_slots = sum(int(np.sum(v)) for v in allocation[period_key].values())
            print(f"  period {period_before}: {len(allocation[period_key])} steps, {total_slots} active slots")

    with open(args.output, "w") as f:
        json.dump(allocation, f, indent=2)

    print(f"\n[rl_export] Allocation saved to: {args.output}")
    print(f"[rl_export] Periods captured: {len(allocation)}")
    print(f"\nNext step (Step 2):")
    print(f"  python configs/bosch/milp_posthoc.py \\")
    print(f"    --config {args.config} \\")
    print(f"    --rl_allocation {args.output} \\")
    print(f"    --output milp_lot_sizes.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1 → Step 2 bridge: export RL allocation to JSON")
    parser.add_argument("--config", type=str, required=True, help="Path to the problem JSON config file")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory (contains actor_agent*.pt)")
    parser.add_argument("--output", type=str, default="rl_allocation.json", help="Output file for RL allocation")
    parser.add_argument("--num_products", type=int, required=True)
    parser.add_argument("--num_lines", type=int, required=True)
    parser.add_argument("--num_periods", type=int, required=True)
    parser.add_argument("--max_actions_per_period", type=int, default=8)
    parser.add_argument("--lookahead_days", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=64, help="Must match training hidden_size")
    parser.add_argument("--recurrent_N", type=int, default=1, help="Must match training recurrent_N")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")

    run_inference(args)
