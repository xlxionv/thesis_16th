#!/usr/bin/env python3
"""
run_sweep_step1.py — W&B sweep agent for Step 1 of the 3-step RL+MILP pipeline.

Trains the Process Agent (allocation) and Machine Agents (sequencing) using:
  - allocator_mode = jit      (quantities = raw demand, strictly JIT)
  - reward_mode   = step1     (Process Agent: -processing_time; Machine: -setup_cost)
  - obs_mode      = binary    (quantity-blind observations)
  - kill_switch_penalty       (massive penalty if demand unmet; backlog wiped)

Launch a sweep:
    cd /Users/nhi/thesis_14th-1
    wandb sweep onpolicy/scripts/train_bosch_scripts/sweep_config_step1.yaml
    wandb agent <sweep-id>
"""

import sys
from pathlib import Path

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Fixed args — edit problem size here, not in the yaml.
FIXED_ARGS = [
    "--algorithm_name", "rmappo",
    "--experiment_name", "sweep_step1_5_2_4",
    "--seed", "1",
    "--num_products", "5",
    "--num_lines", "2",
    "--num_periods", "4",
    "--lookahead_days", "4",
    # --- Step 1 pipeline flags ---
    "--reward_mode", "step1",
    "--allocator_mode", "jit",
    "--obs_mode", "binary",
    # --- training ---
    "--n_rollout_threads", "8",
    "--n_training_threads", "1",
    "--num_env_steps", "500000",
    "--log_interval", "5",
    "--num_mini_batch", "1",
    "--use_linear_lr_decay",
    # --- eval: all 100 instances ---
    "--use_eval",
    "--n_eval_rollout_threads", "5",
    "--eval_interval", "50",
    "--eval_configs",
        *[f"dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_{i}.json" for i in range(1, 101)],
    # --- W&B ---
    "--use_wandb",
    "--user_name", "tntvan-iac-instagram",
]

# Sweep parameters and their corresponding CLI flags.
SWEEP_PARAM_FLAGS = {
    "lr":                   "--lr",
    "critic_lr":            "--critic_lr",
    "entropy_coef":         "--entropy_coef",
    "clip_param":           "--clip_param",
    "ppo_epoch":            "--ppo_epoch",
    "gamma":                "--gamma",
    "gae_lambda":           "--gae_lambda",
    "dense_setup_penalty":  "--dense_setup_penalty",
    "dense_production_reward": "--dense_production_reward",
    "kill_switch_penalty":  "--kill_switch_penalty",
    "hidden_size":          "--hidden_size",
}


def main():
    run = wandb.init()
    cfg = dict(run.config)

    sweep_args = []
    for param, flag in SWEEP_PARAM_FLAGS.items():
        if param in cfg:
            sweep_args += [flag, str(cfg[param])]

    from onpolicy.scripts.train.train_bosch import main as train_main
    train_main(FIXED_ARGS + sweep_args)


if __name__ == "__main__":
    main()
