#!/usr/bin/env python3
"""
run_sweep_step1_team.py — W&B sweep for Step 1 with shared team reward.

Team reward = -(proc_time + setup + PM + CM) per period.
All agents receive the same signal; inventory/backlog excluded (MILP's job).

Launch:
    cd /Users/nhi/thesis_14th-1
    wandb sweep onpolicy/scripts/train_bosch_scripts/sweep_config_step1_team.yaml
    wandb agent <sweep-id>
"""

import sys
from pathlib import Path

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

FIXED_ARGS = [
    "--algorithm_name", "rmappo",
    "--experiment_name", "sweep_step1_team_5_2_4",
    "--seed", "1",
    "--num_products", "5",
    "--num_lines", "2",
    "--num_periods", "4",
    "--lookahead_days", "4",
    "--reward_mode", "step1",
    "--allocator_mode", "jit",
    "--obs_mode", "binary",
    "--dense_setup_penalty", "0.0",   # excluded — already in team reward
    "--n_rollout_threads", "8",
    "--n_training_threads", "1",
    "--num_env_steps", "500000",
    "--log_interval", "5",
    "--num_mini_batch", "1",
    "--use_linear_lr_decay",
    "--use_eval",
    "--n_eval_rollout_threads", "5",
    "--eval_interval", "50",
    "--eval_configs",
        *[f"dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_{i}.json" for i in range(1, 101)],
    "--use_wandb",
    "--user_name", "tntvan-iac-instagram",
]

SWEEP_PARAM_FLAGS = {
    "lr":                      "--lr",
    "critic_lr":               "--critic_lr",
    "entropy_coef":            "--entropy_coef",
    "clip_param":              "--clip_param",
    "ppo_epoch":               "--ppo_epoch",
    "gamma":                   "--gamma",
    "gae_lambda":              "--gae_lambda",
    "dense_production_reward": "--dense_production_reward",
    "kill_switch_penalty":     "--kill_switch_penalty",
    "hidden_size":             "--hidden_size",
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
