#!/usr/bin/env python3

import sys
from pathlib import Path

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Fixed production parameters — mirror train_bosch.sh exactly.
# Edit these if you change the problem size or allocator settings.
FIXED_ARGS = [
    "--algorithm_name", "rmappo",
    "--experiment_name", "sweep_5_2_4",
    "--seed", "1",
    "--num_products", "5",
    "--num_lines", "2",
    "--num_periods", "4",
    "--lookahead_days", "4",
    "--allocator_mode", "relaxed_milp",
    "--relaxed_milp_lookahead", "4",
    "--relaxed_milp_time_limit", "30",
    "--relaxed_milp_setup_time_mode", "p90",
    "--relaxed_milp_capacity_safety", "0.85",
    "--relaxed_milp_use_manager_mask",
    "--alpha_cost_weight", "1.0",
    "--n_rollout_threads", "8",
    "--n_training_threads", "1",
    "--use_eval",
    "--n_eval_rollout_threads", "5",
    "--eval_interval", "50",
    "--eval_configs",
        "dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_1.json",
        "dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_2.json",
        "dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_3.json",
        "dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_4.json",
        "dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_5.json",
    "--num_env_steps", "100000",
    "--log_interval", "5",
    "--num_mini_batch", "1",
    "--use_linear_lr_decay",
    "--use_wandb",
    "--user_name", "tntvan-iac-instagram",  # <-- replace with your WandB username/entity
]

# Maps sweep parameter names (as defined in sweep_config.yaml) to CLI flags.
SWEEP_PARAM_FLAGS = {
    "lr":                      "--lr",
    "critic_lr":               "--critic_lr",
    "entropy_coef":            "--entropy_coef",
    "clip_param":              "--clip_param",
    "ppo_epoch":               "--ppo_epoch",
    "gamma":                   "--gamma",
    "gae_lambda":              "--gae_lambda",
    "activation_penalty":      "--activation_penalty",
    "load_balance_penalty":    "--load_balance_penalty",
    "dense_production_reward": "--dense_production_reward",
    "dense_setup_penalty":     "--dense_setup_penalty",
    "hidden_size":             "--hidden_size",
}


def main():
    # wandb.init() receives the sampled hyperparameters from the sweep agent.
    # train_bosch.py also calls wandb.init(reinit=True), which is harmless here.
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
