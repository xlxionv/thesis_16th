#!/bin/sh

# Load W&B API key from environment file
source ~/.wandb_env

env="BOSCH"
algo="rmappo"
exp="large_27_9_4_milp_agent0_active"
seed_max=1

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    # To change experiment settings (products/lines/periods/eval configs), edit the
    # option values below. Keep each option on its own line (no inline comments).
    CUDA_VISIBLE_DEVICES=0 python3 ../train/train_bosch.py \
      --algorithm_name ${algo} \
      --experiment_name "${exp}" \
      --seed ${seed} \
      --num_products 17 \
      --num_lines 6 \
      --num_periods 4 \
      --lookahead_days 4 \
      --allocator_mode relaxed_milp \
      --relaxed_milp_lookahead 4 \
      --relaxed_milp_time_limit 30 \
      --relaxed_milp_setup_time_mode p90 \
      --relaxed_milp_capacity_safety 0.85 \
      --relaxed_milp_use_manager_mask \
      --dense_setup_penalty 1.0 \
      --dense_production_reward 1.0 \
      --alpha_cost_weight 1.0 \
      --activation_penalty 20.0 \
      --load_balance_penalty 50.0 \
      --n_rollout_threads 8 \
      --n_training_threads 1 \
      --use_eval \
      --n_eval_rollout_threads 4 \
      --eval_interval 50 \
      --eval_configs dataset/test_benchmark_17_6_4/*.json \
      --num_env_steps 1000000 \
      --log_interval 5 \
      --entropy_coef 0.01 \
      --lr 3e-4 \
      --critic_lr 3e-4 \
      --clip_param 0.2 \
      --ppo_epoch 10 \
      --num_mini_batch 1 \
      --use_linear_lr_decay \
      --use_wandb
done