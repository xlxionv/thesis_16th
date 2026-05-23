#!/bin/sh
# Sensitivity analysis: vary relaxed_milp_setup_time_mode, all other params frozen.
# Run AFTER the Bayesian sweep — replace the BEST_* values with the best config from WandB.

BEST_LR=3e-4
BEST_ENTROPY=0.01
BEST_CLIP=0.2
BEST_PPO_EPOCH=10
BEST_GAMMA=0.99
BEST_GAE=0.95
BEST_ACTIVATION_PENALTY=20.0
BEST_LOAD_BALANCE_PENALTY=50.0
BEST_DENSE_PROD=1.0
BEST_DENSE_SETUP=1.0
BEST_HIDDEN=64

# Must equal the number of --eval_configs entries below so each thread = one instance
N_EVAL_CONFIGS=20

for mode in mean p85 p95 worst; do
    echo "Running sensitivity mode: ${mode}"
    CUDA_VISIBLE_DEVICES=0 python3 ../train/train_bosch.py \
      --algorithm_name rmappo \
      --experiment_name "sensitivity_${mode}" \
      --seed 1 \
      --num_products 5 --num_lines 2 --num_periods 4 \
      --lookahead_days 4 \
      --allocator_mode relaxed_milp \
      --relaxed_milp_lookahead 4 \
      --relaxed_milp_time_limit 30 \
      --relaxed_milp_setup_time_mode ${mode} \
      --relaxed_milp_setup_time_std_mult 1.0 \
      --relaxed_milp_capacity_safety 0.85 \
      --relaxed_milp_use_manager_mask \
      --alpha_cost_weight 1.0 \
      --hidden_size ${BEST_HIDDEN} \
      --lr ${BEST_LR} --critic_lr ${BEST_LR} \
      --entropy_coef ${BEST_ENTROPY} \
      --clip_param ${BEST_CLIP} \
      --ppo_epoch ${BEST_PPO_EPOCH} \
      --gamma ${BEST_GAMMA} \
      --gae_lambda ${BEST_GAE} \
      --activation_penalty ${BEST_ACTIVATION_PENALTY} \
      --load_balance_penalty ${BEST_LOAD_BALANCE_PENALTY} \
      --dense_production_reward ${BEST_DENSE_PROD} \
      --dense_setup_penalty ${BEST_DENSE_SETUP} \
      --n_rollout_threads 8 --n_training_threads 1 \
      --use_eval \
      --n_eval_rollout_threads ${N_EVAL_CONFIGS} \
      --eval_interval 50 \
      --eval_configs \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_1.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_2.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_3.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_4.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_5.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_6.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_7.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_8.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_9.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_10.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_11.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_12.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_13.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_14.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_15.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_16.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_17.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_18.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_19.json \
        dataset/Small/test_benchmark_5_2_4/test_P5_L2_T4_20.json \
      --num_env_steps 300000 \
      --log_interval 5 --num_mini_batch 1 \
      --use_linear_lr_decay \
      --use_wandb --user_name tntvan-iac-instagram
done
