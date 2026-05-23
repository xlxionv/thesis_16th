import sys
import os
import multiprocessing as mp
import platform
import json
import socket
from datetime import datetime
import numpy as np
from pathlib import Path
import torch

# Allow running this file directly from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import wandb
except ImportError:
    wandb = None

try:
    import setproctitle
except ImportError:
    setproctitle = None

from onpolicy.config import get_config
from onpolicy.envs.bosch import BoschEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def _load_bosch_config(config_path):
    if config_path is None:
        return {}
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Bosch config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Bosch config must be a JSON object (dict).")
    return data


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "BOSCH":
                env = BoschEnv(all_args, rank=rank, is_eval=False)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "BOSCH":
                env = BoschEnv(all_args, rank=rank, is_eval=True)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    """
    Add Bosch-specific arguments on top of the common config.
    """
    # --- NEW: Evaluation Configs ---
    parser.add_argument(
        "--eval_configs",
        type=str,
        nargs='+',
        default=None,
        help="List of multiple evaluation config files (JSON).",
    )

    # --- RESTORED: All Original Arguments and Help Strings ---
    parser.add_argument("--num_lines", type=int, default=17)
    parser.add_argument("--num_products", type=int, default=6)
    parser.add_argument("--num_periods", type=int, default=4)
    parser.add_argument(
        "--capacity_per_line",
        type=str,
        default="100.0",
        help="Per-line capacity (comma-separated) or scalar.",
    )
    parser.add_argument("--max_lot_size", type=int, default=10)
    parser.add_argument(
        "--manager_max_horizon",
        type=int,
        default=7,
        help="Max number of days of demand to cover per line (manager action).",
    )
    parser.add_argument(
        "--debug_daily_report",
        action="store_true",
        default=True,
        help="Print per-period capacity/assignment/backlog report for env 0.",
    )
    parser.add_argument(
        "--debug_report_interval",
        type=int,
        default=17,
        help="Print debug report every N periods.",
    )
    parser.add_argument(
        "--debug_report_episode_interval",
        type=int,
        default=100,
        help="Print detailed schedule every N episodes.",
    )
    parser.add_argument(
        "--debug_report_file",
        type=str,
        default=None,
        help="File to write detailed schedule analysis to.",
    )

    parser.add_argument("--holding_cost", type=float, default=1.0)
    parser.add_argument("--backlog_cost", type=float, default=10.0)
    parser.add_argument(
        "--per_product_backlog_penalty",
        type=float,
        default=500.0,
        help="Flat penalty added per product that has any backlog.",
    )
    parser.add_argument("--production_cost", type=float, default=1.0)
    parser.add_argument("--setup_cost", type=float, default=2.0)
    parser.add_argument(
        "--pm_cost",
        type=str,
        default="20.0",
        help="Per-line PM cost (comma-separated) or scalar.",
    )
    parser.add_argument(
        "--cm_cost",
        type=str,
        default="40.0",
        help="Per-line CM cost (comma-separated) or scalar.",
    )
    parser.add_argument("--alpha_cost_weight", type=float, default=0.5)
    parser.add_argument(
        "--hazard_rate",
        type=str,
        default="1e-3",
        help="Per-line hazard rates (comma-separated) or scalar.",
    )

    # Time-based parameters
    parser.add_argument(
        "--pm_time",
        type=str,
        default="0.0",
        help="Per-line PM time (comma-separated) or scalar.",
    )
    parser.add_argument(
        "--cm_time",
        type=str,
        default="0.0",
        help="Per-line CM time (comma-separated) or scalar.",
    )

    # Comma-separated lists for per-product parameters
    parser.add_argument(
        "--processing_time",
        type=str,
        default="1.0",
        help="Per-product processing time (hours per unit), comma-separated or scalar.",
    )
    parser.add_argument(
        "--processing_time_matrix",
        type=str,
        default=None,
        help="Flattened line×product processing time matrix (row-major).",
    )
    parser.add_argument(
        "--mean_demand",
        type=str,
        default="10.0",
        help="Per-product mean demand per period, comma-separated or scalar.",
    )

    # Optional scalar or matrices for setup and production
    parser.add_argument(
        "--setup_time",
        type=float,
        default=0.0,
        help="Base setup time (hours) for switching between different products.",
    )
    parser.add_argument(
        "--setup_cost_matrix",
        type=str,
        default=None,
        help="Flattened product×product setup cost matrix (row-major).",
    )
    parser.add_argument(
        "--setup_time_matrix",
        type=str,
        default=None,
        help="Flattened product×product setup time matrix (row-major).",
    )
    parser.add_argument(
        "--first_setup_cost",
        type=str,
        default=None,
        help="Initial setup cost when a line with no prior product starts production.",
    )
    parser.add_argument(
        "--first_setup_time",
        type=str,
        default=None,
        help="Initial setup time when a line with no prior product starts production.",
    )
    parser.add_argument(
        "--production_cost_matrix",
        type=str,
        default=None,
        help="Flattened line×product production cost matrix (row-major).",
    )
    parser.add_argument(
        "--eligibility_matrix",
        type=str,
        default=None,
        help="Flattened line×product eligibility matrix (0/1, row-major).",
    )
    parser.add_argument(
        "--demand_profile",
        type=str,
        default=None,
        help="Flattened period×product demand profile (row-major).",
    )
    parser.add_argument(
        "--max_actions_per_period",
        type=int,
        default=8,
        help="Maximum number of machine micro-actions per period.",
    )
    parser.add_argument(
        "--dense_production_reward",
        type=float,
        default=1.0,
        help="Immediate reward weight per produced unit for machine agents.",
    )
    parser.add_argument(
        "--dense_setup_penalty",
        type=float,
        default=2.0,
        help="Immediate penalty weight for setup cost during changeovers.",
    )
    parser.add_argument(
        "--dense_pm_penalty",
        type=float,
        default=1.0,
        help="Immediate penalty weight for PM cost when PM action is taken.",
    )
    parser.add_argument(
        "--allocator_lookahead",
        type=int,
        default=0,
        help="Lookahead days for heuristic allocator (0 = use lookahead_days).",
    )
    parser.add_argument(
        "--activation_penalty",
        type=float,
        default=0.0,
        help="Per-activated-slot penalty for manager reward.",
    )
    parser.add_argument(
        "--load_balance_penalty",
        type=float,
        default=0.0,
        help="Penalty weight on std-dev of per-line setup costs, added to manager reward.",
    )
    parser.add_argument(
        "--allocator_mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "jit", "relaxed_milp", "milp", "relaxed", "relaxed_lp"],
        help="Queue allocator used after the manager step.",
    )
    parser.add_argument(
        "--relaxed_milp_lookahead",
        type=int,
        default=4,
        help="Rolling-horizon window for allocator_mode=relaxed_milp.",
    )
    parser.add_argument(
        "--relaxed_milp_time_limit",
        type=float,
        default=10,
        help="CBC time limit per relaxed MILP allocator solve in seconds.",
    )
    parser.add_argument(
        "--relaxed_milp_use_manager_mask",
        action="store_true",
        default=False,
        help="Constrain relaxed MILP line-product choices by manager action.",
    )
    parser.add_argument(
        "--no_relaxed_milp_fallback",
        dest="relaxed_milp_fallback_to_heuristic",
        action="store_false",
        default=True,
        help="Disable fallback to heuristic allocator if relaxed MILP fails.",
    )
    parser.add_argument(
        "--relaxed_milp_setup_time_mode",
        type=str,
        default="average",
        choices=["average", "mean", "mean_std", "p75", "p85", "p90", "p95", "worst"],
        help="Robust setup-time estimate used in relaxed MILP capacity.",
    )
    parser.add_argument(
        "--relaxed_milp_setup_time_std_mult",
        type=float,
        default=1.0,
        help="Std multiplier when relaxed_milp_setup_time_mode=mean_std.",
    )
    parser.add_argument(
        "--relaxed_milp_capacity_safety",
        type=float,
        default=1.0,
        help="Capacity multiplier in relaxed MILP; use 0.90-0.95 for buffer.",
    )
    parser.add_argument(
        "--debug_actions",
        action="store_true",
        default=False,
        help="Print manager/machine actions for a few early steps.",
    )
    parser.add_argument(
        "--debug_action_steps",
        type=int,
        default=5,
        help="Number of early steps to print when debug_actions is enabled.",
    )
    parser.add_argument(
        "--shared_machine_policy",
        dest="shared_machine_policy",
        action="store_true",
        default=True,
        help="Share one policy among all machine agents (agents 1..num_lines).",
    )
    parser.add_argument(
        "--no_shared_machine_policy",
        dest="shared_machine_policy",
        action="store_false",
        help="Disable shared machine policy and use one policy per machine agent.",
    )
    parser.add_argument(
        "--kill_switch_penalty",
        type=float,
        default=1000.0,
        help=(
            "Flat penalty applied to all agents when any demand goes unmet at the "
            "end of a period (only active in reward_mode=step1). Backlog is wiped "
            "to zero after the penalty so it does not degrade future states."
        ),
    )
    parser.add_argument(
        "--obs_mode",
        type=str,
        default="full",
        choices=["full", "binary"],
        help=(
            "Observation representation. 'full': log-scaled continuous quantities "
            "(default). 'binary': quantity-blind — inventory, backlog, queue, and "
            "demand are shown as binary present/absent signals. The reward still "
            "uses raw quantities; agents feel consequences without seeing magnitudes."
        ),
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="full",
        choices=["full", "step1"],
        help=(
            "Reward signal for RL agents. "
            "'full': inventory+backlog+production+setup costs (default). "
            "'step1': Process Agent minimizes total processing time; "
            "Machine Agents minimize setup costs only (Step 1 of 3-step pipeline)."
        ),
    )
    parser.add_argument(
        "--milp_lot_sizes_path",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing MILP-optimized lot sizes per period "
            "(output of milp_posthoc.py). When set, the environment pre-fills the "
            "queue with these quantities and bypasses the RL allocator (Step 3 inference)."
        ),
    )

    all_args = parser.parse_known_args(args)[0]

    # Load eval configs into memory
    all_args.eval_config_dicts = []
    if all_args.eval_configs:
        for p in all_args.eval_configs:
            all_args.eval_config_dicts.append(_load_bosch_config(p))

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.n_rollout_threads > 1 and platform.system() == "Darwin":
        try:
            mp.set_start_method("fork")
            print("BOSCH: using multiprocessing start_method='fork' on macOS.")
        except RuntimeError:
            pass

    all_args.env_name = "BOSCH"

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False

    all_args.share_policy = False
    all_args.use_centralized_V = True
    all_args.use_policy_active_masks = True
    all_args.use_value_active_masks = False
    all_args.episode_length = all_args.num_periods * (all_args.max_actions_per_period + 1)

    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path("results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.experiment_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
        run_nums = [int(f.name.split("run")[1]) for f in run_dir.iterdir() if f.name.startswith("run")]
        curr_run = f"run{max(run_nums) + 1 if run_nums else 1}"
        run_dir = run_dir / curr_run
        run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": 1 + all_args.num_lines,
        "device": device,
        "run_dir": run_dir,
    }

    from onpolicy.runner.separated.mpe_runner import MPERunner as Runner
    runner = Runner(config)
    
    if runner.model_dir is not None:
        runner.restore()

    runner.run()

    envs.close()
    if eval_envs: 
        eval_envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])