import json
import os
import numpy as np
import random

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

try:
    import pulp
except ImportError:
    pulp = None

from onpolicy.utils.multi_discrete import MultiDiscrete


class BoschEnv(object):

    def __init__(self, args, rank=0, is_eval=False):
        self.args = args
        self.rng = np.random.RandomState(getattr(args, "seed", 1) + rank)
        self.is_eval = is_eval
        
        # Load evaluation configs if we are in testing mode
        self.eval_configs = getattr(args, "eval_config_dicts", []) if is_eval else []
        # Each eval env is pinned to one fixed instance (rank-based) so W&B tracks
        # progress on the same instances every eval call rather than cycling.
        self.eval_idx = rank % max(1, len(self.eval_configs)) if is_eval else 0

        # Extract fixed dimensions (These CANNOT change during a single training run)
        self.num_lines = int(getattr(args, "num_lines", 6))
        self.num_products = int(getattr(args, "num_products", 18))
        
        num_periods = getattr(args, "num_periods", None)
        if num_periods is None:
            num_periods = getattr(args, "episode_length", 24)
        self.num_periods = int(num_periods)
        
        self.max_actions_per_period = int(getattr(args, "max_actions_per_period", 8))
        self.manager_max_horizon = int(getattr(args, "manager_max_horizon", 7))
        self.lookahead_days = int(getattr(args, "lookahead_days", 5))
        self.allocator_lookahead = int(getattr(args, "allocator_lookahead", 0))
        if self.allocator_lookahead <= 0:
            self.allocator_lookahead = self.lookahead_days

        self.num_agents = 1 + self.num_lines
        
        self._init_spaces()

        # Initialize attributes before the first real reset
        if self.is_eval and len(self.eval_configs) > 0:
            self._load_config(self.eval_configs[0])
        else:
            self._load_config(self._generate_random_instance())

        self.current_step = 0
        self.period_index = 0
        self.step_in_period = 0
        self._reset_state()

    def _init_spaces(self):
        self.obs_dim = (
            2 * self.num_products
            + self.num_lines * self.num_products
            + self.num_products
            + self.num_products
            + self.num_products
            + self.lookahead_days * self.num_products
            + 1
            + self.num_lines
            + self.num_lines * self.num_products
            + self.num_lines
            + self.num_lines
            + self.num_lines
            + self.num_products
            + self.num_lines * self.num_products
            + self.num_products
            + self.num_products
        )

        high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        low = -high
        self.observation_space = [
            spaces.Box(low=low, high=high, dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        share_high = np.full(self.obs_dim * self.num_agents, np.inf, dtype=np.float32)
        share_low = -share_high
        self.share_observation_space = [
            spaces.Box(low=share_low, high=share_high, dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        lot_dims = [[0, 1]] * (self.num_lines * self.num_products)
        agent0_act = MultiDiscrete(lot_dims)
        machine_act = spaces.Discrete(self.num_products + 2)
        self.action_space = [agent0_act] + [machine_act for _ in range(self.num_lines)]

    def _generate_random_instance(self):
        P = self.num_products
        L = self.num_lines
        T = self.num_periods
        capacity_hours = 120.0
        
        cfg = {
            "num_products": P,
            "num_lines": L,
            "num_periods": T,
            "capacity_per_line": [capacity_hours] * L,
            "eligibility_matrix": [[1 for _ in range(P)] for _ in range(L)],
            "processing_time_matrix": [[self.rng.uniform(11/60, 15/60) for _ in range(P)] for _ in range(L)],
            "production_cost_matrix": [[self.rng.uniform(1.3, 1.6) for _ in range(P)] for _ in range(L)],
            "demand_profile": [[self.rng.randint(0, 150) for _ in range(P)] for _ in range(T)],
            "hazard_rate": [self.rng.choice([2.0, 3.0]) / capacity_hours for _ in range(L)],
            "cm_time": [self.rng.uniform(7/60, 11/60) for _ in range(L)],
            "pm_time": [self.rng.uniform(2.5, 3.5) for _ in range(L)],
            "pm_cost": [self.rng.uniform(10.0, 20.0) for _ in range(L)],
            "cm_cost": [self.rng.uniform(2.0, 3.0) for _ in range(L)],
            
            "holding_cost": 1.75,
            "backlog_cost": 5.25,
            "per_product_backlog_penalty": 12.25,
            
            "first_setup_time": [self.rng.uniform(1.2, 2.4) for _ in range(L)],
            "first_setup_cost": [self.rng.uniform(25.0, 30.0) for _ in range(L)],
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
                        t_row.append(self.rng.uniform(1.2, 2.4))
                        c_row.append(self.rng.uniform(25.0, 30.0))
                t_mat.append(t_row)
                c_mat.append(c_row)
            cfg["setup_time_matrix"].append(t_mat)
            cfg["setup_cost_matrix"].append(c_mat)

        return cfg

    def _get_array_arg(self, cfg, name, length, default):
        raw = cfg.get(name, getattr(self.args, name, None))
        if raw is None:
            return np.ones(length, dtype=np.float32) * float(default)

        if isinstance(raw, (list, np.ndarray)):
            arr = np.asarray(raw, dtype=np.float32)
        else:
            parts = str(raw).split(",")
            arr = np.array([float(p) for p in parts], dtype=np.float32)

        if arr.size == 1:
            arr = np.repeat(arr, length)

        if arr.size != length:
            raise ValueError(f"Argument {name} expects length {length}, got {arr.size}.")
        return arr

    def _get_matrix_arg(self, cfg, name, shape, default):
        raw = cfg.get(name, getattr(self.args, name, None))
        rows, cols = shape
        if raw is None:
            return np.ones((rows, cols), dtype=np.float32) * float(default)

        if isinstance(raw, (list, np.ndarray)):
            arr = np.asarray(raw, dtype=np.float32)
        else:
            parts = str(raw).split(",")
            arr = np.array([float(p) for p in parts], dtype=np.float32)

        if arr.size == 1:
            arr = np.repeat(arr, rows * cols)

        if arr.size != rows * cols:
            raise ValueError(f"Argument {name} expects {rows * cols} values, got {arr.size}.")
        return arr.reshape(rows, cols)

    def _get_tensor_arg(self, cfg, name, shape, default):
        raw = cfg.get(name, getattr(self.args, name, None))
        total = int(np.prod(shape))
        if raw is None:
            return np.ones(shape, dtype=np.float32) * float(default)

        if isinstance(raw, (list, np.ndarray)):
            arr = np.asarray(raw, dtype=np.float32)
        else:
            parts = str(raw).split(",")
            arr = np.array([float(p) for p in parts], dtype=np.float32)

        if arr.size == 1:
            arr = np.repeat(arr, total)

        per_line = int(np.prod(shape[1:]))
        if arr.size == per_line and shape[0] > 1:
            arr = np.tile(arr, shape[0])

        if arr.size != total:
            raise ValueError(
                f"Argument {name} expects {total} values (or {per_line} to "
                f"tile across lines), got {arr.size}."
            )
        return arr.reshape(shape)

    def _load_config(self, cfg):
        self.capacity_per_line = self._get_array_arg(cfg, "capacity_per_line", self.num_lines, default=120.0)
        self.holding_cost = float(cfg.get("holding_cost", 1.75))
        self.backlog_cost = float(cfg.get("backlog_cost", 5.25))
        
        val = cfg.get("per_product_backlog_penalty", 12.25)
        if isinstance(val, (list, tuple)):
            self.per_product_backlog_penalty = np.array(val, dtype=np.float32)
        else:
            self.per_product_backlog_penalty = np.full(self.num_products, float(val), dtype=np.float32)
            
        self.production_cost_matrix = self._get_matrix_arg(cfg, "production_cost_matrix", (self.num_lines, self.num_products), default=1.0)
        self.pm_cost = self._get_array_arg(cfg, "pm_cost", self.num_lines, default=15.0)
        self.cm_cost = self._get_array_arg(cfg, "cm_cost", self.num_lines, default=2.5)
        
        self.alpha_cost_weight = float(cfg.get("alpha_cost_weight", getattr(self.args, "alpha_cost_weight", 0.5)))

        self.machine_service_cost_share_beta = float(cfg.get("machine_service_cost_share_beta", getattr(self.args, "machine_service_cost_share_beta", 0.0)))
        self.machine_service_cost_share_mode = str(cfg.get("machine_service_cost_share_mode", "assignment")).strip().lower()
        self.machine_service_cost_share_include_inventory = bool(cfg.get("machine_service_cost_share_include_inventory", False))
        self.machine_service_cost_share_include_backlog = bool(cfg.get("machine_service_cost_share_include_backlog", True))

        self.product_codes = cfg.get("product_codes", getattr(self.args, "product_codes", None))
        self.line_codes = cfg.get("line_codes", getattr(self.args, "line_codes", None))

        self.hazard_rate = self._get_array_arg(cfg, "hazard_rate", self.num_lines, default=2.5/120.0)
        self.pm_time = self._get_array_arg(cfg, "pm_time", self.num_lines, default=3.0)
        self.cm_time = self._get_array_arg(cfg, "cm_time", self.num_lines, default=9/60)

        self.dense_production_reward = float(cfg.get("dense_production_reward", getattr(self.args, "dense_production_reward", 1.0)))
        self.dense_setup_penalty = float(cfg.get("dense_setup_penalty", getattr(self.args, "dense_setup_penalty", 2.0)))
        self.dense_pm_penalty = float(cfg.get("dense_pm_penalty", getattr(self.args, "dense_pm_penalty", 1.0)))
        self.activation_penalty = float(cfg.get("activation_penalty", getattr(self.args, "activation_penalty", 0.0)))
        self.load_balance_penalty = float(cfg.get("load_balance_penalty", getattr(self.args, "load_balance_penalty", 0.0)))

        self.allocator_mode = str(cfg.get("allocator_mode", getattr(self.args, "allocator_mode", "heuristic"))).strip().lower()
        if self.allocator_mode in ("milp", "relaxed", "relaxed_lp"):
            self.allocator_mode = "relaxed_milp"
            
        self.relaxed_milp_lookahead = int(cfg.get("relaxed_milp_lookahead", getattr(self.args, "relaxed_milp_lookahead", 2)))
        self.relaxed_milp_time_limit = float(cfg.get("relaxed_milp_time_limit", getattr(self.args, "relaxed_milp_time_limit", 1.0)))
        self.relaxed_milp_use_manager_mask = bool(cfg.get("relaxed_milp_use_manager_mask", getattr(self.args, "relaxed_milp_use_manager_mask", False)))
        self.relaxed_milp_fallback_to_heuristic = bool(cfg.get("relaxed_milp_fallback_to_heuristic", getattr(self.args, "relaxed_milp_fallback_to_heuristic", True)))
        self.relaxed_milp_setup_time_mode = str(cfg.get("relaxed_milp_setup_time_mode", getattr(self.args, "relaxed_milp_setup_time_mode", "average"))).strip().lower()
        if self.relaxed_milp_setup_time_mode == "mean":
            self.relaxed_milp_setup_time_mode = "average"
            
        self.relaxed_milp_setup_time_std_mult = float(cfg.get("relaxed_milp_setup_time_std_mult", getattr(self.args, "relaxed_milp_setup_time_std_mult", 1.0)))
        self.relaxed_milp_capacity_safety = float(cfg.get("relaxed_milp_capacity_safety", getattr(self.args, "relaxed_milp_capacity_safety", 1.0)))

        self.processing_time_matrix = self._get_matrix_arg(cfg, "processing_time_matrix", (self.num_lines, self.num_products), default=0.2)

        self.first_setup_cost = self._get_array_arg(cfg, "first_setup_cost", self.num_lines, default=27.5)
        self.first_setup_time = self._get_array_arg(cfg, "first_setup_time", self.num_lines, default=1.8)
        self.setup_cost_matrix = self._get_tensor_arg(cfg, "setup_cost_matrix", (self.num_lines, self.num_products, self.num_products), default=27.5)
        self.setup_time_matrix = self._get_tensor_arg(cfg, "setup_time_matrix", (self.num_lines, self.num_products, self.num_products), default=1.8)

        self.line_eligibility = self._get_matrix_arg(cfg, "eligibility_matrix", (self.num_lines, self.num_products), default=1.0)
        self.line_eligibility = (self.line_eligibility > 0.5).astype(np.float32)

        demand_profile_raw = cfg.get("demand_profile", None)
        if demand_profile_raw is not None:
            self.demand_profile = self._get_matrix_arg(cfg, "demand_profile", (self.num_periods, self.num_products), default=0.0)
            self.demand = self.demand_profile.astype(np.float32).copy()
        else:
            self.demand = np.zeros((self.num_periods, self.num_products), dtype=np.float32)
            
        self.avg_demand_per_product = np.mean(self.demand, axis=0).astype(np.float32)

        self.reward_mode = str(cfg.get("reward_mode", getattr(self.args, "reward_mode", "full"))).strip().lower()
        self.kill_switch_penalty = float(cfg.get("kill_switch_penalty", getattr(self.args, "kill_switch_penalty", 1000.0)))
        self.obs_mode = str(cfg.get("obs_mode", getattr(self.args, "obs_mode", "full"))).strip().lower()

        milp_lot_sizes_path = cfg.get("milp_lot_sizes_path", getattr(self.args, "milp_lot_sizes_path", None))
        self.milp_lot_sizes = None
        if milp_lot_sizes_path:
            _p = str(milp_lot_sizes_path)
            if os.path.exists(_p):
                with open(_p) as _f:
                    self.milp_lot_sizes = json.load(_f)

    def seed(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

    def reset(self):
        if self.is_eval and len(self.eval_configs) > 0:
            cfg = self.eval_configs[self.eval_idx]  # always same instance for this env
            self._load_config(cfg)
        else:
            cfg = self._generate_random_instance()
            self._load_config(cfg)
            
        self._reset_state()
        self._start_new_period()
        return self._build_observations()

    def step(self, actions_env):
        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)
        done = False

        # Per-micro-step: manager allocates then machines produce at every step.
        # MILP inference mode (Step 3) keeps old queue-drain behaviour — manager
        # loads lot sizes at step 0 only; machines drain over all steps.
        self._manager_step(actions_env)
        self._machines_step(actions_env, rewards)

        self.period_queue_sum += np.sum(self.queue, axis=1)
        self.period_queue_steps += 1

        self.step_in_period += 1
        self.current_step += 1

        end_of_period = (
            self.step_in_period > self.max_actions_per_period
            or self._period_effectively_over()
        )

        if end_of_period:
            day_rewards = self._end_period()
            rewards += day_rewards
            self.period_index += 1
            self.step_in_period = 0

            if self.period_index < self.num_periods:
                self._start_new_period()
            else:
                done = True

        obs = self._build_observations()
        dones = [done for _ in range(self.num_agents)]
        next_available_actions = self._build_available_actions()
        infos = []
        for agent_id in range(self.num_agents):
            info = {"available_actions": next_available_actions[agent_id].copy()}
            if agent_id == 0:
                info["manager_active"] = 1.0  # manager acts at every micro-step
                if end_of_period:
                    info["period_inv_cost"] = float(self.last_inv_cost)
                    info["period_backlog_cost"] = float(self.last_backlog_cost)
                    info["period_inv_qty"] = float(self.last_inv_qty)
                    info["period_backlog_qty"] = float(self.last_backlog_qty)
                    info["period_manager_horizons"] = self.last_manager_horizons.copy()
                    info["period_manager_horizon_mean"] = float(np.mean(self.last_manager_horizons))
                    info["period_manager_activated_mean"] = float(np.sum(self.last_manager_masks) / max(1, self.num_lines))
                    info["period_index"] = int(self.last_period_index)
                    info["period_capacity_per_product"] = self.last_capacity_per_product.copy()
                    info["period_assigned_lines_per_product"] = self.last_assigned_lines_per_product.copy()
                    info["period_unmet_demand_per_product"] = self.last_unmet_demand_per_product.copy()
                    if self.product_codes is not None:
                        info["period_product_codes"] = list(self.product_codes)
                    if self.line_codes is not None:
                        info["period_line_codes"] = list(self.line_codes)
                    info["period_queue_avg_per_line"] = self.last_queue_avg_per_line.copy()
                    info["period_backlog_per_product"] = self.last_backlog_per_product.copy()
                    info["period_inventory_per_product"] = self.last_inventory_per_product.copy()
                    info["period_prod_cost"] = float(self.last_prod_cost)
                    info["period_prod_cost_per_line"] = self.last_prod_cost_per_line.copy()
                    info["period_setup_cost"] = float(self.last_setup_cost)
                    info["period_setup_cost_per_line"] = self.last_setup_cost_per_line.copy()
                    info["period_pm_cost"] = float(self.last_pm_cost)
                    info["period_cm_cost"] = float(self.last_cm_cost)
                    info["period_utilization"] = float(self.last_utilization_total)
                    info["period_utilization_per_line"] = self.last_utilization_per_line.copy()
                    info["period_allocator_status"] = self.last_allocator_status
                    info["period_allocator_objective"] = float(self.last_allocator_objective)
                    info["period_allocator_capacity_safety"] = float(self.relaxed_milp_capacity_safety)
                    
                    period_total = (
                        float(self.last_inv_cost)
                        + float(self.last_backlog_cost)
                        + float(self.last_prod_cost)
                        + float(self.last_setup_cost)
                        + float(self.last_pm_cost)
                        + float(self.last_cm_cost)
                    )
                    info["period_total_cost"] = period_total
                    self.episode_total_cost += period_total
                    info["episode_total_cost"] = float(self.episode_total_cost)
            else:
                info["machine_active"] = 1.0  # machines act at every micro-step
            infos.append(info)

        rewards *= 0.001

        return obs, rewards, dones, infos

    def _reset_state(self):
        self.current_step = 0
        self.period_index = 0
        self.step_in_period = 0

        self.inventory = np.zeros(self.num_products, dtype=np.float32)
        self.backlog = np.zeros(self.num_products, dtype=np.float32)

        self.ages = np.zeros(self.num_lines, dtype=np.float32)
        self.line_setup = np.full(self.num_lines, -1, dtype=np.int32)

        self.queue = np.zeros(
            (self.num_lines, self.num_products), dtype=np.float32
        )

        self.remaining_capacity = np.zeros(self.num_lines, dtype=np.float32)
        self.line_done = np.zeros(self.num_lines, dtype=bool)
        self.period_produced_per_line = np.zeros(
            (self.num_lines, self.num_products), dtype=np.float32
        )
        self.period_produced_per_product = np.zeros(
            self.num_products, dtype=np.float32
        )
        self.period_setup_costs = np.zeros(self.num_lines, dtype=np.float32)
        self.period_pm_costs = np.zeros(self.num_lines, dtype=np.float32)
        self.period_cm_costs = np.zeros(self.num_lines, dtype=np.float32)
        self.period_queue_sum = np.zeros(self.num_lines, dtype=np.float32)
        self.period_queue_steps = 0

        self.last_inv_cost = 0.0
        self.last_backlog_cost = 0.0
        self.last_inv_qty = 0.0
        self.last_backlog_qty = 0.0
        self.last_manager_horizons = np.zeros(self.num_lines, dtype=np.float32)
        self.last_manager_products = np.full(self.num_lines, -1, dtype=np.int32)
        self.last_manager_masks = np.zeros(
            (self.num_lines, self.num_products), dtype=np.float32
        )
        self.episode_total_cost = 0.0  
        self.last_capacity_per_product = np.zeros(self.num_products, dtype=np.float32)
        self.last_assigned_lines_per_product = np.zeros(self.num_products, dtype=np.float32)
        self.last_unmet_demand_per_product = np.zeros(self.num_products, dtype=np.float32)
        self.last_period_index = 0
        self.last_queue_avg_per_line = np.zeros(self.num_lines, dtype=np.float32)
        self.last_backlog_per_product = np.zeros(self.num_products, dtype=np.float32)
        self.last_inventory_per_product = np.zeros(self.num_products, dtype=np.float32)
        self.last_prod_cost = 0.0
        self.last_prod_cost_per_line = np.zeros(self.num_lines, dtype=np.float32)
        self.last_setup_cost = 0.0
        self.last_setup_cost_per_line = np.zeros(self.num_lines, dtype=np.float32)
        self.last_pm_cost = 0.0
        self.last_cm_cost = 0.0
        self.last_product_service_costs = np.zeros(self.num_products, dtype=np.float32)
        self.last_utilization_total = 0.0
        self.last_utilization_per_line = np.zeros(self.num_lines, dtype=np.float32)
        self.last_allocator_status = ""
        self.last_allocator_objective = 0.0
        self.micro_step_masks = {}
        self.last_period_micro_step_masks = {}

    def _start_new_period(self):
        self.remaining_capacity[:] = self.capacity_per_line
        self.line_done[:] = False
        self.period_produced_per_line.fill(0.0)
        self.period_produced_per_product.fill(0.0)
        self.period_setup_costs.fill(0.0)
        self.period_pm_costs.fill(0.0)
        self.period_cm_costs.fill(0.0)
        self.period_queue_sum.fill(0.0)
        self.period_queue_steps = 0
        self.micro_step_masks = {}
        self.last_manager_masks[:] = 0.0
        # Step 3 inference: pre-load MILP lot sizes so that _build_available_actions()
        # at the end of the previous period (and at reset) sees queue > 0 and marks
        # products as available — without this, machines commit to "end" before
        # _manager_step() runs at step 0 and loads the queue.
        if self.milp_lot_sizes is not None:
            self.queue[:] = 0.0
            period_key = f"period_{self.period_index}"
            if period_key in self.milp_lot_sizes:
                queue_additions = np.array(self.milp_lot_sizes[period_key], dtype=np.float32)
                self.queue[:] = queue_additions * self.line_eligibility

    def _manager_step(self, actions_env):
        masks = self._decode_agent0_action(actions_env[0])
        masks = masks * self.line_eligibility

        # Record per-step mask for Step 2a export (4D allocation)
        self.micro_step_masks[self.step_in_period] = masks.copy()

        if self.milp_lot_sizes is not None:
            # Step 3 inference: queue was pre-loaded in _start_new_period() so
            # _build_available_actions() could see it. Nothing to do here.
            # For all steps: do not touch the queue; machines drain it.
            step_mask = (self.queue > 1e-6).astype(np.float32)
            self.last_manager_masks = np.maximum(self.last_manager_masks, (step_mask > 0.5).astype(np.float32))
            self.last_manager_horizons = np.zeros(self.num_lines, dtype=np.float32)
            self.last_manager_products = np.full(self.num_lines, -1, dtype=np.int32)
            for l in range(self.num_lines):
                active_prods = np.where(self.last_manager_masks[l] > 0.5)[0]
                if len(active_prods) > 0:
                    self.last_manager_products[l] = int(active_prods[0])
                    self.last_manager_horizons[l] = float(self.allocator_lookahead)
            return

        # Training modes: fresh per-step allocation; machines consume immediately.
        self.queue[:] = 0.0
        if self.allocator_mode == "jit":
            queue_additions = self._jit_allocate(masks)
        elif self.allocator_mode == "relaxed_milp":
            allowed_masks = (
                masks.copy()
                if self.relaxed_milp_use_manager_mask
                else self.line_eligibility.copy()
            )
            queue_additions = self._relaxed_milp_allocate(allowed_masks)
            if (
                queue_additions is None
                and self.relaxed_milp_fallback_to_heuristic
            ):
                queue_additions = self._heuristic_allocate(allowed_masks)
            elif queue_additions is None:
                queue_additions = np.zeros_like(self.queue)
        else:
            queue_additions = self._heuristic_allocate(masks)

        self.queue += queue_additions
        step_mask = (queue_additions > 1e-6).astype(np.float32)
        # Union accumulation: track every (line, product) activated across micro-steps
        self.last_manager_masks = np.maximum(self.last_manager_masks, step_mask)

        self.last_manager_horizons = np.zeros(self.num_lines, dtype=np.float32)
        self.last_manager_products = np.full(self.num_lines, -1, dtype=np.int32)
        for l in range(self.num_lines):
            active_prods = np.where(step_mask[l] > 0.5)[0]
            if len(active_prods) > 0:
                self.last_manager_products[l] = int(active_prods[0])
                self.last_manager_horizons[l] = float(self.allocator_lookahead)

    def _jit_allocate(self, masks):
        """Per-micro-step JIT allocator.

        At each step, queues the full outstanding demand for every activated
        (line, product) pair.  The queue is reset before this call, so machines
        consume what they can this step; any unconsumed quota is naturally
        re-queued next step from the updated period_produced_per_product.

        Dividing by remaining_steps is deliberately avoided: it makes the
        per-step quota too small relative to setup overhead, causing machines
        to stall after a single unit of production.  Full-remaining queuing
        lets machines work at full pace whenever the manager activates a slot.
        """
        L, P = self.num_lines, self.num_products
        queue_add = np.zeros((L, P), dtype=np.float32)
        t = self.period_index
        demand = self.demand[t].astype(np.float32)

        # Remaining demand after what has already been produced this period
        remaining_demand = np.maximum(demand - self.period_produced_per_product, 0.0)

        for p in range(P):
            if remaining_demand[p] <= 0.0:
                continue

            active_lines = [
                l for l in range(L)
                if masks[l, p] > 0.5 and self.line_eligibility[l, p] > 0.5
            ]
            # If manager activated no line for this product, fall back to all
            # eligible lines — guarantees every demanded product gets queued so
            # the manager learns routing (which line) not gatekeeping (whether).
            if not active_lines:
                active_lines = [
                    l for l in range(L) if self.line_eligibility[l, p] > 0.5
                ]
            if not active_lines:
                continue

            qty_per_line = remaining_demand[p] / len(active_lines)
            for l in active_lines:
                queue_add[l, p] = qty_per_line

        return queue_add

    def _heuristic_allocate(self, masks):
        L, P = self.num_lines, self.num_products
        queue_add = np.zeros((L, P), dtype=np.float32)
        t = self.period_index

        today_demand = self.demand[t].astype(np.float32)
        raw_need = np.maximum(
            today_demand + self.backlog - self.inventory - np.sum(self.queue, axis=0),
            0.0
        )
        total_qty_target = raw_need.copy()

        prebuild_end = min(t + 2, self.num_periods)
        if prebuild_end > t:
            prebuild_demand = np.sum(
                self.demand[t:prebuild_end], axis=0
            ).astype(np.float32)
        else:
            prebuild_demand = np.zeros(P, dtype=np.float32)

        for p in range(P):
            if total_qty_target[p] <= 0.0 and prebuild_demand[p] > 0.0:
                any_line_active = any(
                    masks[l, p] > 0.5 and self.line_eligibility[l, p] > 0.5
                    for l in range(L)
                )
                if any_line_active:
                    total_qty_target[p] = prebuild_demand[p]

        last_allocated = self.line_setup.copy()

        for l in range(L):
            remaining_cap = float(self.capacity_per_line[l])

            active = [
                p for p in range(P)
                if masks[l, p] > 0.5
                and self.line_eligibility[l, p] > 0.5
                and total_qty_target[p] > 0.0
            ]

            if not active:
                continue

            last_p = int(last_allocated[l])
            if last_p < 0:
                est_total_setup = float(self.first_setup_time[l])
            else:
                est_total_setup = sum(float(self.setup_time_matrix[l, last_p, p]) for p in active) / len(active)
                
            if len(active) > 1:
                avg_intra_setup = np.mean(self.setup_time_matrix[l]) 
                est_total_setup += avg_intra_setup * (len(active) - 1)

            pure_production_cap = max(0.0, remaining_cap - est_total_setup)
            total_active_target = sum(float(total_qty_target[p]) for p in active)

            for p in active:
                target = float(total_qty_target[p])
                proc = float(self.processing_time_matrix[l, p])
                
                allocated_time = pure_production_cap * (target / total_active_target)
                
                max_units_today = allocated_time / proc
                qty = min(target, max_units_today)

                if qty > 0.0:
                    queue_add[l, p] = qty
                    total_qty_target[p] -= qty
                    
            last_allocated[l] = active[-1]

        return queue_add

    def _relaxed_setup_time_candidates(self, line_idx, product_idx, offset):
        candidates = []
        if offset == 0:
            # Use exact known state — do NOT average with hypotheticals
            return [self._estimate_setup_time(line_idx, product_idx)]
        
        # offset > 0: future state unknown, average over possibilities
        candidates.append(float(self.first_setup_time[line_idx]))
        for prev in range(self.num_products):
            if prev == product_idx:
                continue
            if self.line_eligibility[line_idx, prev] > 0.5:
                candidates.append(
                    float(self.setup_time_matrix[line_idx, prev, product_idx])
                )
        return candidates

    def _relaxed_setup_time(self, line_idx, product_idx, offset):
        candidates = np.asarray(
            self._relaxed_setup_time_candidates(line_idx, product_idx, offset),
            dtype=np.float32,
        )
        if candidates.size == 0:
            return 0.0

        mode = self.relaxed_milp_setup_time_mode
        if mode == "worst":
            return float(np.max(candidates))
        if mode == "p95":
            return float(np.percentile(candidates, 95))
        if mode == "p90":
            return float(np.percentile(candidates, 90))
        if mode == "p85":
            return float(np.percentile(candidates, 85))
        if mode == "p75":
            return float(np.percentile(candidates, 75))
        if mode == "mean_std":
            return float(
                np.mean(candidates)
                + self.relaxed_milp_setup_time_std_mult * np.std(candidates)
            )
        return float(np.mean(candidates))

    def _relaxed_setup_cost(self, line_idx, product_idx, offset):
        if offset == 0:
            last_prod = int(self.line_setup[line_idx])
            if last_prod < 0:
                return float(self.first_setup_cost[line_idx])
            if last_prod != product_idx:
                return float(
                    self.setup_cost_matrix[line_idx, last_prod, product_idx]
                )
            return 0.0

        candidates = []
        for prev in range(self.num_products):
            if prev == product_idx:
                continue
            if self.line_eligibility[line_idx, prev] > 0.5:
                candidates.append(
                    float(self.setup_cost_matrix[line_idx, prev, product_idx])
                )
        if candidates:
            return float(np.mean(candidates))
        return float(self.first_setup_cost[line_idx])

    def _relaxed_milp_allocate(self, allowed_masks):
        if pulp is None:
            self.last_allocator_status = "pulp_missing"
            if self.relaxed_milp_fallback_to_heuristic:
                return None
            raise ImportError(
                "PuLP is required for allocator_mode='relaxed_milp'. "
                "Install it with `pip install PuLP`."
            )

        L, P = self.num_lines, self.num_products
        start_t = self.period_index
        window = max(
            1,
            min(
                int(self.relaxed_milp_lookahead),
                self.num_periods - start_t,
            ),
        )
        allowed = (np.asarray(allowed_masks, dtype=np.float32) > 0.5) & (
            self.line_eligibility > 0.5
        )

        prob = pulp.LpProblem(
            f"RelaxedAllocator_{start_t}", pulp.LpMinimize
        )

        x = [
            [
                [
                    pulp.LpVariable(f"x_{k}_{l}_{p}", lowBound=0.0)
                    for p in range(P)
                ]
                for l in range(L)
            ]
            for k in range(window)
        ]
        y = [
            [
                [
                    pulp.LpVariable(f"y_{k}_{l}_{p}", cat="Binary")
                    for p in range(P)
                ]
                for l in range(L)
            ]
            for k in range(window)
        ]
        inv = [
            [
                pulp.LpVariable(f"inv_{k}_{p}", lowBound=0.0)
                for p in range(P)
            ]
            for k in range(window)
        ]
        back = [
            [
                pulp.LpVariable(f"back_{k}_{p}", lowBound=0.0)
                for p in range(P)
            ]
            for k in range(window)
        ]
        has_backlog = [
            [
                pulp.LpVariable(f"has_backlog_{k}_{p}", cat="Binary")
                for p in range(P)
            ]
            for k in range(window)
        ]

        demand_window = self.demand[start_t : start_t + window].astype(np.float32)
        cap_units = np.zeros((L, P), dtype=np.float32)
        for l in range(L):
            for p in range(P):
                proc = float(self.processing_time_matrix[l, p])
                if proc > 0.0 and self.line_eligibility[l, p] > 0.5:
                    cap_units[l, p] = float(self.capacity_per_line[l]) / proc

        product_cap = np.sum(cap_units, axis=0)
        backlog_big_m = (
            np.sum(demand_window, axis=0)
            + self.backlog
            + product_cap * float(window)
            + self.inventory
            + 1.0
        )

        objective = []
        for k in range(window):
            global_t = start_t + k
            for p in range(P):
                prev_inv = self.inventory[p] if k == 0 else inv[k - 1][p]
                prev_back = self.backlog[p] if k == 0 else back[k - 1][p]
                produced = pulp.lpSum(x[k][l][p] for l in range(L))
                prob += (
                    prev_inv
                    + produced
                    - inv[k][p]
                    + back[k][p]
                    == float(self.demand[global_t, p]) + prev_back
                )
                prob += (
                    back[k][p]
                    <= float(backlog_big_m[p]) * has_backlog[k][p]
                )

                objective.append(float(self.holding_cost) * inv[k][p])
                objective.append(float(self.backlog_cost) * back[k][p])
                objective.append(
                    float(self.per_product_backlog_penalty[p])
                    * has_backlog[k][p]
                )

            for l in range(L):
                capacity_terms = []
                for p in range(P):
                    proc = float(self.processing_time_matrix[l, p])
                    if proc <= 0.0 or not bool(allowed[l, p]):
                        prob += x[k][l][p] == 0.0
                        prob += y[k][l][p] == 0.0
                        continue

                    prob += x[k][l][p] <= float(cap_units[l, p]) * y[k][l][p]

                    setup_time = self._relaxed_setup_time(l, p, k)
                    setup_cost = self._relaxed_setup_cost(l, p, k)
                    capacity_terms.append(proc * x[k][l][p])
                    capacity_terms.append(setup_time * y[k][l][p])

                    objective.append(
                        float(self.production_cost_matrix[l, p]) * x[k][l][p]
                    )
                    objective.append(
                        float(self.hazard_rate[l])
                        * float(self.cm_cost[l])
                        * proc
                        * x[k][l][p]
                    )
                    objective.append(setup_cost * y[k][l][p])

                prob += (
                    pulp.lpSum(capacity_terms)
                    <= float(self.capacity_per_line[l])
                    * self.relaxed_milp_capacity_safety
                )

        prob += pulp.lpSum(objective)

        solver = pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=(
                self.relaxed_milp_time_limit
                if self.relaxed_milp_time_limit > 0.0
                else None
            ),
        )
        prob.solve(solver)

        status = pulp.LpStatus.get(prob.status, str(prob.status))
        self.last_allocator_status = status
        objective_value = pulp.value(prob.objective)
        self.last_allocator_objective = (
            float(objective_value) if objective_value is not None else 0.0
        )
        if status != "Optimal":
            return None

        queue_add = np.zeros((L, P), dtype=np.float32)
        for l in range(L):
            for p in range(P):
                val = pulp.value(x[0][l][p])
                if val is not None and val > 1e-6:
                    queue_add[l, p] = float(val)

        return queue_add

    def _estimate_setup_time(self, line_idx, product_idx):
        last_prod = int(self.line_setup[line_idx])
        if last_prod < 0:
            return float(self.first_setup_time[line_idx])
        elif last_prod != product_idx:
            return float(
                self.setup_time_matrix[line_idx, last_prod, product_idx]
            )
        return 0.0

    def _machines_step(self, actions_env, rewards):
        pm_index = self.num_products
        end_index = self.num_products + 1

        for line_idx in range(self.num_lines):
            if self.line_done[line_idx]:
                continue

            agent_id = 1 + line_idx
            a_vec = np.asarray(actions_env[agent_id], dtype=np.float32)
            act_idx = int(np.argmax(a_vec))

            if act_idx == end_index:
                self.line_done[line_idx] = True
                continue

            if act_idx == pm_index:
                pm_cost = (
                    float(self.pm_cost[line_idx])
                    if np.ndim(self.pm_cost) > 0
                    else float(self.pm_cost)
                )
                pm_time = (
                    float(self.pm_time[line_idx])
                    if np.ndim(self.pm_time) > 0
                    else float(self.pm_time)
                )
                if pm_time > 0.0 and self.remaining_capacity[line_idx] > 0.0:
                    used = min(pm_time, self.remaining_capacity[line_idx])
                    self.remaining_capacity[line_idx] -= used
                self.period_pm_costs[line_idx] += pm_cost
                self.ages[line_idx] = 0.0
                rewards[agent_id, 0] -= self.dense_pm_penalty * pm_cost
                continue

            if act_idx < 0 or act_idx >= self.num_products:
                continue

            if self.line_eligibility[line_idx, act_idx] < 0.5:
                continue

            requested_qty = float(self.queue[line_idx, act_idx])
            if requested_qty <= 0.0:
                continue

            last_prod = int(self.line_setup[line_idx])
            setup_time = 0.0
            setup_cost = 0.0
            if last_prod < 0:
                setup_time = float(self.first_setup_time[line_idx])
                setup_cost = float(self.first_setup_cost[line_idx])
            elif last_prod != act_idx:
                setup_time = float(self.setup_time_matrix[line_idx, last_prod, act_idx])
                setup_cost = float(self.setup_cost_matrix[line_idx, last_prod, act_idx])

            proc_time_per_unit = float(
                self.processing_time_matrix[line_idx, act_idx]
            )
            if proc_time_per_unit <= 0.0:
                continue

            available_for_proc = self.remaining_capacity[line_idx] - setup_time
            if available_for_proc <= 0.0:
                continue

            # Remove integer division (//) to keep it as a float
            max_qty_cap = available_for_proc / proc_time_per_unit
            if max_qty_cap <= 0.0:
                continue

            # Compare floats directly
            qty = min(float(requested_qty), float(max_qty_cap))
            if qty <= 1e-5:  # Use a small epsilon instead of 0 to prevent float errors
                continue

            time_used = qty * proc_time_per_unit + setup_time
            self.remaining_capacity[line_idx] = max(
                0.0, self.remaining_capacity[line_idx] - time_used
            )

            self.queue[line_idx, act_idx] = max(
                0.0, self.queue[line_idx, act_idx] - qty
            )
            self.period_produced_per_line[line_idx, act_idx] += qty
            self.period_produced_per_product[act_idx] += qty

            if setup_cost > 0.0 or setup_time > 0.0:
                self.period_setup_costs[line_idx] += setup_cost
                rewards[agent_id, 0] -= (
                    self.dense_setup_penalty * float(setup_cost)
                )
                # Running penalty for Process Agent: forward each setup cost as
                # it fires so its critic learns the allocation → changeover link
                # immediately, not just at period-end via the team reward.
                # Only active in step1 mode; full mode uses a different reward path.
                if self.reward_mode == "step1":
                    rewards[0, 0] -= float(setup_cost)

            cap = float(self.capacity_per_line[line_idx])
            if cap > 0.0:
                time_fraction = (qty * proc_time_per_unit) / cap
                rewards[agent_id, 0] += (
                    self.dense_production_reward * float(time_fraction) * cap
                )

            self.line_setup[line_idx] = act_idx

            delta_age = qty * proc_time_per_unit
            self.ages[line_idx] += delta_age

            expected_failures = float(self.hazard_rate[line_idx]) * delta_age
            step_cm_cost = expected_failures * float(self.cm_cost[line_idx])
            self.period_cm_costs[line_idx] += step_cm_cost
            rewards[agent_id, 0] -= step_cm_cost

            cm_time = (
                float(self.cm_time[line_idx])
                if np.ndim(self.cm_time) > 0
                else float(self.cm_time)
            )
            step_cm_time = expected_failures * cm_time
            if step_cm_time > 0.0:
                self.remaining_capacity[line_idx] = max(
                    0.0, self.remaining_capacity[line_idx] - step_cm_time
                )

            if self.remaining_capacity[line_idx] <= 0.0:
                self.line_done[line_idx] = True

    def _can_process_product(self, line_idx, product_idx):
        if self.line_eligibility[line_idx, product_idx] < 0.5:
            return False

        requested_qty = float(self.queue[line_idx, product_idx])
        if requested_qty <= 0.0:
            return False

        proc_time_per_unit = float(
            self.processing_time_matrix[line_idx, product_idx]
        )
        if proc_time_per_unit <= 0.0:
            return False

        last_prod = int(self.line_setup[line_idx])
        setup_time = 0.0
        if last_prod < 0:
            setup_time = float(self.first_setup_time[line_idx])
        elif last_prod != product_idx:
            setup_time = float(self.setup_time_matrix[line_idx, last_prod, product_idx])

        available_for_proc = self.remaining_capacity[line_idx] - setup_time
        if available_for_proc <= 0.0:
            return False

        max_qty_cap = available_for_proc / proc_time_per_unit
        if max_qty_cap <= 1e-5:
            return False

        return True

    def _can_produce_with_demand(self, line_idx, product_idx):
        """Training-mode availability: demand-based instead of queue-based.

        In the per-micro-step architecture, the queue is reset and refilled at
        the START of each step (inside _manager_step), so it is 0 when
        _build_available_actions() runs at the END of the previous step.
        Using current queue state would permanently block machines.  Instead,
        check whether there is remaining demand and physical capacity.
        """
        if self.line_eligibility[line_idx, product_idx] < 0.5:
            return False

        t = min(self.period_index, self.num_periods - 1)
        remaining = (
            float(self.demand[t, product_idx])
            - float(self.period_produced_per_product[product_idx])
        )
        if remaining <= 1e-3:
            return False

        proc_time_per_unit = float(self.processing_time_matrix[line_idx, product_idx])
        if proc_time_per_unit <= 0.0:
            return False

        last_prod = int(self.line_setup[line_idx])
        setup_time = 0.0
        if last_prod < 0:
            setup_time = float(self.first_setup_time[line_idx])
        elif last_prod != product_idx:
            setup_time = float(self.setup_time_matrix[line_idx, last_prod, product_idx])

        available_for_proc = self.remaining_capacity[line_idx] - setup_time
        if available_for_proc <= 0.0:
            return False

        return available_for_proc / proc_time_per_unit > 1e-5

    def _line_available_actions(self, line_idx):
        pm_index = self.num_products
        end_index = self.num_products + 1
        mask = np.zeros(self.num_products + 2, dtype=np.float32)

        if self.line_done[line_idx]:
            mask[end_index] = 1.0
            return mask

        pm_time_l = float(self.pm_time[line_idx]) if np.ndim(self.pm_time) > 0 else float(self.pm_time)
        if self.remaining_capacity[line_idx] >= max(pm_time_l, 1e-6):
            mask[pm_index] = 1.0

        # In training modes (no pre-loaded MILP lot sizes) the queue is 0 when
        # this is called because _manager_step refills it at the START of the
        # NEXT step.  Use demand-based feasibility so machines are not blocked.
        use_demand_check = self.milp_lot_sizes is None

        can_work = False
        for product_idx in range(self.num_products):
            ok = (
                self._can_produce_with_demand(line_idx, product_idx)
                if use_demand_check
                else self._can_process_product(line_idx, product_idx)
            )
            if ok:
                mask[product_idx] = 1.0
                can_work = True

        if not can_work:
            mask[end_index] = 1.0

        return mask

    def _build_available_actions(self):
        if self.num_lines > 0 and self.num_products > 0:
            masks = []
            for line_idx in range(self.num_lines):
                for prod_idx in range(self.num_products):
                    if (
                        self.allocator_mode == "relaxed_milp"
                        and not self.relaxed_milp_use_manager_mask
                    ):
                        masks.append(np.array([1.0, 0.0], dtype=np.float32))
                    elif self.line_eligibility[line_idx, prod_idx] > 0.5:
                        masks.append(np.array([1.0, 1.0], dtype=np.float32))
                    else:
                        masks.append(np.array([1.0, 0.0], dtype=np.float32))
            manager_mask = np.concatenate(masks, axis=0)
        else:
            manager_mask = np.zeros(0, dtype=np.float32)
        available_actions = [manager_mask]
        for line_idx in range(self.num_lines):
            available_actions.append(self._line_available_actions(line_idx))
        return available_actions

    def _period_effectively_over(self):
        return bool(np.all(self.line_done | (self.remaining_capacity <= 0.0)))

    def _end_period(self):
        self.last_period_index = self.period_index
        # Persist per-step masks before _start_new_period clears micro_step_masks
        self.last_period_micro_step_masks = {k: v.copy() for k, v in self.micro_step_masks.items()}

        inv_cost, backlog_cost, inv_costs, backlog_costs = self._update_inventory_and_backlog(
            self.period_produced_per_product
        )
        self.last_inv_cost = float(inv_cost)
        self.last_backlog_cost = float(backlog_cost)
        self.last_inv_qty = float(np.sum(self.inventory))
        self.last_backlog_qty = float(np.sum(self.backlog))
        self.last_backlog_per_product = self.backlog.astype(np.float32).copy()
        self.last_inventory_per_product = self.inventory.astype(np.float32).copy()
        self.last_unmet_demand_per_product = self.backlog.astype(np.float32).copy()

        cap_units = np.zeros(self.num_products, dtype=np.float32)
        for p in range(self.num_products):
            mask = (self.line_eligibility[:, p] > 0.5) & (
                self.processing_time_matrix[:, p] > 0.0
            )
            if np.any(mask):
                cap_units[p] = float(
                    np.sum(
                        self.capacity_per_line[mask]
                        / self.processing_time_matrix[mask, p]
                    )
                )
        self.last_capacity_per_product = cap_units

        assigned = np.zeros(self.num_products, dtype=np.float32)
        for p in range(self.num_products):
            assigned[p] = float(np.sum(self.last_manager_masks[:, p]))
        self.last_assigned_lines_per_product = assigned

        if self.period_queue_steps > 0:
            self.last_queue_avg_per_line = (
                self.period_queue_sum / float(self.period_queue_steps)
            ).astype(np.float32)
        else:
            self.last_queue_avg_per_line = np.zeros(
                self.num_lines, dtype=np.float32
            )

        setup_cost_total = float(np.sum(self.period_setup_costs))
        pm_cost_total = float(np.sum(self.period_pm_costs))
        expected_cm_cost_total = float(np.sum(self.period_cm_costs))

        prod_cost_total = float(
            np.sum(self.period_produced_per_line * self.production_cost_matrix)
        )
        prod_cost_per_line = np.sum(
            self.period_produced_per_line * self.production_cost_matrix, axis=1
        ).astype(np.float32)

        runtime_per_line = np.sum(
            self.period_produced_per_line * self.processing_time_matrix, axis=1
        )
        cap = self.capacity_per_line.astype(np.float32)
        util_per_line = np.zeros(self.num_lines, dtype=np.float32)
        for i in range(self.num_lines):
            if cap[i] > 0.0:
                util_per_line[i] = float(runtime_per_line[i] / cap[i])
        util_total = float(
            np.sum(runtime_per_line) / max(1e-6, float(np.sum(cap)))
        )

        self.last_prod_cost = prod_cost_total
        self.last_prod_cost_per_line = prod_cost_per_line
        self.last_setup_cost = setup_cost_total
        self.last_setup_cost_per_line = self.period_setup_costs.astype(np.float32).copy()
        self.last_pm_cost = pm_cost_total
        self.last_cm_cost = expected_cm_cost_total
        self.last_utilization_per_line = util_per_line
        self.last_utilization_total = util_total

        manager_direct_costs = inv_cost + backlog_cost + prod_cost_total
        worker_total_direct_costs = setup_cost_total + pm_cost_total + expected_cm_cost_total

        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)

        if self.reward_mode == "step1":
            # Proportional backlog penalty: scales with unmet demand so agents always
            # benefit from producing more. Wipe backlog so it does not accumulate
            # across periods (lot sizing is MILP's job in Step 2).
            unmet = float(np.sum(self.backlog))
            if unmet > 1e-3:
                rewards[:, 0] -= self.kill_switch_penalty * unmet
                if not self.is_eval:
                    self.backlog[:] = 0.0
                    self.last_backlog_qty = 0.0
                    self.last_backlog_per_product[:] = 0.0
                    self.last_unmet_demand_per_product[:] = 0.0

            # Team reward always applied: -(proc_time + setup + PM + CM).
            # Covers structural decisions (allocation, sequencing, maintenance).
            # Given alongside the proportional backlog penalty above so agents
            # learn efficiency even when demand is not fully met.
            total_proc_time = float(
                np.sum(self.period_produced_per_line * self.processing_time_matrix)
            )
            team = -(total_proc_time + float(worker_total_direct_costs))

            # Process Agent shaping: penalise over-activation and load imbalance.
            if self.activation_penalty > 0.0:
                num_activated = float(np.sum(self.last_manager_masks))
                team -= self.activation_penalty * num_activated
            if self.load_balance_penalty > 0.0:
                setup_std = float(np.std(self.period_setup_costs))
                team -= self.load_balance_penalty * setup_std

            rewards[:, 0] += team
        else:
            rewards[0, 0] = -float(manager_direct_costs) + self.alpha_cost_weight * (
                -float(worker_total_direct_costs)
            )

            if self.activation_penalty > 0.0:
                num_activated = float(np.sum(self.last_manager_masks))
                rewards[0, 0] -= self.activation_penalty * num_activated

            if self.load_balance_penalty > 0.0:
                setup_std = float(np.std(self.period_setup_costs))
                rewards[0, 0] -= self.load_balance_penalty * setup_std

            rewards[1:, 0] = 0.0

        beta = float(self.machine_service_cost_share_beta)
        if beta > 0.0 and self.num_lines > 0:
            include_inv = bool(self.machine_service_cost_share_include_inventory)
            include_backlog = bool(self.machine_service_cost_share_include_backlog)
            if include_inv or include_backlog:
                service_costs = np.zeros(self.num_products, dtype=np.float32)
                if include_inv:
                    service_costs += inv_costs.astype(np.float32)
                if include_backlog:
                    service_costs += backlog_costs.astype(np.float32)

                mode = str(self.machine_service_cost_share_mode).lower()
                weights = np.zeros((self.num_lines, self.num_products), dtype=np.float32)

                if mode == "production":
                    weights = self.period_produced_per_line.astype(np.float32).copy()
                elif mode == "queue":
                    weights = np.maximum(self.queue.astype(np.float32), 0.0)
                else:
                    weights = self.last_manager_masks.astype(np.float32).copy()

                denom = np.sum(weights, axis=0) 
                for p in range(self.num_products):
                    cost_p = float(service_costs[p])
                    if cost_p <= 0.0:
                        continue
                    d = float(denom[p])
                    if d <= 1e-8:
                        share = cost_p / float(self.num_lines)
                        for l in range(self.num_lines):
                            rewards[1 + l, 0] -= beta * share
                        continue
                    for l in range(self.num_lines):
                        w = float(weights[l, p])
                        if w <= 0.0:
                            continue
                        rewards[1 + l, 0] -= beta * (w / d) * cost_p

        return rewards

    def _update_inventory_and_backlog(self, produced):
        t = min(self.period_index, self.num_periods - 1)
        period_demand = self.demand[t]

        inv_costs = np.zeros(self.num_products, dtype=np.float32)
        backlog_costs = np.zeros(self.num_products, dtype=np.float32)

        for p in range(self.num_products):
            total_demand = period_demand[p] + self.backlog[p]
            total_supply = self.inventory[p] + produced[p]

            if total_supply >= total_demand:
                self.inventory[p] = total_supply - total_demand
                self.backlog[p] = 0.0
            else:
                self.inventory[p] = 0.0
                self.backlog[p] = total_demand - total_supply

            inv_costs[p] = self.holding_cost * self.inventory[p]
            backlog_costs[p] = self.backlog_cost * self.backlog[p]
            if self.backlog[p] > 0:
                backlog_costs[p] += self.per_product_backlog_penalty[p]
        
        self.last_product_service_costs = inv_costs + backlog_costs
        inv_cost = float(np.sum(inv_costs))
        backlog_cost = float(np.sum(backlog_costs))
        
        return inv_cost, backlog_cost, inv_costs, backlog_costs

    def _decode_agent0_action(self, action_vec):
        one_hot = np.asarray(action_vec, dtype=np.float32).ravel()

        expected_len = self.num_lines * self.num_products * 2

        if one_hot.size != expected_len:
            raise ValueError(
                f"Agent 0 action length {one_hot.size} does not match "
                f"expected {expected_len} for "
                f"{self.num_lines}x{self.num_products} binary mask."
            )

        masks = np.zeros(
            (self.num_lines, self.num_products), dtype=np.float32
        )
        offset = 0
        for l in range(self.num_lines):
            for p in range(self.num_products):
                seg = one_hot[offset : offset + 2]
                masks[l, p] = float(np.argmax(seg))
                offset += 2

        return masks

    def _build_observations(self):
        remaining_periods = self.num_periods - self.period_index
        binary_mode = (self.obs_mode == "binary")

        if binary_mode:
            # Quantity-blind: agents see WHAT needs to be done, not HOW MANY units.
            # Continuous quantities are replaced with binary (present/absent) signals.
            # The reward function still uses raw quantities — agents feel the
            # consequences (kill switch, setup costs) without seeing the magnitudes.
            inv = (self.inventory > 0).astype(np.float32)
            back = (self.backlog > 0).astype(np.float32)
            queue_total = (np.sum(self.queue, axis=0) > 0).astype(np.float32)
        else:
            inv = np.log1p(self.inventory.astype(np.float32))
            back = np.log1p(self.backlog.astype(np.float32))
            queue_total = np.log1p(np.sum(self.queue, axis=0).astype(np.float32))

        queue_segment_len = self.num_lines * self.num_products
        coverage = (self.queue > 0).astype(np.float32).sum(axis=0)
        if self.num_lines > 0:
            coverage /= float(self.num_lines)
        demand_window = np.zeros(
            (self.lookahead_days, self.num_products), dtype=np.float32
        )
        for d in range(self.lookahead_days):
            target_idx = self.period_index + d
            if target_idx < self.num_periods:
                if binary_mode:
                    demand_window[d] = (self.demand[target_idx] > 0).astype(np.float32)
                else:
                    demand_window[d] = np.log1p(self.demand[target_idx])

        contention = np.zeros(self.num_lines, dtype=np.float32)
        if self.num_periods > 0:
            t_idx = min(self.period_index, self.num_periods - 1)
            for l in range(self.num_lines):
                competing = 0
                for p in range(self.num_products):
                    if (
                        self.line_eligibility[l, p] > 0.5
                        and self.demand[t_idx, p] > 0
                    ):
                        competing += 1
                if self.num_products > 0:
                    contention[l] = float(competing) / float(self.num_products)

        scarcity = np.zeros(self.num_products, dtype=np.float32)
        one_period_cap = np.zeros(self.num_products, dtype=np.float32)
        
        for p in range(self.num_products):
            n_eligible = np.sum(self.line_eligibility[:, p])
            if n_eligible > 0:
                scarcity[p] = 1.0 / float(n_eligible)
                
            for l in range(self.num_lines):
                if self.line_eligibility[l, p] > 0.5:
                    proc = float(self.processing_time_matrix[l, p])
                    if proc > 0.0:
                        one_period_cap[p] += float(self.capacity_per_line[l]) / proc

        lines_needed = np.zeros(self.num_products, dtype=np.float32)
        for p in range(self.num_products):
            if one_period_cap[p] > 0:
                lines_needed[p] = min(
                    self.backlog[p] / one_period_cap[p],
                    float(self.num_periods)
                )

        urgency = np.zeros(self.num_products, dtype=np.float32)
        for p in range(self.num_products):
            for d in range(self.lookahead_days):
                idx = self.period_index + d
                if idx < self.num_periods and self.demand[idx, p] > 0:
                    urgency[p] = 1.0 / float(d + 1)
                    break

        line_availability = np.ones(self.num_lines, dtype=np.float32)

        line_setup_oh = np.zeros(
            (self.num_lines, self.num_products), dtype=np.float32
        )
        for l in range(self.num_lines):
            idx = int(self.line_setup[l])
            if 0 <= idx < self.num_products:
                line_setup_oh[l, idx] = 1.0
        line_setup_flat = line_setup_oh.reshape(-1)

        ages = self.ages.astype(np.float32)

        obs_all = []
        for agent_id in range(self.num_agents):
            vec = np.zeros(self.obs_dim, dtype=np.float32)
            pos = 0

            vec[pos : pos + self.num_products] = inv
            pos += self.num_products

            vec[pos : pos + self.num_products] = back
            pos += self.num_products

            if agent_id == 0:
                if binary_mode:
                    queue_vec = (self.queue > 0).astype(np.float32).reshape(-1)
                else:
                    queue_vec = np.log1p(self.queue.reshape(-1).astype(np.float32))
            else:
                queue_vec = np.zeros(queue_segment_len, dtype=np.float32)
                line_idx = agent_id - 1
                if 0 <= line_idx < self.num_lines:
                    start = line_idx * self.num_products
                    end = start + self.num_products
                    if binary_mode:
                        queue_vec[start:end] = (self.queue[line_idx] > 0).astype(np.float32)
                    else:
                        queue_vec[start:end] = np.log1p(self.queue[line_idx].astype(np.float32))
            vec[pos : pos + queue_segment_len] = queue_vec
            pos += queue_segment_len

            if agent_id == 0:
                vec[pos : pos + self.num_products] = coverage
            pos += self.num_products

            if agent_id == 0:
                vec[pos : pos + self.num_products] = queue_total
            pos += self.num_products

            if agent_id == 0:
                raw_demand_next = np.sum(self.demand[self.period_index : self.period_index + self.lookahead_days], axis=0)
                raw_shortfall = raw_demand_next + self.backlog - self.inventory - np.sum(self.queue, axis=0)
                if binary_mode:
                    shortfall = (raw_shortfall > 0).astype(np.float32)
                else:
                    shortfall = np.log1p(np.maximum(raw_shortfall, 0.0))
                vec[pos : pos + self.num_products] = shortfall.astype(np.float32)
            pos += self.num_products

            window_flat = demand_window.reshape(-1)
            vec[pos : pos + window_flat.size] = window_flat
            pos += window_flat.size

            vec[pos] = float(remaining_periods)
            pos += 1

            vec[pos : pos + self.num_lines] = line_availability
            pos += self.num_lines

            vec[pos : pos + self.num_lines * self.num_products] = line_setup_flat
            pos += self.num_lines * self.num_products

            vec[pos : pos + self.num_lines] = ages
            pos += self.num_lines

            line_id_oh = np.zeros(self.num_lines, dtype=np.float32)
            if agent_id > 0:
                line_idx = agent_id - 1
                if 0 <= line_idx < self.num_lines:
                    line_id_oh[line_idx] = 1.0
            vec[pos : pos + self.num_lines] = line_id_oh
            pos += self.num_lines

            if agent_id == 0:
                vec[pos : pos + self.num_lines] = contention
            pos += self.num_lines

            if agent_id == 0:
                vec[pos : pos + self.num_products] = urgency
            pos += self.num_products

            if agent_id == 0:
                vec[pos : pos + self.num_lines * self.num_products] = (
                    self.line_eligibility.reshape(-1).astype(np.float32)
                )
            pos += self.num_lines * self.num_products

            if agent_id == 0:
                vec[pos : pos + self.num_products] = scarcity
            pos += self.num_products

            if agent_id == 0:
                vec[pos : pos + self.num_products] = lines_needed
            pos += self.num_products

            obs_all.append(vec)

        return np.asarray(obs_all, dtype=np.float32)

    def render(self, mode="human"):
        return None

    def close(self):
        return None