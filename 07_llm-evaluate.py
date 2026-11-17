import multiprocessing
import os
import argparse
import csv
import json
import numpy as np
import time
import pickle
import pyscipopt as scip
from pathlib import Path
from multiprocessing import get_context, Queue
from queue import Empty

svmrank = None  # Optional competitor dependency.

import utilities
from experiment_config import (
    DEFAULT_INSTANCE_LIMIT,
    DEFAULT_INSTANCE_SIZE,
    RESULT_FIELDNAMES,
    SEEDS,
    TIME_LIMIT,
    build_branching_policies,
    build_instances,
    load_ml_competitor_policies,
)
from experiment_results import (
    build_csv_row,
    build_summary_item,
    collect_solver_metrics,
    print_solver_metrics,
)
from utilities import RuleSwitchController
from utilities import LLMQueryScheduler
from experience_index import (
    ExperienceConfig,
    HierarchicalExperienceIndex,
    compute_reward,
    compute_reward_components,
    experience_path_for_problem,
)

from llm_tools.prompter import build_initial_branching_prompt, build_dynamic_branching_prompt
from llm_tools.llm_parser import parse_llm_output
from llm_tools.llm_agent import llmClient
from llm_tools.rules import FALLBACK_RULE, normalize_executable_rule
from signal import signal, SIGPIPE, SIG_IGN
signal(SIGPIPE, SIG_IGN)

from llm_settings import load_llm_settings
_llm_settings = load_llm_settings()
API_KEY = _llm_settings['api_key']
MODEL_NAME = _llm_settings['model']
API_URL = _llm_settings['api_url']


def guard_executable_rule(rule, fallback=FALLBACK_RULE):
    guarded = normalize_executable_rule(rule, fallback=fallback)
    if guarded != rule:
        print(f"[Rule guard] '{rule}' is not in safe executable rules; using '{guarded}'", flush=True)
    return guarded

class PolicyBranching(scip.Branchrule):

    def __init__(self, policy, instance_path, problem_type="unknown", time_limit=None,
                 problem_info=None):
        super().__init__()
        self.policy_type = policy['type']
        self.policy_name = policy['name']
        self.instance_path = instance_path  #这个path好像没用到
        self.problem_type = problem_type
        self.time_limit = time_limit
        self.problem_info = problem_info or {}
        self.llm_async = True
        self.total_llm_query_time = 0.0
        self.record = True
        self.exp_config = ExperienceConfig()
        self.experience_index = None
        self.experience_path = None
        self.scheduling_log_path = None
        self.last_llm_query_step = -10**9
        self.pending_episodes = []
        self.completed_episodes_since_save = 0
        if self.record: self.step_records = []

        if self.policy_type == 'internal':
            self.policy = policy['name']

        elif self.policy_type == 'ml-competitor':
            self.policy = policy['model']

            # feature parameterization
            self.feat_shift = policy['feat_shift']
            self.feat_scale = policy['feat_scale']
            self.feat_specs = policy['feat_specs']

        elif self.policy_type == 'llm' or self.policy_type == 'llm-dynamic':
            self.policy = policy['name']
            self.llm_client = policy['model']

        else:
            raise NotImplementedError

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.llm_step_counter = 0
        self.state_buffer = {}
        self.khalil_root_buffer = {}
        self.step_records = []
        self.scheduler = LLMQueryScheduler(win=50, N_min=50, N_max=1000)
        self.scheduler.reset()
        self.pending_episodes = []
        self.completed_episodes_since_save = 0
        self.last_llm_query_step = -10**9
        self.max_depth_seen = 0
        self.static_state_info = None
        self.experience_initialized = False

    def _ensure_llm_worker(self):
        """按需启动LLM后台进程（仅在 self.llm_async=True 时）。"""
        if not getattr(self, 'llm_async', False):
            return
        if getattr(self, '_llm_worker', None) and self._llm_worker.is_alive():
            return
        ctx = get_context('spawn')   # 跨平台更稳
        self._llm_req_q = ctx.Queue(maxsize=1)
        self._llm_resp_q = ctx.Queue(maxsize=1)
        self._llm_time_q = ctx.Queue(maxsize=1)
        self._llm_worker = ctx.Process(
            target=_llm_worker_process,
            args=(self._llm_req_q, self._llm_resp_q, self._llm_time_q),
            daemon=True
        )
        self._llm_worker.start()

    def _stop_llm_worker(self):
        if getattr(self, '_llm_worker', None) and self._llm_worker.is_alive():
            try:
                self._llm_req_q.put_nowait(None)  # 发送停止信号
            except Exception:
                pass
            self._llm_req_q.close()
            self._llm_req_q.join_thread()
            self._llm_worker.join(timeout=1.0)
            if self._llm_worker.is_alive():
                self._llm_worker.terminate()
            self._llm_worker = None

    def _apply_llm_decision_if_any(self):
        """非阻塞读取最新建议并应用。"""
        if not getattr(self, 'llm_async', False):
            return
        if not hasattr(self, '_llm_resp_q'):
            return
        try:
            parsed = self._llm_resp_q.get_nowait()
            # 同时获取查询时间
            try:
                query_time = self._llm_time_q.get_nowait()
                self.total_llm_query_time += query_time
            except Empty:
                pass
        except Empty:
            return

        if isinstance(parsed, dict):
            new_rule = guard_executable_rule(parsed.get("branching_rule", 'relpscost'), fallback=self.llm_current_rule)
            if new_rule and new_rule != self.llm_current_rule:
                print(self.llm_current_rule)
                print(f"[LLM-DYNAMIC] Step {self.llm_step_counter} Switching rule {self.llm_current_rule} -> {new_rule}")
                self.llm_current_rule = new_rule

    def _init_experience_memory_if_needed(self):
        """Lazy initialization for the scheduling memory.

        PySCIPOpt attaches ``self.model`` after branchrule registration, so the
        experience index is initialized lazily inside the callback rather than
        in ``__init__``. This method deliberately avoids taking a full state
        snapshot on the first callback because older SCIP/PySCIPOpt builds can
        crash when several state getters are called at that point.
        """
        if self.policy_type != 'llm-dynamic' or self.experience_initialized:
            return

        self.llm_state_history = []
        self.llm_current_rule = self.policy
        self.llm_start_step = 20
        self.llm_last_decision_step = -1
        static_info = self._collect_static_state_info()
        self.switch_ctrl = RuleSwitchController(
            num_vars=static_info.get("num_variables") or self.model.getNVars(),
            num_constraints=static_info.get("num_constraints") or self.model.getNConss(),
            consecutive_required=2,
            cooldown_factor=0.5,
            stagnation_factor=0.6,
            min_gap_improve=1e-3,
        )

        self.experience_path = experience_path_for_problem(self.problem_type)
        self.experience_index = HierarchicalExperienceIndex.load(
            self.experience_path,
            problem_type=self.problem_type,
            config=self.exp_config,
        )
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        safe_name = Path(self.instance_path).stem if self.instance_path else "unknown_instance"
        self.scheduling_log_path = Path("experiment_results") / f"rule_scheduling_{safe_name}_{timestamp}.jsonl"
        self.scheduling_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Do not collect B&B state or create the first episode from this initial
        # callback. SCIP 6.0.1/PySCIPOpt can segfault when several state getters
        # are called before the first branch has fully returned to SCIP. The
        # first ordinary checkpoint will record the active rule after the tree
        # has advanced to a callback-safe point.
        self.last_llm_query_step = self.llm_step_counter
        self.experience_initialized = True

    def _collect_static_state_info(self):
        """Collect static problem features once per solve.

        This method avoids traversing variables/constraints from inside the
        SCIP callback. Some PySCIPOpt/SCIP builds are fragile at that boundary,
        so the main loop passes a pre-optimization snapshot whenever possible.
        """
        if self.static_state_info is not None:
            return self.static_state_info

        info = self.problem_info or {}
        num_vars = info.get("total_number_of_variables")
        num_constraints = info.get("total_number_of_constraints")
        num_binary = info.get("number_of_binary_variables")
        avg_nnz = info.get("average_nonzeros_per_constraint")

        if num_vars is None:
            num_vars = self.model.getNVars()
        if num_constraints is None:
            num_constraints = self.model.getNConss()
        if num_binary is None:
            num_binary = 0

        binary_ratio = num_binary / float(num_vars) if num_vars > 0 else None

        num_nonzeros = None
        matrix_density = None
        if avg_nnz is not None and num_constraints is not None:
            num_nonzeros = float(avg_nnz) * float(num_constraints)
        denom = float(num_vars * num_constraints)
        matrix_density = (num_nonzeros / denom) if num_nonzeros is not None and denom > 0 else None

        self.static_state_info = {
            "problem_type": self.problem_type,
            "num_variables": num_vars,
            "num_constraints": num_constraints,
            "num_nonzeros": num_nonzeros,
            "binary_variable_ratio": binary_ratio,
            "matrix_density": matrix_density,
        }
        return self.static_state_info

    def _collect_state_summary(self):
        """Return a callback-safe dict snapshot for indexing/reward/logging.

        Keep this deliberately conservative. In the deployed SCIP 6.0.1
        environment, calling richer getters such as getCurrentNode(), getGap(),
        getPrimalbound(), getDualbound(), or getPseudoBranchCands() from inside
        branchexeclp can trigger delayed C-layer segfaults. The first stable
        version of the experience index uses only counters that are already
        used elsewhere in the original experiment code.
        """
        state = dict(self._collect_static_state_info())

        current_depth = None
        if self.exp_config.enable_callback_depth_metrics:
            try:
                node = self.model.getCurrentNode()
                current_depth = node.getDepth() if node is not None else None
                if current_depth is not None:
                    self.max_depth_seen = max(getattr(self, "max_depth_seen", 0), current_depth)
            except Exception:
                current_depth = None

        try:
            nnodes = self.model.getNNodes()
        except Exception:
            nnodes = self.llm_step_counter

        try:
            nlps = self.model.getNLPs()
        except Exception:
            nlps = None

        try:
            solving_time = self.model.getSolvingTime()
        except Exception:
            solving_time = None

        gap = None
        if self.exp_config.enable_callback_gap_metrics:
            try:
                gap = self.model.getGap()
            except Exception:
                gap = None

        primal_bound = None
        dual_bound = None
        if self.exp_config.enable_callback_bound_metrics:
            try:
                primal_bound = self.model.getPrimalbound()
            except Exception:
                primal_bound = None
            try:
                dual_bound = self.model.getDualbound()
            except Exception:
                dual_bound = None

        elapsed_ratio = None
        if solving_time is not None and self.time_limit:
            elapsed_ratio = solving_time / max(float(self.time_limit), 1e-9)

        state.update({
            "explored_nodes": nnodes,
            "current_depth": current_depth,
            "max_depth": getattr(self, "max_depth_seen", None) if current_depth is not None else None,
            "current_gap": gap,
            "gap": gap,
            "elapsed_time": solving_time,
            "elapsed_time_ratio": elapsed_ratio,
            "current_active_rule": getattr(self, "llm_current_rule", None) or self.policy,
            "nnodes": nnodes,
            "nlps": nlps,
            "stime": solving_time,
            "primal_bound": primal_bound,
            "dual_bound": dual_bound,
            "ndomchgs": self.ndomchgs,
            "ncutoffs": self.ncutoffs,
        })
        return state

    def _collect_final_state_summary(self):
        """Collect richer metrics after optimize() has returned.

        These getters are intentionally kept out of branchexeclp because the
        target SCIP/PySCIPOpt stack was observed to segfault when some of them
        were called from the callback. At finalization time SCIP is no longer in
        the branching callback, so the snapshot can safely improve reward
        attribution for episodes that are still pending when the solve ends.
        """
        state = dict(self._collect_static_state_info())

        def safe_get(getter, default=None):
            try:
                value = getter()
                return default if value is None else value
            except Exception:
                return default

        nnodes = safe_get(self.model.getNNodes, self.llm_step_counter)
        nlps = safe_get(self.model.getNLPs, None)
        solving_time = safe_get(self.model.getSolvingTime, None)
        gap = safe_get(self.model.getGap, None)
        primal_bound = safe_get(self.model.getPrimalbound, None)
        dual_bound = safe_get(self.model.getDualbound, None)
        status = safe_get(self.model.getStatus, None)

        elapsed_ratio = None
        if solving_time is not None and self.time_limit:
            elapsed_ratio = solving_time / max(float(self.time_limit), 1e-9)

        state.update({
            "explored_nodes": nnodes,
            "current_depth": None,
            "max_depth": getattr(self, "max_depth_seen", None),
            "current_gap": gap,
            "gap": gap,
            "elapsed_time": solving_time,
            "elapsed_time_ratio": elapsed_ratio,
            "current_active_rule": getattr(self, "llm_current_rule", None) or self.policy,
            "nnodes": nnodes,
            "nlps": nlps,
            "stime": solving_time,
            "primal_bound": primal_bound,
            "dual_bound": dual_bound,
            "status": status,
            "ndomchgs": self.ndomchgs,
            "ncutoffs": self.ncutoffs,
        })
        return state

    def _start_episode(self, leaf_key, state, selected_rule, current_active_rule, source,
                       switched, llm_recommended_rule=None, guard_rejected=False,
                       llm_throttled=False):
        """Create one non-overlapping decision episode.

        The episode stores the start leaf and start metrics because the B&B
        state at settlement time may fall into a different leaf. Reward is
        attributed to the decision context, not the later context.
        """
        self.pending_episodes.append({
            "start_leaf": tuple(leaf_key),
            "start_state": dict(state),
            "selected_rule": selected_rule,
            "executed_rule": selected_rule,
            "current_active_rule": current_active_rule,
            "source": source,
            "start_step": self.llm_step_counter,
            "end_step": self.llm_step_counter + self.exp_config.episode_window,
            "start_time": state.get("stime"),
            "start_metrics": dict(state),
            "switched": bool(switched),
            "llm_recommended_rule": llm_recommended_rule,
            "guard_rejected": bool(guard_rejected),
            "llm_throttled": bool(llm_throttled),
        })

    def _settle_due_episodes(self, force=False, current_state=None):
        if not self.experience_index:
            return
        if not self.pending_episodes:
            return

        current_state = current_state or self._collect_state_summary()
        end_leaf = self.experience_index.build_leaf_key(current_state)
        remaining = []
        for episode in self.pending_episodes:
            due = self.llm_step_counter >= episode["end_step"]
            if not due and not force:
                remaining.append(episode)
                continue

            truncated = not due
            weight = self.exp_config.truncated_weight if truncated else 1.0
            components = compute_reward_components(
                episode["start_metrics"],
                current_state,
                config=self.exp_config,
            )
            reward = compute_reward(episode["start_metrics"], current_state, config=self.exp_config)
            entry = self.experience_index.update_rule_reward(
                episode["start_leaf"],
                episode["selected_rule"],
                reward=reward,
                weight=weight,
                components=components,
            )
            if tuple(end_leaf) != tuple(episode["start_leaf"]):
                self.experience_index.record_passive_exposure(end_leaf)

            self.completed_episodes_since_save += 1
            saved = False
            if self.completed_episodes_since_save >= self.exp_config.save_every_episodes:
                self._save_experience_index()
                saved = True

            self._log_scheduling_event({
                "event": "episode_complete",
                "step": self.llm_step_counter,
                "leaf_key": episode["start_leaf"],
                "end_leaf_key": end_leaf,
                "source": episode["source"],
                "current_active_rule": episode["current_active_rule"],
                "selected_rule": episode["selected_rule"],
                "executed_rule": episode["executed_rule"],
                "switched": episode["switched"],
                "guard_rejected": episode["guard_rejected"],
                "llm_throttled": episode["llm_throttled"],
                "truncated": truncated,
                "reward": reward,
                "reward_components": components,
                "mean_reward_after": entry.get("mean_reward"),
                "n_effective_after": entry.get("n_effective"),
                "saved_index": saved,
            })

        self.pending_episodes = remaining

    def _run_rule_scheduling_checkpoint(self):
        """Select the next active rule using UCB first, then synchronous LLM."""
        if self.policy_type != 'llm-dynamic' or not self.experience_index:
            return
        if self.llm_step_counter <= 0:
            return
        if self.llm_step_counter % self.exp_config.checkpoint_interval != 0:
            return

        state = self._collect_state_summary()
        leaf_key = self.experience_index.build_leaf_key(state)
        current_rule = self.llm_current_rule
        decision = self.experience_index.select_rule(leaf_key, current_active_rule=current_rule)

        source = "keep"
        selected_rule = current_rule
        llm_recommended_rule = None
        llm_throttled = False
        parsed = None

        if decision.confident:
            source = "ucb"
            selected_rule = decision.rule
        else:
            since_llm = self.llm_step_counter - self.last_llm_query_step
            if since_llm >= self.exp_config.llm_min_interval:
                source = "llm"
                selected_rule, parsed = self._query_llm_for_rule()
                llm_recommended_rule = selected_rule
                self.last_llm_query_step = self.llm_step_counter
            else:
                llm_throttled = True

        selected_rule = guard_executable_rule(selected_rule or current_rule, fallback=current_rule)
        guard_rejected = False
        switched = False
        executed_rule = selected_rule

        if selected_rule != current_rule:
            if source == "llm":
                self.experience_index.record_recommendation(
                    leaf_key,
                    selected_rule,
                    source="llm",
                    adopted=False,
                )
            allow, target = self.switch_ctrl.allow_switch(
                step=self.llm_step_counter,
                current_rule=current_rule,
                llm_next_rule=selected_rule,
                llm_switch_flag=True,
            )
            if allow and target != current_rule:
                self.llm_current_rule = target
                executed_rule = target
                switched = True
                self.switch_ctrl.commit_switch(self.llm_step_counter, target)
                if source == "ucb":
                    self.experience_index.record_recommendation(leaf_key, target, source="ucb", adopted=True)
                elif source == "llm":
                    self.experience_index.record_adoption(leaf_key, target)
            else:
                guard_rejected = True
                executed_rule = current_rule
                selected_rule = current_rule
                source = "keep"
                self.experience_index.record_recommendation(leaf_key, current_rule, source="keep", adopted=True)
        else:
            self.experience_index.record_recommendation(leaf_key, current_rule, source=source, adopted=True)

        self._start_episode(
            leaf_key=leaf_key,
            state=state,
            selected_rule=executed_rule,
            current_active_rule=current_rule,
            source=source,
            switched=switched,
            llm_recommended_rule=llm_recommended_rule,
            guard_rejected=guard_rejected,
            llm_throttled=llm_throttled,
        )
        self._log_scheduling_event({
            "event": "decision",
            "step": self.llm_step_counter,
            "leaf_key": leaf_key,
            "source": source,
            "ucb_reason": decision.reason,
            "ucb_scores": decision.scores,
            "llm_response": parsed,
            "current_active_rule": current_rule,
            "selected_rule": selected_rule,
            "executed_rule": executed_rule,
            "switched": switched,
            "guard_rejected": guard_rejected,
            "llm_throttled": llm_throttled,
        })

    def _query_llm_for_rule(self):
        """Synchronous fallback used only when local experience is uncertain."""
        start = time.perf_counter()
        prompt = build_dynamic_branching_prompt(self.llm_state_history, max_len=10)
        llm_response = self.llm_client.get_content(self.llm_client.getResponse(prompt))
        parsed = parse_llm_output(llm_response)
        self.total_llm_query_time += time.perf_counter() - start
        if isinstance(parsed, dict):
            if "next_rule" in parsed:
                if not bool(parsed.get("switch", False)):
                    return self.llm_current_rule, parsed
                return guard_executable_rule(parsed.get("next_rule", self.llm_current_rule), fallback=self.llm_current_rule), parsed
            if "branching_rule" in parsed:
                return guard_executable_rule(parsed.get("branching_rule", self.llm_current_rule), fallback=self.llm_current_rule), parsed
        return self.llm_current_rule, parsed

    def _after_branching_step(self):
        if self.policy_type != 'llm-dynamic':
            return
        state = self._collect_state_summary()
        explored_nodes = state.get("explored_nodes") or 0
        cutoff_ratio = self.ncutoffs / float(1 + max(explored_nodes, 0))
        domchg_ratio = self.ndomchgs / float(1 + max(explored_nodes, 0))
        tree_state = {
            # Keep prompt history string-only. Some older prompter versions use
            # string concatenation instead of f-string formatting and fail on
            # floats with "must be str, not float".
            "depth": "-",
            "nodes_explored": str(explored_nodes),
            "gap": "-",
            "cutoff_ratio": "{:.6f}".format(cutoff_ratio),
            "domchg_ratio": "{:.6f}".format(domchg_ratio),
            "num_branch_candidates": "-",
            "entropy": "0.000000",
        }
        self.llm_state_history.append(tree_state)
        if state.get("gap") is not None:
            self.switch_ctrl.update_progress(self.llm_step_counter, state.get("gap"))
        self._settle_due_episodes(force=False)
        self._run_rule_scheduling_checkpoint()

    def _save_experience_index(self):
        if self.experience_index and self.experience_path:
            self.experience_index.save(self.experience_path)
            self.completed_episodes_since_save = 0

    def _flush_experience_memory(self, final_state=None):
        if self.policy_type != 'llm-dynamic' or not self.experience_index:
            return
        self._settle_due_episodes(force=True, current_state=final_state)
        self._save_experience_index()

    def _log_scheduling_event(self, event):
        if not self.scheduling_log_path:
            return
        payload = dict(event)
        payload.update({
            "problem_type": self.problem_type,
            "instance_path": self.instance_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        # JSON has no tuple type; lists make leaf keys readable in JSONL.
        for key in ("leaf_key", "end_leaf_key"):
            if key in payload and isinstance(payload[key], tuple):
                payload[key] = list(payload[key])
        with self.scheduling_log_path.open("a") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

    def _log_scheduling_error(self, exc):
        self._log_scheduling_event({
            "event": "scheduling_error",
            "step": self.llm_step_counter,
            "current_active_rule": getattr(self, "llm_current_rule", None) or self.policy,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })

    def record_final_state(self):
        final_state = None
        if self.policy_type == 'llm-dynamic':
            final_state = self._collect_final_state_summary()
            self._flush_experience_memory(final_state=final_state)
        # 这里必须先pop，求解过程中记录的最后一步的gap不为0
        if self.step_records:
            self.step_records.pop()
        if final_state is None:
            final_state = {
                "nnodes": self.model.getNNodes(),
                "nlps": self.model.getNLPs(),
                "stime": self.model.getSolvingTime(),
                "gap": self.model.getGap() if self.model.getGap() is not None else -1,
                "status": self.model.getStatus(),
            }
        final_record = {
            'step': self.llm_step_counter,
            'current_policy': getattr(self, "llm_current_rule", None) or self.policy,
            'nnodes': final_state.get("nnodes"),
            'nlps': final_state.get("nlps"),
            'stime': final_state.get("stime"),
            'gap': final_state.get("gap") if final_state.get("gap") is not None else -1,
            'status': final_state.get("status"),
            'ndomchgs': self.ndomchgs,
            'ncutoffs': self.ncutoffs
        }
        if not self.step_records or self.step_records[-1]['step'] != self.llm_step_counter:
            self.step_records.append(final_record)

    def branchexeclp(self, allowaddcons):
        # SCIP internal branching rule
        if self.policy_type == 'internal':
            result = self.model.executeBranchRule(self.policy, allowaddcons)
        elif self.policy_type == 'llm':
            result = self.model.executeBranchRule(self.policy, allowaddcons)
        elif self.policy_type == 'llm-dynamic':
            self._init_experience_memory_if_needed()
            result = self.model.executeBranchRule(self.llm_current_rule, allowaddcons)

        # custom ml-based branching
        else:
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]
            # print(candidate_mask)

            # initialize root buffer for Khalil features extraction
            if self.model.getNNodes() == 1 \
                    and self.policy_type == 'ml-competitor' \
                    and self.feat_specs['type'] in ('khalil', 'all'):
                utilities.extract_khalil_variable_features(self.model, [], self.khalil_root_buffer)

            if len(candidate_vars) == 1:
                best_var = candidate_vars[0]

            elif self.policy_type == 'ml-competitor':

                # build candidate features
                candidate_states = []
                if self.feat_specs['type'] in ('all', 'gcnn_agg'):
                    state = utilities.extract_state(self.model, self.state_buffer)
                    candidate_states.append(utilities.compute_extended_variable_features(state, candidate_mask))
                if self.feat_specs['type'] in ('all', 'khalil'):
                    candidate_states.append(utilities.extract_khalil_variable_features(self.model, candidate_vars, self.khalil_root_buffer))
                candidate_states = np.concatenate(candidate_states, axis=1)

                # feature preprocessing
                candidate_states = utilities.preprocess_variable_features(candidate_states, self.feat_specs['augment'], self.feat_specs['qbnorm'])

                # feature normalization
                candidate_states =  (candidate_states - self.feat_shift) / self.feat_scale

                candidate_scores = self.policy.predict(candidate_states)
                best_var = candidate_vars[candidate_scores.argmax()]

            else:
                raise NotImplementedError
            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED

        self.llm_step_counter += 1
        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1
        if self.policy_type == 'llm-dynamic':
            try:
                self._after_branching_step()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._log_scheduling_error(e)
                print("[LLM-DYNAMIC] scheduling step failed; keep current rule", flush=True)
        if self.record:
            self.step_records.append({
            'step': self.llm_step_counter,
            'current_policy': getattr(self, "llm_current_rule", None) or self.policy,
            'nnodes': self.model.getNNodes(),
            'nlps': self.model.getNLPs(),
            'stime': self.model.getSolvingTime(),
            'gap': self.model.getGap() if self.model.getGap() is not None else -1,
            'status': self.model.getStatus(),
            'ndomchgs': self.ndomchgs,
            'ncutoffs': self.ncutoffs
        })

        return {'result': result}

def select_branching_rule_llm(model, llm_client, problem_type, problem_info=None):  #, time_tracker=None
    """
    使用 LLM 根据问题属性选择分支策略
    :param model: 已读入的问题模型
    :param problem_info: dict，包括问题维度、约束类型、变量类型等信息
    :param llm_client: 实现 get_response(prompt: str) -> str 的 LLM 客户端
    :return: str, 分支规则名称
    """

    problem_info = problem_info or utilities.extract_problem_info(model)
    prompt = build_initial_branching_prompt(problem_type, problem_info)

    try:
        start = time.perf_counter()
        result = llm_client.infer_rule(prompt, parse_fn=parse_llm_output, fallback="relpscost", parallel=True)
        query_time = time.perf_counter() - start
        result = result["branching_rule"]
        result = guard_executable_rule(result, fallback="relpscost")
        print('llm recommended branching rule and inference time:',result, query_time)
    except Exception as e:
        print(f"[LLM fallback] Failed to get rule: {e}")
        result = "relpscost"

    return result

def _llm_worker_process(req_q: Queue, resp_q: Queue, time_q: Queue):
        """子进程：阻塞读请求 -> 构造prompt -> 调LLM -> 回写解析结果。"""
        try:
            llm_client = llmClient(api_key=API_KEY, model_name=MODEL_NAME, api_url=API_URL)
        except Exception:
            llm_client = None
        while True:
            req = req_q.get()  # 阻塞等待
            if req is None:    # 停止信号
                break
            try:
                prompt = build_dynamic_branching_prompt(req['history'], max_len=10)
                if llm_client is None:
                    parsed = {}
                    query_time = 0.0
                else:
                    query_start = time.perf_counter()
                    # parsed = llm_client.infer_rule(prompt, parse_fn=parse_llm_output, fallback="relpscost", parallel=True)
                    parsed = {'branching_rule': 'relpscost'}
                    query_time = time.perf_counter() - query_start
                    print('result:',parsed)
                if not isinstance(parsed, dict):
                    parsed = {"branching_rule": 'relpscost'}
                while True:
                    try:
                        resp_q.get_nowait()
                    except Empty:
                        break
                resp_q.put(parsed)
                while True:
                    try:
                        time_q.get_nowait()
                    except Empty:
                        break
                time_q.put(query_time)
            except Exception as e:
                while True:
                    try:
                        resp_q.get_nowait()
                    except Empty:
                        break
                print('errir and use default')
                resp_q.put({"branching_rule": 'relpscost'})
                while True:
                    try:
                        time_q.get_nowait()
                    except Empty:
                        break
                time_q.put(0.0)

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-a', '--asyn',
        help='asyn. llm.',
        type=bool,
        default=True,
    )
    parser.add_argument(
        '-s', '--size',
        help='Benchmark scale to process.',
        choices=['small', 'medium', 'big'],
        default=DEFAULT_INSTANCE_SIZE,
    )
    parser.add_argument(
        '-l', '--limit',
        help='Maximum number of instances to process from the selected scale.',
        type=int,
        default=DEFAULT_INSTANCE_LIMIT,
    )
    parser.add_argument('--baseline-only', action='store_true', help='Run internal rules without API calls.')
    parser.add_argument('--time-limit', type=float, default=TIME_LIMIT, help='Per-solve SCIP seconds.')
    args = parser.parse_args()
    if args.time_limit <= 0:
        parser.error('--time-limit must be positive')
    TIME_LIMIT = args.time_limit
    if not args.baseline_only and not all((API_KEY, MODEL_NAME, API_URL)):
        parser.error('Set LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME or use --baseline-only')


    result_file = f"{args.problem}_{args.size}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    llm_agent = llmClient(api_key=API_KEY, model_name=MODEL_NAME, api_url=API_URL)
    results = []
    instances = build_instances(args.problem, size=args.size, limit=args.limit)
    branching_policies = build_branching_policies(llm_agent, seeds=SEEDS)

    print(f"problem: {args.problem}")
    print(f"size: {args.size}")
    print(f"instance limit: {args.limit}")
    print(f"gpu: {args.gpu}")
    print(f"time limit: {TIME_LIMIT} s")

    if args.baseline_only:
        branching_policies = [p for p in branching_policies if p['type'] == 'internal']
    if any(p['type'] == 'ml-competitor' and p['name'].startswith('svmrank') for p in branching_policies):
        import svmrank
    load_ml_competitor_policies(branching_policies, svmrank, pickle)
    print("running SCIP...")
    for output_dir in ('results', 'experiment_results', 'experience_indexes'):
        os.makedirs(output_dir, exist_ok=True)
    with open(f"results/{result_file}", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            for policy in branching_policies:
                m = scip.Model()
                m.setIntParam('display/verblevel', 0)
                print(instance['path'])
                m.readProblem(f"{instance['path']}")
                utilities.init_scip_params(m, seed=policy['seed'])
                problem_info = utilities.extract_problem_info(m)
                if policy['type'] in ['llm', 'llm-dynamic']:
                    policy['name'] = select_branching_rule_llm(
                        m,
                        llm_agent,
                        args.problem,
                        problem_info=problem_info,
                    )#, time_tracker=brancher
                    print('policy_name:', policy['name'])
                m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
                m.setRealParam('limits/time', TIME_LIMIT)
                brancher = PolicyBranching(
                    policy,
                    instance['path'],
                    problem_type=args.problem,
                    time_limit=TIME_LIMIT,
                    problem_info=problem_info,
                )
                if args.asyn and policy['type'] != 'llm-dynamic':
                    brancher._ensure_llm_worker()
                m.includeBranchrule(
                    branchrule=brancher,
                    name=f"{policy['type']}:{policy['name']}",
                    desc=f"Custom PySCIPOpt branching policy.",
                    priority=666666, maxdepth=-1, maxbounddist=1)

                walltime = time.perf_counter()
                proctime = time.process_time()

                m.optimize()
                brancher._stop_llm_worker()
                walltime = time.perf_counter() - walltime
                proctime = time.process_time() - proctime

                # 关键：补充记录求解结束时的最终状态
                if brancher.record: brancher.record_final_state()

                metrics = collect_solver_metrics(m, brancher, walltime, proctime)
                print_solver_metrics(policy, instance, metrics)
                writer.writerow(build_csv_row(policy, instance, metrics))
                csvfile.flush()
                results.append(build_summary_item(policy, instance, metrics))

                if brancher.record:
                    utilities.record_step_result(args.problem, instance['path'], brancher.step_records)

                m.freeProb()
                time.sleep(1)
        summary = utilities.result_summary(results, csvfile)
        print(summary)
