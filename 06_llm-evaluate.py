import imp
import multiprocessing
import os
import sys
import importlib
import argparse
import csv
import numpy as np
import time
import random
import pickle
import json
import pdb
import pyscipopt as scip
import threading, queue
from multiprocessing import get_context, Queue, Process
from queue import Empty
# from nlp_reader import read_lp_file

import svmrank

import utilities
from utilities import RuleSwitchController
from utilities import LLMQueryScheduler

from llm_tools.prompter import build_initial_branching_prompt, build_dynamic_branching_prompt
# from llm_tools.request_agent import OPLLMAgent
from llm_tools.llm_parser import parse_llm_reply, parse_llm_output
from llm_tools.llm_agent import llmClient
import llm_config
from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)

#国内API接口
API_KEY = "sk-ctoowrguzjggztifabtbbonkhjtyprzesyrsuwdxyoffpfyz"
MODEL_NAME = "Qwen/Qwen3-8B" #"Qwen/Qwen3-30B-A3B, Qwen/Qwen3-Next-80B-A3B-Instruct, deepseek-ai/DeepSeek-V3.1"#, THUDM/GLM-Z1-32B-0414,  Qwen/Qwen3-30B-A3B
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

#国外API接口
API_KEY_ABO = "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"
MODEL_NAME_ABO = "qwen/qwq-32b:free, anthropic/claude-3.5-sonnet, google/gemini-2.5-flash, deepseek/deepseek-chat-v3.1, qwen/qwen3-30b-a3b-thinking-2507"#"anthropic/claude-3.5-sonnet" #"anthropic/claude-3.5-sonnet" #"google/gemini-2.5-flash" "openai/gpt-4o", "anthropic/claude-3.5-sonnet", "x-ai/grok-3", "deepseek/deepseek-chat"     
API_URL_ABO = "https://openrouter.ai/api/v1/chat/completions"

class PolicyBranching(scip.Branchrule):

    def __init__(self, policy, instance_path):
        super().__init__()
        self.policy_type = policy['type']
        self.policy_name = policy['name']
        self.instance_path = instance_path  #这个path好像没用到
        # self._llm_idx_queue = []
        self.llm_async = True
        self.total_llm_query_time = 0.0
        # self._req_q = None
        # self._resp_q = None
        # self._worker = None
        # self._stop_evt = None
        self.record = True
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
            # self.policy = policy['model']
            # self.llm_parser = LLMReplyParser()

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
        # self.total_llm_query_time = 0.0  # total accumulate query time (seconds)

    # import threading, queue

    # def _ensure_llm_worker(self):
    #     if not getattr(self, 'llm_async', False):
    #         return
    #     if getattr(self, '_llm_worker', None) and self._llm_worker.is_alive():
    #         return
    #     ctx = get_context('spawn')
    #     # 用 SimpleQueue：没有 feeder 线程，关闭时更干净
    #     self._llm_req_q = ctx.SimpleQueue()
    #     self._llm_resp_q = ctx.SimpleQueue()
    #     self._llm_worker = ctx.Process(
    #         target=_llm_worker_process,         # 顶层函数
    #         args=(self._llm_req_q, self._llm_resp_q),
    #         daemon=False                        # 关键：不要 daemon
    #     )
    #     self._llm_worker.start()


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
        # api_key  = getattr(llm_config, 'API_KEY', None)
        # base_url = getattr(llm_config, 'BASE_URL', None)
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

    # def _stop_llm_worker(self):
    #     p = getattr(self, '_llm_worker', None)
    #     if not p:
    #         return
    #     try:
    #         # 发停止信号
    #         self._llm_req_q.put(None)
    #     except Exception:
    #         pass

    #     p.join(timeout=2.0)
    #     if p.is_alive():
    #         p.terminate()
    #         p.join(timeout=1.0)

    #     # 显式关闭队列端，避免后续 put 写到已关闭的管道
    #     for q in (getattr(self, '_llm_req_q', None), getattr(self, '_llm_resp_q', None)):
    #         try:
    #             q.close()
    #         except Exception:
    #             pass

    #    self._llm_worker = None

    # ---- 3) 非阻塞应用最新决策（用 empty()+get() 代替 get_nowait） ----
    # def _apply_llm_decision_if_any(self):
    #     if not getattr(self, 'llm_async', False):
    #         return
    #     if not hasattr(self, '_llm_resp_q'):
    #         return
    #     try:
    #         if self._llm_resp_q.empty():   # SimpleQueue 没有 get_nowait，只能先判空
    #             return
    #         parsed = self._llm_resp_q.get()
    #         self._llm_req_inflight = False
    #     except Exception:
    #         # 读取失败，清除在途标记，避免饿死
    #         self._llm_req_inflight = False
    #         return

    #     if isinstance(parsed, dict) and parsed.get("switch", False):
    #         new_rule = parsed.get("next_rule") or 'relpscost'
    #         if new_rule != self.llm_current_rule:
    #             print(f"[LLM-DYNAMIC] Step {self.llm_step_counter} Switching rule "
    #                 f"{self.llm_current_rule} -> {new_rule}")
    #             self.llm_current_rule = new_rule

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
            # new_rule = parsed.get("next_rule", 'relpscost')
            new_rule = parsed.get("branching_rule", 'relpscost') 
            if new_rule and new_rule != self.llm_current_rule:
                print(self.llm_current_rule)
                print(f"[LLM-DYNAMIC] Step {self.llm_step_counter} Switching rule {self.llm_current_rule} -> {new_rule}")
                self.llm_current_rule = new_rule

    def record_final_state(self):
        # 这里必须先pop，求解过程中记录的最后一步的gap不为0
        self.step_records.pop()
        final_record = {
            'step': self.llm_step_counter,
            'current_policy': getattr(self, "llm_current_rule", None) or self.policy,
            'nnodes': self.model.getNNodes(),
            'nlps': self.model.getNLPs(),
            'stime': self.model.getSolvingTime(),
            'gap': self.model.getGap() if self.model.getGap() is not None else -1,
            'status': self.model.getStatus(),
            'ndomchgs': self.ndomchgs,
            'ncutoffs': self.ncutoffs
        }
        if not self.step_records or self.step_records[-1]['step'] != self.llm_step_counter:
            self.step_records.append(final_record)

        # llm_switch = bool(parsed.get("switch", False))
        # llm_next_rule = parsed.get("next_rule", self.llm_current_rule)
        # allow, target = self.switch_ctrl.allow_switch(
        #     step=self.llm_step_counter,
        #     current_rule=self.llm_current_rule,
        #     llm_next_rule=llm_next_rule,
        #     llm_switch_flag=llm_switch
        # )
        # if allow and target != self.llm_current_rule:
        #     print(f"[LLM-DYNAMIC] Step {self.llm_step_counter}: Switching rule from {self.llm_current_rule} to {target} "
        #             f"(min_hold={self.switch_ctrl.min_hold}, cooldown={self.switch_ctrl.cooldown})")
        #     self.llm_current_rule = target
        #     self.switch_ctrl.commit_switch(self.llm_step_counter, target)

          
    # def _llm_worker_loop(self):
    #     """后台线程：消费状态→调用LLM→产出建议。不可访问 self.model！"""
    #     while not self._stop_evt.is_set():
    #         try:
    #             req = self._req_q.get(timeout=0.1)  # {'step': int, 'history': list}
    #         except queue.Empty:
    #             continue
    #         try:
    #             prompt = build_dynamic_branching_prompt(req['history'], max_len=10)
    #             # —— 这里调用你现有的同步 LLM 客户端 ——
    #             llm_response = self.llm_client.get_content(self.llm_client.getResponse(prompt))
    #             parsed = parse_llm_output(llm_response)  # 期望 dict: {"switch":..., "next_rule":..., "correction":...}
    #             # 只保留最新响应
    #             while not self._resp_q.empty():
    #                 try: self._resp_q.get_nowait()
    #                 except queue.Empty: break
    #             self._resp_q.put(parsed, timeout=0.01)
    #         except Exception as e:
    #             # 静默失败，不影响主循环
    #             pass

    # def _try_apply_llm_decision(self):
    #     """非阻塞读取一条最新建议并应用。"""
    #     if not self.llm_async or not self._resp_q:
    #         return
    #     try:
    #         parsed = self._resp_q.get_nowait()
    #     except queue.Empty:
    #         return
    #     if isinstance(parsed, dict) and parsed.get("switch", False):
    #         new_rule = parsed.get("next_rule")
    #         if new_rule and new_rule != self.llm_current_rule:
    #             print(f"[LLM-DYNAMIC] Switching rule from {self.llm_current_rule} to {new_rule}")
    #             self.llm_current_rule = new_rule
    #         elif parsed.get("correction", False):
    #             print(f"[LLM-DYNAMIC] Previous rule {self.llm_current_rule} seems ineffective.")


    # def branchfree(self):
        # self._stop_llm_worker()
        # if self._stop_evt: self._stop_evt.set()
        # if self._worker and self._worker.is_alive(): self._worker.join(timeout=0.5)

    def branchexeclp(self, allowaddcons):
        # SCIP internal branching rule
        if self.policy_type == 'internal':
            result = self.model.executeBranchRule(self.policy, allowaddcons)
        elif self.policy_type == 'llm':
            result = self.model.executeBranchRule(self.policy, allowaddcons)
        elif self.policy_type == 'llm-dynamic':
            # 初始化用于动态调度的参数
            if not hasattr(self, 'llm_state_history'):
                self.llm_state_history = []
                self.llm_step_counter = 0            # 记录分枝次数
                self.llm_current_rule = self.policy  # 初始 rule，比如 'relpscost'
                self.llm_start_step = 20             # 初始积累 多少 步后开始询问 LLM
                self.llm_interval = 5000                # 每隔 多少 步判断一次是否切换
                self.llm_last_decision_step = -1
                # self._ensure_llm_worker()
                # if self.llm_async: self._ensure_llm_worker()
                self.switch_ctrl = RuleSwitchController(
                    num_vars=self.model.getNVars(),
                    num_constraints=self.model.getNConss(),
                    consecutive_required=2,
                    cooldown_factor=0.5,
                    stagnation_factor=0.6,
                    min_gap_improve=1e-3,
                )
            # time_start = time.time()
            
            # 2) 用当前 rule 执行一次分支（不被 LLM 阻塞）
            result = self.model.executeBranchRule(self.llm_current_rule, allowaddcons)
            # print('time point 2:',time.time()-time_start)
            # 3) 收集状态并更新计数
            tree_state = utilities.collect_tree_state(self.model, self.ndomchgs, self.ncutoffs)
            self.llm_state_history.append(tree_state)
            _ = self.scheduler.sample(self.model, self.llm_step_counter)
            # self.llm_step_counter += 1
            self.switch_ctrl.update_progress(self.llm_step_counter, tree_state.get("gap", None))
            # print('time point 3:',time.time()-time_start)
            # 4) 判断是否触发一次“策略评估”
            # should_query_llm = (
            #     self.llm_step_counter >= self.llm_start_step and
            #     (self.llm_step_counter - self.llm_last_decision_step >= self.llm_interval)
            # )

            # if should_query_llm:
            # if False:
            if self.scheduler.should_query(self.llm_step_counter, self.llm_last_decision_step, self.llm_start_step):
                self.llm_last_decision_step = self.llm_step_counter
                if self.llm_async:
                    # --- 异步：仅投递请求，主流程不等待 ---
                    try:
                        # 丢弃旧请求，仅保留最新窗口
                        while True:
                            self._llm_req_q.get_nowait()
                    except Empty:
                        pass
                    try:
                        self._llm_req_q.put_nowait({'history': list(self.llm_state_history)})
                    except Exception:
                        pass
                else:
                    # --- 同步：按原逻辑阻塞调用 ---
                    try:
                        llm_query_start = time.perf_counter() # start record the LLM query
                        prompt = build_dynamic_branching_prompt(self.llm_state_history, max_len=10)
                        llm_response = self.llm_client.get_content(self.llm_client.getResponse(prompt))
                        parsed = parse_llm_output(llm_response)
                        # record the accumulative query time
                        llm_query_time = time.perf_counter() - llm_query_start
                        self.total_llm_query_time += llm_query_time


                        if isinstance(parsed, dict):
                            new_rule = parsed.get("next_rule")
                            llm_switch = bool(parsed.get("switch", False))
                            new_rule = parsed.get("next_rule", self.llm_current_rule)

                            # 经过“防抖控制器”裁决
                            allow, target = self.switch_ctrl.allow_switch(
                                step=self.llm_step_counter,
                                current_rule=self.llm_current_rule,
                                llm_next_rule=new_rule,
                                llm_switch_flag=llm_switch
                            )

                            if allow and target != self.llm_current_rule:
                                print(f"[LLM-DYNAMIC] Step {self.llm_step_counter}: Switching rule from {self.llm_current_rule} to {target} "
                                        f"(min_hold={self.switch_ctrl.min_hold}, cooldown={self.switch_ctrl.cooldown})")
                                self.llm_current_rule = target
                                self.switch_ctrl.commit_switch(self.llm_step_counter, target)
                            elif parsed.get("correction", False):
                                print(f"[LLM-DYNAMIC] Step {self.llm_step_counter}: Previous rule '{self.llm_current_rule}' seems ineffective.")

                            # if new_rule and new_rule != self.llm_current_rule:
                            #     print(f"[LLM-DYNAMIC] Step {self.llm_step_counter}: Switching rule from {self.llm_current_rule} to {new_rule}")
                            #     self.llm_current_rule = new_rule
                            # elif parsed.get("correction", False):
                            #     print(f"[LLM-DYNAMIC] Step {self.llm_step_counter}: Previous rule switch to {self.llm_current_rule} was ineffective.")
                    except Exception as e:
                        print(f"[LLM-DYNAMIC] LLM decision failed: {e}")

            # 4) 异步模式：非阻塞应用“已准备好”的最新建议
            if self.llm_async:
                self._apply_llm_decision_if_any()

            # if should_query_llm:
            #     self.llm_last_decision_step = self.llm_step_counter

            #     if self.llm_async:
            #         # —— 异步：把状态窗口丢给后台线程，主循环不等待 ——
            #         req = {'step': self.llm_step_counter, 'history': self.llm_state_history[-10:]}
            #         if self._req_q:
            #             # 始终保留最新请求，旧的清掉
            #             try:
            #                 while not self._req_q.empty():
            #                     self._req_q.get_nowait()
            #             except queue.Empty:
            #                 pass
            #             try:
            #                 self._req_q.put_nowait(req)
            #             except queue.Full:
            #                 pass
            #         print('time point 4:',time.time()-time_start)
            #     else:
            #         # —— 同步：按原逻辑串行调用 LLM（会产生等待）——
            #         prompt = build_dynamic_branching_prompt(self.llm_state_history, max_len=10)
            #         try:
            #             time_start = time.time()
            #             llm_response = self.llm_client.get_content(self.llm_client.getResponse(prompt))
            #             print('LLM query time-consume:',time.time()-time_start)
            #             parsed = parse_llm_output(llm_response)
            #             if isinstance(parsed, dict) and parsed.get("switch", False):
            #                 new_rule = parsed.get("next_rule")
            #                 if new_rule and new_rule != self.llm_current_rule:
            #                     print(f"[LLM-DYNAMIC] Step {self.llm_step_counter}: Switching rule from {self.llm_current_rule} to {new_rule}")
            #                     self.llm_current_rule = new_rule
            #                 elif parsed.get("correction", False):
            #                     print(f"[LLM-DYNAMIC] Step {self.llm_step_counter}: Previous rule switch to {self.llm_current_rule} was ineffective.")
            #         except Exception as e:
            #             print(f"[LLM-DYNAMIC] LLM decision failed: {e}")
            #         print('time point 5:',time.time()-time_start)
            
            
            
            # # 使用当前策略运行分支
            # result = self.model.executeBranchRule(self.llm_current_rule, allowaddcons)

            # # 收集树结构状态
            # tree_state = utilities.collect_tree_state(self.model, self.ndomchgs, self.ncutoffs)

            # self.llm_state_history.append(tree_state)
            # self.llm_step_counter += 1

            # # --- 是否触发 LLM 判断 ---
            # should_query_llm = (
            #     self.llm_step_counter >= self.llm_start_step and
            #     (self.llm_step_counter - self.llm_last_decision_step >= self.llm_interval)
            # )

            # if should_query_llm:
            #     self.llm_last_decision_step = self.llm_step_counter

            #     prompt = build_dynamic_branching_prompt(self.llm_state_history, max_len=10)

            #     try:
            #         llm_response = self.llm_client.get_content(self.llm_client.getResponse(prompt))
            #         parsed = parse_llm_output(llm_response)
            #         # messages = self.llm_agent.compose_messages(['user'], [prompt])
            #         # reply = self.llm_agent.get_reply(messages, model="qwen-turbo-latest", temperature=0.0)
            #         # parsed = self.llm_parser.parse_llm_reply(reply)[1]  # Expecting a dict
            #         if isinstance(parsed, dict) and parsed.get("switch", False):
            #             new_rule = parsed.get("next_rule")
            #             if new_rule and new_rule != self.llm_current_rule:
            #                 print(f"[LLM-DYNAMIC] Step {self.llm_step_counter}: Switching rule from {self.llm_current_rule} to {new_rule}")
            #                 self.llm_current_rule = new_rule
            #             elif parsed.get("correction", False):
            #                 print(f"[LLM-DYNAMIC] Step {self.llm_step_counter}: Previous rule switch to {self.llm_current_rule} was ineffective.")
            #     except Exception as e:
            #         print(f"[LLM-DYNAMIC] LLM decision failed: {e}")

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

            # elif self.policy_type == 'llm':
            #     result = self.model.executeBranchRule(self.policy, allowaddcons)

                # cand_feats = utilities.extract_variable_features_for_llm(
                #     self.model, candidate_vars, self.khalil_root_buffer
                # )  # shape: (num_candidates, num_features)
                # TopK = min(10, len(candidate_vars)) # 5 can be change
                # prompt = build_prompt(cand_feats, self.model.getNNodes() == 1, k=TopK)

                # # 3) Choose the variable in self._llm_idx_queue
                # chosen_idx = None
                # while len(self._llm_idx_queue) > 0 and not (0 <= self._llm_idx_queue[0] < len(candidate_vars)):
                #     self._llm_idx_queue.pop(0)
                # if len(self._llm_idx_queue) > 0:
                #     chosen_idx = self._llm_idx_queue.pop(0)
                #     best_var = candidate_vars[chosen_idx]
                # if len(self._llm_idx_queue) == 0:
                #     try:
                #         print("querying LLM ....")
                #         llm_response = self.policy.get_content(self.policy.getResponse(prompt))
                #         _, res = self.llm_parser.parse_llm_reply(llm_response)
                #         self._llm_idx_queue.extend(int(idx) for idx in res if idx >=0 and idx < len(candidate_vars))
                #         print(self._llm_idx_queue,len(candidate_vars))
                #         # pdb.set_trace()
                #         while len(self._llm_idx_queue) > 0 and not (0 <= self._llm_idx_queue[0] < len(candidate_vars)):
                #             self._llm_idx_queue.pop(0)
                #         if len(self._llm_idx_queue) > 0: chosen_idx = self._llm_idx_queue.pop(0)
                #         best_var = candidate_vars[chosen_idx]
                #     except Exception as e:
                #         print(f"Error getting LLM reply: {e}")
                #         self._llm_idx_queue = random.sample(range(len(candidate_vars)), k=min(TopK, len(candidate_vars)))
                #         while len(self._llm_idx_queue) > 0 and not (0 <= self._llm_idx_queue[0] < len(candidate_vars)):
                #             self._llm_idx_queue.pop(0)
                #         if len(self._llm_idx_queue) > 0: chosen_idx = self._llm_idx_queue.pop(0)

            else:
                raise NotImplementedError
            # print(best_var)
            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED

        self.llm_step_counter += 1
        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1
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

def select_branching_rule_llm(model, llm_client, problem_type):  #, time_tracker=None
    """
    使用 LLM 根据问题属性选择分支策略
    :param model: 已读入的问题模型
    :param problem_info: dict，包括问题维度、约束类型、变量类型等信息
    :param llm_client: 实现 get_response(prompt: str) -> str 的 LLM 客户端
    :return: str, 分支规则名称
    """

    # 简单的 cache 防止重复调用
    # cache_key = (str(problem_info), depth)
    # if cache_key in cache:
    #     return cache[cache_key]
    problem_info = utilities.extract_problem_info(model)
    prompt = build_initial_branching_prompt(problem_type, problem_info)

    try:
        # llm_response = llm_client.get_content(llm_client.getResponse(prompt))
        start = time.perf_counter()
        result = llm_client.infer_rule(prompt, parse_fn=parse_llm_output, fallback="relpscost", parallel=True)
        query_time = time.perf_counter() - start
        # print('initial rule selecting time:', query_time)
        # if time_tracker is not None:
        #     time_tracker.total_llm_query_time += query_time
        result = result["branching_rule"]
        print('llm recommended branching rule and inference time:',result, query_time)
        # result = json.loads(response)
        # rule = result.get("branching_rule", "relpscost")
    except Exception as e:
        print(f"[LLM fallback] Failed to get rule: {e}")
        result = "relpscost"

    # cache[cache_key] = rule
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
                # 在子进程里构造 prompt，避免主进程重负载
                prompt = build_dynamic_branching_prompt(req['history'], max_len=10)
                if llm_client is None:
                    parsed = {}
                    query_time = 0.0
                else:
                    # start recording
                    query_start = time.perf_counter()
                    # parsed = llm_client.infer_rule(prompt, parse_fn=parse_llm_output, fallback="relpscost", parallel=True)
                    parsed = {'branching_rule': 'relpscost'}
                    query_time = time.perf_counter() - query_start #ending recording
                    print('result:',parsed)
                    # llm_response = llm_client.get_content(llm_client.getResponse(prompt))
                    # parsed = parse_llm_output(llm_response)  # 期望 dict: {"switch":..., "next_rule":...}
                if not isinstance(parsed, dict):
                    # parsed = {"switch": False, "next_rule": 'relpscost'}
                    parsed = {"branching_rule": 'relpscost'}
                # 只保留最新响应
                while True:
                    try:
                        resp_q.get_nowait()
                    except Empty:
                        break
                resp_q.put(parsed)
                # 将查询时间放入时间队列
                while True:
                    try:
                        time_q.get_nowait()
                    except Empty:
                        break
                time_q.put(query_time)
            except Exception as e:
                # 静默失败，返回空 dict
                while True:
                    try:
                        resp_q.get_nowait()
                    except Empty:
                        break
                # resp_q.put({"switch": False, "next_rule": 'relpscost'})
                print('errir and use default')
                resp_q.put({"branching_rule": 'relpscost'})
                # 错误时也记录时间（为0）
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
    args = parser.parse_args()

    result_file = f"{args.problem}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    llm_agent = llmClient(api_key=API_KEY, model_name=MODEL_NAME, api_url=API_URL)
    # if args.asyn: self._ensure_llm_worker()
    instances = []
    results = []
    seeds = [0]
    other_models = ['extratrees_gcnn_agg', 'lambdamart_khalil', 'svmrank_khalil']
    llm_models = ['llm']
    # llm_recomend_branchers = Scheduled_branching_rule_llm(problem_info, depth, llm_agent)
    internal_branchers = ['relpscost']
    time_limit = 3600

    if args.problem == 'setcover':
        # instances += [{'type': 'small', 'path': f"data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp"} for i in range(100)]
        # instances += [{'type': 'medium', 'path': f"data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(100)]
        instances += [{'type': 'big', 'path': f"data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(10)]
    #
    elif args.problem == 'cauctions':
        instances += [{'type': 'small', 'path': f"data/instances/cauctions/transfer_100_500/instance_{i+1}.lp"} for i in range(10)]
        # instances += [{'type': 'medium', 'path': f"data/instances/cauctions/transfer_200_1000/instance_{i+1}.lp"} for i in range(100)]
        # instances += [{'type': 'big', 'path': f"data/instances/cauctions/transfer_300_1500/instance_{i+1}.lp"} for i in range(10)]
    #
    elif args.problem == 'facilities':
        # instances += [{'type': 'small', 'path': f"data/instances/facilities/transfer_100_100_5/instance_{i+1}.lp"} for i in range(100)]
        # instances += [{'type': 'medium', 'path': f"data/instances/facilities/transfer_200_100_5/instance_{i+1}.lp"} for i in range(100)]
        instances += [{'type': 'big', 'path': f"data/instances/facilities/transfer_400_100_5/instance_{i+1}.lp"} for i in range(20)]
        # instances += [{'type': 'medium', 'path': f"data/instances/facilities/transfer_200_100_5/instance_5.lp"}]
    #
    elif args.problem == 'indset':
        # instances += [{'type': 'small', 'path': f"data/instances/indset/transfer_500_4/instance_{i+1}.lp"} for i in range(100)]
        # instances += [{'type': 'medium', 'path': f"data/instances/indset/transfer_1000_4/instance_{i+1}.lp"} for i in range(100)]
        instances += [{'type': 'big', 'path': f"data/instances/indset/transfer_1500_4/instance_{i+1}.lp"} for i in range(10)]

    else:
        raise NotImplementedError

    branching_policies = []

    # SCIP internal brancher baselines
    for brancher in internal_branchers:
        for seed in seeds:
            branching_policies.append({
                    'type': 'internal',
                    'name': brancher,
                    'seed': seed,
             })

    # # ML baselines
    # for model in other_models:
    #     for seed in seeds:
    #         branching_policies.append({
    #             'type': 'ml-competitor',
    #             'name': model,
    #             'seed': seed,
    #             'model': f'trained_models/{args.problem}/{model}/{seed}',
    #         })

    # LLM models
    # for model in llm_models:
    #     for seed in seeds:
    #         branching_policies.append({
    #             'type': 'llm',
    #             'name': model,
    #             'model': llm_agent,
    #             'seed': seed
    #         })

    # LLM-Dynamic scheduler
    for model in llm_models:
        for seed in seeds:
            branching_policies.append({
                'type': 'llm-dynamic',
                'name': model,
                'model': llm_agent,
                'seed': seed
            })

    print(f"problem: {args.problem}")
    print(f"gpu: {args.gpu}")
    print(f"time limit: {time_limit} s")

    # load ml-competitor models
    for policy in branching_policies:
        if policy['type'] == 'ml-competitor':
            try:
                with open(f"{policy['model']}/normalization.pkl", 'rb') as f:
                    policy['feat_shift'], policy['feat_scale'] = pickle.load(f)
            except:
                policy['feat_shift'], policy['feat_scale'] = 0, 1

            with open(f"{policy['model']}/feat_specs.pkl", 'rb') as f:
                policy['feat_specs'] = pickle.load(f)

            if policy['name'].startswith('svmrank'):
                policy['model'] = svmrank.Model().read(f"{policy['model']}/model.txt")
            else:
                with open(f"{policy['model']}/model.pkl", 'rb') as f:
                    policy['model'] = pickle.load(f)


    # load llm
    # loaded_models = {}
    # for policy in branching_policies:
    #     if policy['type'] == 'llm':
    #         if policy['name'] not in loaded_models:
    #             loaded_models[policy['name']] = llm_agent
    #         policy['model'] = llm_agent

    print("running SCIP...")

    fieldnames = [
        'policy',
        'seed',
        'type',
        'instance',
        'nnodes',
        'nlps',
        'stime',
        'gap',
        'status',
        'rpg',
        'ndomchgs',
        'ncutoffs',
        'walltime',
        'proctime',
        'llm_query_time',
    ]
    os.makedirs('results', exist_ok=True)
    with open(f"results/{result_file}", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            for policy in branching_policies:
                m = scip.Model()
                m.setIntParam('display/verblevel', 0)
                print(instance['path'])
                m.readProblem(f"{instance['path']}")
                utilities.init_scip_params(m, seed=policy['seed'])
                # llm_recomend_branchers = select_branching_rule_llm(m, llm_agent, args.problem)
                # policy['name'] = llm_recomend_branchers if policy['type'] in ['llm', 'llm-dynamic'] else policy['name']
                if policy['type'] in ['llm', 'llm-dynamic']:
                    policy['name'] = select_branching_rule_llm(m, llm_agent, args.problem)#, time_tracker=brancher
                    print('policy_name:', policy['name'])
                m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
                m.setRealParam('limits/time', time_limit)
                brancher = PolicyBranching(policy, instance['path'])
                if args.asyn:
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

                stime = m.getSolvingTime()
                nnodes = m.getNNodes()
                nlps = m.getNLPs()
                gap = m.getGap()
                status = m.getStatus()
                ndomchgs = brancher.ndomchgs
                ncutoffs = brancher.ncutoffs

                # === calculate relative primal gap ===
                primal = m.getPrimalbound()
                dual = m.getDualbound()

                if abs(primal) < 1e-9:  # 防止除零
                    rel_primal_gap = float('inf')
                else:
                    rel_primal_gap = abs(primal - dual) / max(1.0, abs(primal))

                # 打印详细指标
                print("=== Solver Result Metrics ===")
                print(f"Policy: {policy['type']}:{policy['name']}")
                print(f"Seed: {policy['seed']}")
                print(f"Instance: {instance['path']}")
                print(f"Number of Nodes: {nnodes}")
                print(f"Number of LP Solves: {nlps}")
                print(f"SCIP Solving Time: {stime:.2f} s")
                print(f"LLM Query Time: {getattr(brancher, 'total_llm_query_time', 0.0):.2f} s")
                print(f"Wall Time: {walltime:.2f} s")
                print(f"Process Time: {proctime:.2f} s")
                print(f"Optimality Gap: {gap:.6f}")
                print(f"Relative Primal Gap: {rel_primal_gap:.6f}")
                print(f"Solving Status: {status}")
                print(f"Number of Domain Changes: {ndomchgs}")
                print(f"Number of Cutoffs: {ncutoffs}")
                print(f"Fair Node Count: {nnodes + 2 * (ndomchgs + ncutoffs)}")
                print("=" * 30)

                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': policy['seed'],
                    'type': instance['type'],
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'stime': stime,
                    'rpg': rel_primal_gap,
                    'gap': gap,
                    'status': status,
                    'ndomchgs': ndomchgs,
                    'ncutoffs': ncutoffs,
                    'walltime': walltime,
                    'proctime': proctime,
                    'llm_query_time': getattr(brancher, 'total_llm_query_time', 0.0),  # 新增字段
                })

                csvfile.flush()

                # 收集结果（用于之后分组求均值）
                results.append({
                    'policy_type': policy['type'],
                    'policy_name': policy['name'],
                    'seed': policy['seed'],
                    'instance_type': instance['type'],
                    'instance_path': instance['path'],
                    'nnodes': float(nnodes),
                    'nlps': float(nlps),
                    'stime': float(stime),
                    'gap': float(gap),
                    'rpg': float(rel_primal_gap),
                    'status': str(status),
                    'ndomchgs': float(ndomchgs),
                    'ncutoffs': float(ncutoffs),
                    'walltime': float(walltime),
                    'proctime': float(proctime),
                    'llm_query_time': float(getattr(brancher, 'total_llm_query_time', 0.0)),  # 新增
                })

                if brancher.record:
                    utilities.record_step_result(args.problem, instance['path'], brancher.step_records)

                m.freeProb()
                time.sleep(1)
        summary = utilities.result_summary(results, csvfile)
        print(summary)
                # print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} ({nnodes+2*(ndomchgs+ncutoffs)}) nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")