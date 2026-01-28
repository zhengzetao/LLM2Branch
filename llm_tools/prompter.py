from nlp_reader import read_lp_file
import numpy as np, json

# def build_initial_branching_prompt(problem_type, problem_info):
#     available_rules = [
#         "pscost", "inference", "mostinf", "relpscost", "leastinf",
#         "distribution", "fullstrong", "cloud", "lookahead", "multaggr",
#         "allfullstrong", "vanillafullstrong", "random", "nodereopt", "multinode"
#     ]

#     rule_descriptions = {
#         "pscost": "Pure pseudo-cost branching (faster but less robust than relpscost).",
#         "relpscost": "Pseudo-cost (reliable after warmup) branching based on observed impact of variables on LP objective.",
#         "mostinf": "Branch on variable with most fractional value (closer to 0.5).",
#         "leastinf": "Branch on variable with least fractional value.",
#         "fullstrong": "Strong branching over all candidates; high accuracy but very expensive.",
#         "allfullstrong": "Even more exhaustive version of fullstrong branching.",
#         "vanillafullstrong": "Basic full strong branching variant.",
#         "hybrid": "Combination of pseudo-cost and strong branching.",
#         "inference": "Branch on variable with most domain propagation (inference gain).",
#         "distribution": "Branching based on domain distribution statistics.",
#         "cloud": "Branch based on structural clustering of problem.",
#         "lookahead": "Use limited lookahead to anticipate impact of branching.",
#         "multaggr": "Branch based on multiple aggregation functions.",
#         "random": "Random branching, typically for baseline or comparison.",
#         "nodereopt": "Use node reoptimization to guide branching.",
#         "multinode": "Use multi-node statistics for decision making.",
#     }

#     rule_doc = "\n".join([f"- {r}: {rule_descriptions[r]}" for r in available_rules if r in rule_descriptions])

#     prompt = (
#     f"""You are an expert in mixed-integer optimization and SCIP solver configuration.
#         Your task is to **choose the best branching rule** from a known list, based on the following:
#         1. Problem type (e.g., setcover, facilities, cauctions, indset)
#         2. Problem structural statistics
#         3. Variable and constraint types

#         Available branching rules:
#         {rule_doc}

#         Problem type: {problem_type}

#         Problem info (JSON):
#         {json.dumps(problem_info, indent=2)}

#         Rules:
#         _ select the best match rule to better solve the specific problem based on its problem type and scale
#         - rule_name must exactly match one from the available rule names.
#         - Do NOT output explanations, alternatives, or additional fields.
#         - Only output the JSON. No markdown, no comments, no reasoning.

#         Think carefully, reason internally, but only return the selected rule in JSON format.
#         """
#             "Please strictly output ONLY the bare JSON object in this format:\n"
#             '{"branching_rule": "rule_name"}\n'
#         )

#     return prompt

# def build_initial_branching_prompt(problem_type, problem_info):
#     available_rules = [
#         "pscost", "inference", "mostinf", "relpscost", "leastinf",
#         "distribution", "fullstrong", "cloud", "lookahead", "multaggr",
#         "allfullstrong", "vanillafullstrong", "random", "nodereopt", "multinode"
#     ]

#     rule_descriptions = {
#         "pscost": "Pure pseudo-cost branching (faster but less robust than relpscost).",
#         "relpscost": "Pseudo-cost (reliable after warmup) branching based on observed impact of variables on LP objective.",
#         "mostinf": "Branch on variable with most fractional value (closer to 0.5).",
#         "leastinf": "Branch on variable with least fractional value.",
#         "fullstrong": "Strong branching over all candidates; high accuracy but very expensive.",
#         "allfullstrong": "Even more exhaustive version of fullstrong branching.",
#         "vanillafullstrong": "Basic full strong branching variant.",
#         "inference": "Branch on variable with most domain propagation (inference gain).",
#         "distribution": "Branching based on domain distribution statistics.",
#         "cloud": "Branch based on structural clustering of problem.",
#         "lookahead": "Use limited lookahead to anticipate impact of branching.",
#         "multaggr": "Branch based on multiple aggregation functions.",
#         "random": "Random branching, typically for baseline or comparison.",
#         "nodereopt": "Use node reoptimization to guide branching.",
#         "multinode": "Use multi-node statistics for decision making.",
#     }

#     # 1. 推荐规则映射表
#     recommended_rules = {
#         "setcover": (
#             "For 'setcover' (Set Covering), problems are highly combinatorial. "
#             "To solve them fastest, prioritizing logical deductions is key. "
#             "**Recommendation: 'pscost'** is often superior as it maximizes constraint propagation, "
#             "quickly reducing the search space."
#         ),
#         "indset": (
#             "For 'indset' (Maximum Independent Set), the graph structure leads to strong logical implications. "
#             "Like set covering, a good branching choice on one node immediately affects its neighbors. "
#             "**Recommendation: 'pscost,inference'** is highly effective for fast convergence."
#         ),
#         "facilities": (
#             "For 'facilities' (Capacitated Facility Location), the binary variables for opening facilities are critical. "
#             "Making the right choice here can prune huge parts of the search tree. "
#             "**Recommendation: 'pscost'**, despite its high per-node cost, is often fastest overall because it makes very high-quality decisions. "
#             "If the number of potential facilities is very large, 'relpscost' is a safer alternative."
#         ),
#         "cauctions": (
#             "For 'cauctions' (Combinatorial Auction), problems often involve selecting a complex combination of bids. "
#             "Identifying the most impactful bids to branch on is crucial for speed. "
#             "**Recommendation: 'pscost'** is very effective for making these critical decisions. 'relppscost' offers a good balance, and 'cloud' can be effective if the item-bid structure has clear clusters."
#         )
#     }
#     rec = recommended_rules.get(problem_type, [])
#     rec_info = ", ".join(rec) if rec else "无特定推荐"

#     # 2. 拼接所有 rule 的简要说明
#     rule_doc = "\n".join(
#         f"- {r}: {rule_descriptions[r]}"
#         for r in available_rules
#         if r in rule_descriptions
#     )

#     prompt = (
#         f"""You are an expert in mixed-integer optimization and SCIP solver configuration.
#         Your task is to **choose the best branching rule** from a known list, based on the following:
#         1. Problem type (e.g., setcover, facilities, auctions, indset)
#         2. Problem structural statistics
#         3. Variable and constraint types

#         Available branching rules:
#         {rule_doc}

#         Problem type: {problem_type}
#         Recommended rules for this type: {rec_info}

#         Problem info (JSON):
#         {json.dumps(problem_info, indent=2)}

#         Rules:
#         - Select the best matching rule to solve the problem fastest, considering type and scale.
#         - `rule_name` must exactly match one from the available list.
#         - Do NOT output explanations, alternatives, or additional fields.
#         - Only output the JSON. No markdown, no comments, no reasoning.
#         """
#         "Please strictly output ONLY the bare JSON object in this format:\n"
#         '{"branching_rule": "rule_name"}\n'
#         )

#     return prompt


def build_initial_branching_prompt(problem_type, problem_info):
    available_rules = [
        "pscost", "inference", "mostinf", "leastinf",
        "distribution", "cloud", "lookahead", "multaggr",
        "allfullstrong", "vanillafullstrong", "random", "nodereopt", "multinode", "relpscost"
    ]
#     rule_descriptions = {
#     "pscost": "Pure pseudo-cost branching (faster but less robust than relpscost).",
#     "relpscost": "Pseudo-cost (reliable after warmup) branching based on observed impact of variables on LP objective.",
#     "mostinf": "Branch on variable with most fractional value (closer to 0.5).",
#     "leastinf": "Branch on variable with least fractional value.",
#     "fullstrong": "Strong branching over all candidates; high accuracy but very expensive.",
#     "allfullstrong": "Even more exhaustive version of fullstrong branching.",
#     "vanillafullstrong": "Basic full strong branching variant.",
#     "inference": "Branch on variable with most domain propagation (inference gain).",
#     "distribution": "Branching based on domain distribution statistics.",
#     "cloud": "Branch based on structural clustering of problem.",
#     "lookahead": "Use limited lookahead to anticipate impact of branching.",
#     "multaggr": "Branch based on multiple aggregation functions.",
#     "random": "Random branching, typically for baseline or comparison.",
#     "nodereopt": "Use node reoptimization to guide branching.",
#     "multinode": "Use multi-node statistics for decision making.",
#     }
    rule_descriptions = {
            "relpscost": "Best when: medium-to-deep trees, many nodes already explored so pseudo-costs are informative; balanced Cut/Dom; need stable scaling. Avoid when: very shallow root where pseudo-costs are untrained and strong probing is more valuable." ,
            "pscost": "Pure pseudo-cost branching (cheaper but less robust); Best when: deep trees with many candidates (large Cands/Entropy), need speed; pseudo-costs already calibrated; Avoid when: early depth or noisy pseudo-costs; when Cut/Dom are very low and robust exploration is needed.",
            "mostinf": "Most-infeasible (most fractional) branching; Best when: moderate depth; many fractional candidates (large Cands/Entropy); need quick low-cost decisions; LP solutions appear ambiguous (fractionality around 0.5). Avoid when: fractionality is polarized (most vars near 0/1) or pseudo-costs are already very informative.",
            "leastinf": "Least-infeasible (least fractional) branching; Best when: LP solutions are nearly integral except few variables; aim to fix stable variables early; Avoid when: many mid-fractional candidates (Entropy high) or early-stage exploration is needed.",
            "fullstrong":  "Full strong branching; most accurate but very expensive; Best when: root or very shallow depth (D≤2~3); poor Cut/Dom; pseudo-costs uncalibrated; Avoid when: deep trees or large candidate sets where per-node budget is tight.",
            "allfullstrong": "Exhaustive full strong branching (very expensive). Best when: tiny instances at root; need maximal reliability for first split. Avoid when: any meaningful depth or large Cands; will explode runtime.",
            "vanillafullstrong": "Vanilla full strong branching. Best when: shallow depth where some but not exhaustive probing is OK; pseudo-costs weak and early correctness matters. Avoid when: depth grows or Cands large.",
            "inference": "Inference-based branching (maximize domain propagation). Best when: Dom ratio high but Cut ratio low; constraints are tight and propagation is powerful; Avoid when: propagation is weak (Dom low) or LP information (pseudo-costs) is already strong.",
            "distribution": "Distribution-based branching (diversification by candidate statistics). Best when: Cands/Entropy large, repeated stagnation (Gap flat), need diversification to escape plateaus. Avoid when: already making steady progress with cost/inference rules.",
            "cloud": "Structure/cluster-aware branching (experimental). Best when: problem shows clear variable clustering/community structure; mid-depth diversification need. Avoid when: structure unclear; rely on proven rules instead.",
            "lookahead": "Limited lookahead branching. Best when: shallow-to-mid depth with stagnating Gap; small-to-moderate Cands; need to anticipate downstream pruning without full strong cost. Avoid when: deep trees or very large Cands (costly).",
            "multaggr": "Multi-aggregation guided branching (blend of signals). Best when: no single signal dominates (Cut/Dom moderate, pseudo-costs partially trained), seek robustness in mid-depth. Avoid when: a clear signal (e.g., strong Dom or early fullstrong need) exists.",
            "random": "Random branching (baseline/exploration). Best when: benchmarking or escaping heavy bias after severe stagnation; very occasional use. Avoid when: performance matters; switch back once signal appears.",
            "nodereopt": "Node-reoptimization guided branching. Best when: repeated substructures; reoptimization feedback strong; mid-depth with recurring patterns. Avoid when: little structure reuse or high per-node cost not justified.",
            "multinode": "Multi-node statistics-based branching. Best when: deep trees with many explored nodes; global statistics stabilize; large instances where local signals are noisy. Avoid when: shallow depth or too few nodes observed.",
        }

    rule_doc = "\n".join([f"- {r}: {rule_descriptions[r]}" for r in available_rules if r in rule_descriptions])

    prompt = (
    f"""You are an expert in mixed-integer optimization and SCIP solver configuration.
        Your task is to **choose the best branching rule** from a known list, based on the following:
        1. Problem type (e.g., set covering(setcover), capacitated facility location(facilities), combinatorial auctions(auctions), maximum independent set(indset)).
        2. Problem structural statistics (e.g., number of variables/constraints, LP density, cut usage).
        3. Variable and constraint types (binary-heavy, general integers, continuous coupling, etc.).

        Available branching rules and their recommended use cases:
        {rule_doc}

        Problem type: {problem_type}

        Problem info (JSON):
        {json.dumps(problem_info, indent=2)}

        Rules:
        - Select the best matching branching rule to solve the problem efficiently, considering type, structure, and scale.
        - rule_name must exactly match one from the available rule names.
        - Do NOT output explanations, alternatives, or additional fields.
        - Only output the JSON. No markdown, no comments, no reasoning.

        Think carefully, reason internally, but only return the selected rule in JSON format.
        """
        "Please strictly output ONLY the bare JSON object in this format:\n"
        '{"branching_rule": "rule_name"}\n'
    )

    return prompt

# def build_initial_branching_prompt(problem_type, problem_info):
#     available_rules = [
#         "pscost", "inference", "mostinf", "relpscost", "leastinf",
#         "distribution", "fullstrong", "cloud", "lookahead", "multaggr",
#         "allfullstrong", "vanillafullstrong", "random", "nodereopt", "multinode"
#     ]

#     # 用“情景 -> 候选规则”而非逐条解释，减少偏置
#     use_case_lines = [
#         # 规模 / 深度 / 候选特征
#         "Small scale + shallow depth + many fractional candidates → fullstrong / vanillafullstrong / lookahead",
#         "Medium scale or unknown scale + need robust cold-start → relpscost (fallback pscost)",
#         "Large scale or tight latency budget → relpscost / pscost / inference (avoid strong variants)",
#         "Deep nodes + few candidates / near-integer → mostinf / inference",
#         # 传播 / 结构
#         "Strong propagation (heavy bound tightenings/cuts) → inference / relpscost",
#         "Clustered or graph-like structure (communities) → cloud / multinode",
#         "Heterogeneous domains / wide ranges → distribution / multaggr",
#         "Stable LP bases across similar nodes → nodereopt",
#         # 其他
#         "Tie-breaking or stress-test only → random / leastinf (avoid unless necessary)"
#     ]
#     use_case_doc = "\n".join(f"- {line}" for line in use_case_lines)

#     prompt = (
#         "You are configuring the INITIAL branching rule for a MILP solver.\n"
#         "Pick exactly ONE rule from the available list to maximize time-to-solution, "
#         "adapting to problem type and structural scale.\n\n"
#         f"Available rules: {', '.join(available_rules)}\n\n"
#         "Recommended use-cases (scenario → candidate rules):\n"
#         f"{use_case_doc}\n\n"
#         f"Problem type: {problem_type}\n"
#         "Problem info (JSON):\n"
#         f"{json.dumps(problem_info, indent=2)}\n\n"
#         "Selection rubric:\n"
#         "- Prefer strong variants ONLY on small & shallow with many fractional candidates.\n"
#         "- Prefer relpscost over pscost for cold-start robustness; use pscost when budget is tight.\n"
#         "- On large instances, avoid fullstrong/allfullstrong/vanillafullstrong.\n"
#         "- If candidates are few or near-integer, favor mostinf/inference.\n"
#         "- If structure suggests clusters, consider cloud/multinode.\n"
#         "- If signals are heterogeneous, consider distribution/multaggr.\n\n"
#         "Output JSON ONLY with exactly one field and an exact rule name from the list:\n"
#         '{"branching_rule": "rule_name"}\n'
#         "No explanations, no markdown, no extra keys."
#     )
#     return prompt



# def build_initial_branching_prompt(problem_type, problem_info):
#     available_rules = [
#         "pscost", "inference", "mostinf", "relpscost", "leastinf",
#         "distribution", "fullstrong", "cloud", "lookahead", "multaggr",
#         "allfullstrong", "vanillafullstrong", "random", "nodereopt", "multinode"
#     ]

#     rule_descriptions = {
#         "pscost": "Pseudo-cost without reliability safeguard: best for very large MILPs where speed dominates.",
#         "relpscost": "Pseudo-cost : balances robustness and speed. Best default for medium-to-large MILPs.",
#         "mostinf": "Most fractional (close to 0.5). Useful for **small-scale combinatorial** problems (below small thresholds).",
#         "leastinf": "Least fractional (close to integer). Rarely optimal except very small problems close to integer feasibility.",
#         "fullstrong": "Full strong branching: expensive but accurate. Best for **very small instances** well below scale thresholds.",
#         "allfullstrong": "Exhaustive strong branching. Benchmark only.",
#         "vanillafullstrong": "Baseline strong branching. Test/research only.",
#         "hybrid": "Mix pseudo-cost and strong branching. Best for large-scale hard MILPs.",
#         "inference": "Branching on inference (domain propagation). Best for scheduling/CSP-like models (auctions, facilities with strong constraints).",
#         "distribution": "Experimental, domain distribution based. Rarely default.",
#         "cloud": "Structural clustering branching. Good for **structured problems like facilities** at large scale.",
#         "lookahead": "Limited lookahead branching. Good for tricky small/medium instances.",
#         "multaggr": "Multiple aggregation branching. Best for medium/large with correlated constraints.",
#         "random": "Baseline random branching. Do not use in production.",
#         "nodereopt": "Node-level reoptimization. Best on medium dense MILPs.",
#         "multinode": "Multi-node statistics. Best on very large structured problems.",
#     }

#     rule_doc = "\n".join([f"- {r}: {rule_descriptions[r]}" for r in available_rules if r in rule_descriptions])

#     prompt = (
#     f"""You are an expert in mixed-integer optimization and SCIP branching selection.

#         Your task: choose the most appropriate branching rule based on:
#         1. Problem type (setcover, facilities, cauction, indset)
#         2. Problem statistics (n_vars, n_constraints, structure, binary ratio)
#         3. Instance scale category (small/medium/large)

#         Available branching rules and suitability:
#         {rule_doc}

#         Guidelines by instance scale:
#         - Small: prefer strong branching (fullstrong/vanillafullstrong) or mostinf/leastinf for quick combinatorial fractionality cuts.
#         - Medium: prefer relpscost, inference, or multaggr depending on structure.
#         - Large: prefer pscost, hybrid, multinode, or cloud (for structured problems).

#         Now apply these rules.

#         Problem type: {problem_type}
#         Problem info (JSON):
#         {json.dumps(problem_info, indent=2)}

#         Rules:
#         - Select ONLY one branching rule from the list.
#         - No explanations, no markdown, no reasoning.
#         """
#         "Please strictly output ONLY the bare JSON object in this format:\n"
#         '{"branching_rule": "rule_name"}\n' 
#     )

#     return prompt


# def build_dynamic_branching_prompt(llm_state_history):             {{"branching_rule": "<RULE_NAME>"}}

#     # 构造 prompt
#     prompt = "Search history:\n"
#     for s in llm_state_history[-10:]:
#         prompt += f"{s['step']:>2} | Rule: {s['rule']:<12} | Depth: {s['depth']:<2} | Cutoff: {s['cutoff_ratio']:<5} | Entropy: {s['entropy']:<5} | Gap: {s['gap']:<6}\n"
#     prompt += (
#         "\nBased on the above state transitions, should we switch to a different branching rule?\n"
#         "Respond strictly in the following JSON format:\n"
#         '{"switch": true/false, "next_rule": "rule_name", "correction": true/false}\n'
#     )

#     return prompt

# def build_dynamic_branching_prompt(llm_state_history):
#     available_rules = [
#         "gomory", "pscost", "inference", "mostinf", "relpscost", "leastinf",
#         "distribution", "fullstrong", "cloud", "lookahead", "multaggr",
#         "allfullstrong", "vanillafullstrong", "random", "nodereopt", "multinode"
#     ]

#     rule_descriptions = {
#         "relpscost": "Reliable pseudo-cost branching based on observed impact of variables on LP objective.",
#         "pscost": "Pure pseudo-cost branching (faster but less robust than relpscost).",
#         "mostinf": "Branch on variable with most fractional value (closer to 0.5).",
#         "leastinf": "Branch on variable with least fractional value.",
#         "fullstrong": "Strong branching over all candidates; high accuracy but very expensive.",
#         "allfullstrong": "Even more exhaustive version of fullstrong branching.",
#         "vanillafullstrong": "Basic full strong branching variant.",
#         "hybrid": "Combination of pseudo-cost and strong branching.",
#         "inference": "Branch on variable with most domain propagation (inference gain).",
#         "distribution": "Branching based on domain distribution statistics.",
#         "gomory": "Use Gomory cut information for branching decisions.",
#         "cloud": "Branch based on structural clustering of problem.",
#         "lookahead": "Use limited lookahead to anticipate impact of branching.",
#         "multaggr": "Branch based on multiple aggregation functions.",
#         "random": "Random branching, typically for baseline or comparison.",
#         "nodereopt": "Use node reoptimization to guide branching.",
#         "multinode": "Use multi-node statistics for decision making.",
#     }

#     rule_doc = "\n".join([f"- {r}: {rule_descriptions[r]}" for r in available_rules if r in rule_descriptions])

#     prompt = "Search history (last 10 steps):\n"
#     for s in llm_state_history:
#         prompt += f"{s['step']:>2} | Rule: {s['rule']:<12} | Depth: {s['depth']:<2} | Cutoff: {s['cutoff_ratio']:<5} | Entropy: {s['entropy']:<5} | Gap: {s['gap']:<6}\n"

#     prompt += (
#         "\nYou are an expert in branch-and-bound optimization with SCIP.\n"
#         "Based on the above search history, determine whether to switch to a different branching rule.\n\n"
#         "Available branching rules and descriptions:\n"
#         f"{rule_doc}\n\n"
#         "Respond strictly in the following JSON format (only these fields):\n"
#         '{"switch": true/false, "next_rule": "rule_name", "correction": true/false}\n\n'
#         "Rules:\n"
#         "- If 'switch' is true, 'next_rule' must be selected from the above list.\n"
#         "- If 'correction' is true, it means the last rule was a mistake and should be corrected.\n"
#         "- Do NOT include explanations or additional fields.\n"
#         "- Only output valid JSON, no comments or markdown.\n"
#     )

#     return prompt

def build_dynamic_branching_prompt(llm_state_history, max_len=20):
    """
    仅基于 llm_state_history 构造用于“是否切换 branching rule”的 LLM 提示。
    llm_state_history: List[dict]，每个元素是一次 tree_state 快照，包含：
      - "depth", "nodes_explored", "primal_bound", "dual_bound", "gap",
        "cutoff_ratio", "domchg_ratio", "num_branch_candidates", "entropy"
    max_len: 传入给 LLM 的最近历史条数上限
    """
    # 1) available rules
    available_rules = [
        "pscost", "inference", "mostinf", "leastinf",
        "distribution", "fullstrong", "cloud", "lookahead", "multaggr",
        "allfullstrong", "vanillafullstrong", "random", "nodereopt", "multinode", "relpscost"
    ]
    # rule_descriptions = {
    #     "relpscost": "Reliable pseudo-cost branching; robust general-purpose choice.",
    #     "pscost": "Pure pseudo-cost branching; cheaper but less robust than relpscost.",
    #     "mostinf": "Branch on the most fractional variable (closest to 0.5).",
    #     "leastinf": "Branch on the least fractional variable.",
    #     "fullstrong": "Full strong branching; accurate but very expensive.",
    #     "allfullstrong": "Even more exhaustive full strong branching variant (very expensive).",
    #     "vanillafullstrong": "Basic full strong branching variant.",
    #     "inference": "Choose variable that maximizes domain propagation (inference gain).",
    #     "distribution": "Use domain/domain-distribution statistics to diversify branching.",
    #     "gomory": "Exploit Gomory cut information for branching.",
    #     "cloud": "Structure/cluster-aware branching (experimental).",
    #     "lookahead": "Limited lookahead to anticipate downstream impact.",
    #     "multaggr": "Branching guided by multiple aggregations.",
    #     "random": "Random branching (baseline/ablation).",
    #     "nodereopt": "Use node reoptimization feedback.",
    #     "multinode": "Leverage multi-node statistics across the tree."
    # }
    rule_descriptions = {
        "relpscost": "Best when: medium-to-deep trees, many nodes already explored so pseudo-costs are informative; balanced Cut/Dom; need stable scaling. Avoid when: very shallow root where pseudo-costs are untrained and strong probing is more valuable." ,
        "pscost": "Pure pseudo-cost branching (cheaper but less robust); Best when: deep trees with many candidates (large Cands/Entropy), need speed; pseudo-costs already calibrated; Avoid when: early depth or noisy pseudo-costs; when Cut/Dom are very low and robust exploration is needed.",
        "mostinf": "Most-infeasible (most fractional) branching; Best when: moderate depth; many fractional candidates (large Cands/Entropy); need quick low-cost decisions; LP solutions appear ambiguous (fractionality around 0.5). Avoid when: fractionality is polarized (most vars near 0/1) or pseudo-costs are already very informative.",
        "leastinf": "Least-infeasible (least fractional) branching; Best when: LP solutions are nearly integral except few variables; aim to fix stable variables early; Avoid when: many mid-fractional candidates (Entropy high) or early-stage exploration is needed.",
        "fullstrong":  "Full strong branching; most accurate but very expensive; Best when: root or very shallow depth (D≤2~3); poor Cut/Dom; pseudo-costs uncalibrated; Avoid when: deep trees or large candidate sets where per-node budget is tight.",
        "allfullstrong": "Exhaustive full strong branching (very expensive). Best when: tiny instances at root; need maximal reliability for first split. Avoid when: any meaningful depth or large Cands; will explode runtime.",
        "vanillafullstrong": "Vanilla full strong branching. Best when: shallow depth where some but not exhaustive probing is OK; pseudo-costs weak and early correctness matters. Avoid when: depth grows or Cands large.",
        "inference": "Inference-based branching (maximize domain propagation). Best when: Dom ratio high but Cut ratio low; constraints are tight and propagation is powerful; Avoid when: propagation is weak (Dom low) or LP information (pseudo-costs) is already strong.",
        "distribution": "Distribution-based branching (diversification by candidate statistics). Best when: Cands/Entropy large, repeated stagnation (Gap flat), need diversification to escape plateaus. Avoid when: already making steady progress with cost/inference rules.",
        "cloud": "Structure/cluster-aware branching (experimental). Best when: problem shows clear variable clustering/community structure; mid-depth diversification need. Avoid when: structure unclear; rely on proven rules instead.",
        "lookahead": "Limited lookahead branching. Best when: shallow-to-mid depth with stagnating Gap; small-to-moderate Cands; need to anticipate downstream pruning without full strong cost. Avoid when: deep trees or very large Cands (costly).",
        "multaggr": "Multi-aggregation guided branching (blend of signals). Best when: no single signal dominates (Cut/Dom moderate, pseudo-costs partially trained), seek robustness in mid-depth. Avoid when: a clear signal (e.g., strong Dom or early fullstrong need) exists.",
        "random": "Random branching (baseline/exploration). Best when: benchmarking or escaping heavy bias after severe stagnation; very occasional use. Avoid when: performance matters; switch back once signal appears.",
        "nodereopt": "Node-reoptimization guided branching. Best when: repeated substructures; reoptimization feedback strong; mid-depth with recurring patterns. Avoid when: little structure reuse or high per-node cost not justified.",
        "multinode": "Multi-node statistics-based branching. Best when: deep trees with many explored nodes; global statistics stabilize; large instances where local signals are noisy. Avoid when: shallow depth or too few nodes observed.",
    }

    rule_doc = "\n".join(f"- {r}: {rule_descriptions[r]}" for r in available_rules)

    # 2) 组织最近 N 条历史为纯文本表
    history = llm_state_history[-max_len:] if llm_state_history else []
    lines = ["Search history snapshots (last up to %d):" % max_len]
    if not history:
        lines.append("(no history yet)")
    else:
        # 给每条加上一个递增的 step 索引（仅用于展示）
        start_idx = max(0, len(llm_state_history) - len(history))
        for i, s in enumerate(history, start=start_idx):
            lines.append(
                f"{i:>3} | D:{s.get('depth','-'):>3} | Nodes:{s.get('nodes_explored','-'):>6} | "
                f"Gap:{s.get('gap','-'):>10} | Cut:{s.get('cutoff_ratio','-'):>7} | "
                f"Dom:{s.get('domchg_ratio','-'):>7} | Cands:{s.get('num_branch_candidates','-'):>4} | "
                f"H:{s.get('entropy','-'):>7}"
            )
    history_block = "\n".join(lines)

    # 3) 当前状态 = 历史最后一条（若有）
    current_state = history[-1] if history else {}

    # 4) 构造 prompt（仅 JSON 输出；从白名单中选 next_rule）
    # prompt = (
    #     f"{history_block}\n\n"
    #     "Current tree state (JSON):\n"
    #     f"{json.dumps(current_state, indent=2) if current_state else '(no current state)'}\n\n"
    #     "You are an expert in branch-and-bound with SCIP.\n"
    #     "Decide whether to SWITCH the branching rule NOW based on BOTH the historical effects above and the current tree state.\n\n"
    #     "Available branching rules (choose ONLY from this list):\n"
    #     f"{rule_doc}\n\n"
    #     "STRICT OUTPUT REQUIREMENTS (JSON ONLY):\n"
    #     '{"switch": true/false, "next_rule": "rule_name", "correction": true/false}\n'
    #     "Constraints:\n"
    #     "- If \"switch\" is true, \"next_rule\" MUST be one of the available rule names listed above.\n"
    #     "- If \"switch\" is false, set \"next_rule\" to the rule you recommend to continue with (still MUST be from the list).\n"
    #     "- \"correction\" = true means the last rule choice was a mistake and should be corrected now.\n"
    #     "- IMPORTANT: Switching back and forth repeatedly (A→B→A→B) is not recommended. "
    #     "- Do NOT recommend switching back to the immediately previous rule unless there has been prolonged stagnation "
    #     "- AND you are confident it will bring significant improvement.\n"
    #     # "Switching has a cost; avoid switching unless you are confident the new rule will outperform for the next N levels.\n"
    #     "- Do NOT include explanations or any extra fields. Return ONLY the JSON object.\n"
    # )

     # 4) Prompt（加入清晰的“如何用这几列做判断”的规程）
    prompt = (
        f"{history_block}\n\n"
        "Current tree state (JSON):\n"
        f"{json.dumps(current_state, indent=2) if current_state else '(no current state)'}\n\n"
        "You are an expert in branch-and-bound with SCIP.\n"
        "Decide whether to SWITCH the branching rule NOW based on BOTH the historical effects above and the current tree state.\n\n"
        "Available branching rules (choose ONLY from this list):\n"
        f"{rule_doc}\n\n"
        "Decision rubric (use the numeric columns in the history table above):\n"
        "1) Build two windows from the last rows: the most recent W_r (e.g., last 5 rows) and the previous W_p (the 5 rows before W_r, if available).\n"
        "   From each window compute:\n"
        "   - Gap = Gap[first] - Gap[last]; Nodes = Nodes[last] - Nodes[first];\n"
        "   - Per-node improvement r = Gap / max(1, Nodes).\n"
        "   Recent stagnation if (Gap < 1e-3) OR (r < 1e-5).\n"
        "   Also compute mean(Cut), mean(Dom), mean(Cands), and use current Depth.\n"
        "2) KEEP current rule if the recent window shows decent progress:\n"
        "   - Gap >= 1e-3 OR r >= 1e-5; OR mean(Cut) increasing vs previous window.\n"
        "3) CONSIDER SWITCH only if stagnation holds AND progress is worse than the previous window.\n"
        # "4) Choosing the next rule (heuristics by pattern):\n"
        # "   - Shallow depth (Depth <= 3) AND large Cands (>= 50) AND stagnation → prefer strong evaluation: 'fullstrong' (or 'lookahead' if very large Cands >= 200).\n"
        # "   - High mean(Dom) but low mean(Cut) with stagnation -> propagation not converting to pruning -> choose cost-based 'relpscost' (or 'pscost' if need cheaper).\n"
        # "   - Low mean(Dom) AND low mean(Cut) with stagnation -> diversify branching: 'mostinf' or 'distribution'.\n"
        # "   - Deep tree (Depth >= 15) with stagnation -> avoid expensive strong branching; choose 'relpscost' (fallback) or 'pscost'.\n"
        # "   - Very high entropy (large candidate set) with stagnation -> 'distribution' or 'multinode'; if still shallow, 'lookahead'.\n"
        "5) Tie-breaks: prefer 'relpscost' as a safe fallback; avoid 'allfullstrong' unless extremely shallow and tiny tree.\n"
        "6) Anti-oscillation: Do NOT recommend immediate backtracking (A->B->A). Only consider switching back to the immediately previous rule if:\n"
        "   - There has been prolonged stagnation in the recent window AND you expect significant improvement with the backtrack.\n\n"
        "STRICT OUTPUT REQUIREMENTS (JSON ONLY):\n"
        '{"switch": true/false, "next_rule": "rule_name"}\n'
        "Constraints:\n"
        "- If \"switch\" is true, \"next_rule\" MUST be one of the available rule names listed above.\n"
        "- If \"switch\" is false, still set \"next_rule\" to the rule you recommend to continue with (must be from the list).\n"
        "- Do NOT include explanations or any extra fields. Return ONLY the JSON object.\n"
    )
    return prompt

