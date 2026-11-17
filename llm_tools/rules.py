FALLBACK_RULE = "relpscost"
CLIENT_FALLBACK_RULE = "pscost"

INITIAL_BRANCHING_RULES = [
    "pscost", "inference", "mostinf", "leastinf",
    "distribution", "cloud", "lookahead", "multaggr",
    "allfullstrong", "vanillafullstrong", "random", "nodereopt",
    "multinode", "relpscost",
]

DYNAMIC_BRANCHING_RULES = [
    "pscost", "inference", "mostinf", "leastinf",
    "distribution", "fullstrong", "cloud", "lookahead", "multaggr",
    "allfullstrong", "vanillafullstrong", "random", "nodereopt",
    "multinode", "relpscost",
]

PARSER_BRANCHING_RULES = ["gomory"] + DYNAMIC_BRANCHING_RULES

# Rules that are known to be executable through SCIP executeBranchRule() in the
# target environment. Keep this list aligned with the rules exposed to the LLM
# so recommendations can be adopted instead of silently falling back.
SAFE_EXECUTABLE_RULES = list(DYNAMIC_BRANCHING_RULES)

RULE_DESCRIPTIONS = {
    "relpscost": "Best when: medium-to-deep trees, many nodes already explored so pseudo-costs are informative; balanced Cut/Dom; need stable scaling. Avoid when: very shallow root where pseudo-costs are untrained and strong probing is more valuable.",
    "pscost": "Pure pseudo-cost branching (cheaper but less robust); Best when: deep trees with many candidates (large Cands/Entropy), need speed; pseudo-costs already calibrated; Avoid when: early depth or noisy pseudo-costs; when Cut/Dom are very low and robust exploration is needed.",
    "mostinf": "Most-infeasible (most fractional) branching; Best when: moderate depth; many fractional candidates (large Cands/Entropy); need quick low-cost decisions; LP solutions appear ambiguous (fractionality around 0.5). Avoid when: fractionality is polarized (most vars near 0/1) or pseudo-costs are already very informative.",
    "leastinf": "Least-infeasible (least fractional) branching; Best when: LP solutions are nearly integral except few variables; aim to fix stable variables early; Avoid when: many mid-fractional candidates (Entropy high) or early-stage exploration is needed.",
    "fullstrong": "Full strong branching; most accurate but very expensive; Best when: root or very shallow depth (D<=2~3); poor Cut/Dom; pseudo-costs uncalibrated; Avoid when: deep trees or large candidate sets where per-node budget is tight.",
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


def normalize_rule(name, available_rules=None, fallback=FALLBACK_RULE):
    if not isinstance(name, str):
        return fallback
    rules = set(available_rules or PARSER_BRANCHING_RULES)
    rule = name.strip().lower()
    return rule if rule in rules else fallback


def normalize_executable_rule(name, fallback=FALLBACK_RULE):
    return normalize_rule(name, available_rules=SAFE_EXECUTABLE_RULES, fallback=fallback)


def split_models(model_name):
    if isinstance(model_name, str):
        models = [m.strip() for m in model_name.split(",") if m.strip()]
        return models if models else [model_name.strip()]
    if isinstance(model_name, (list, tuple)):
        return [str(x).strip() for x in model_name if str(x).strip()]
    return [str(model_name).strip()]
