import json

from llm_tools.rules import (
    DYNAMIC_BRANCHING_RULES,
    INITIAL_BRANCHING_RULES,
    RULE_DESCRIPTIONS,
)


def _rule_doc(rules):
    return "\n".join(f"- {rule}: {RULE_DESCRIPTIONS[rule]}" for rule in rules)


def _fmt_state_value(value, width):
    """Format mixed numeric/string/None state values for prompt tables."""
    if value is None:
        value = "-"
    return f"{str(value):>{width}}"


def build_initial_branching_prompt(problem_type, problem_info):
    prompt = (
        f"""You are an expert in mixed-integer optimization and SCIP solver configuration.
        Your task is to **choose the best branching rule** from a known list, based on the following:
        1. Problem type (e.g., set covering(setcover), capacitated facility location(facilities), combinatorial auctions(auctions), maximum independent set(indset)).
        2. Problem structural statistics (e.g., number of variables/constraints, LP density, cut usage).
        3. Variable and constraint types (binary-heavy, general integers, continuous coupling, etc.).

        Available branching rules and their recommended use cases:
        {_rule_doc(INITIAL_BRANCHING_RULES)}

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


def build_dynamic_branching_prompt(llm_state_history, max_len=20):
    """
    Build the LLM prompt for deciding whether to switch branching rules.

    llm_state_history is a list of tree-state snapshots collected during SCIP
    search. Only the latest max_len snapshots are included in the prompt.
    """
    history = llm_state_history[-max_len:] if llm_state_history else []
    lines = ["Search history snapshots (last up to %d):" % max_len]
    if not history:
        lines.append("(no history yet)")
    else:
        start_idx = max(0, len(llm_state_history) - len(history))
        for i, state in enumerate(history, start=start_idx):
            lines.append(
                f"{i:>3} | D:{_fmt_state_value(state.get('depth', '-'), 3)} | "
                f"Nodes:{_fmt_state_value(state.get('nodes_explored', '-'), 6)} | "
                f"Gap:{_fmt_state_value(state.get('gap', '-'), 10)} | "
                f"Cut:{_fmt_state_value(state.get('cutoff_ratio', '-'), 7)} | "
                f"Dom:{_fmt_state_value(state.get('domchg_ratio', '-'), 7)} | "
                f"Cands:{_fmt_state_value(state.get('num_branch_candidates', '-'), 4)} | "
                f"H:{_fmt_state_value(state.get('entropy', '-'), 7)}"
            )

    history_block = "\n".join(lines)
    current_state = history[-1] if history else {}

    prompt = (
        f"{history_block}\n\n"
        "Current tree state (JSON):\n"
        f"{json.dumps(current_state, indent=2) if current_state else '(no current state)'}\n\n"
        "You are an expert in branch-and-bound with SCIP.\n"
        "Decide whether to SWITCH the branching rule NOW based on BOTH the historical effects above and the current tree state.\n\n"
        "Available branching rules (choose ONLY from this list):\n"
        f"{_rule_doc(DYNAMIC_BRANCHING_RULES)}\n\n"
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
