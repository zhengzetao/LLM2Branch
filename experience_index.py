import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from llm_tools.rules import FALLBACK_RULE, normalize_rule


PROBLEM_TYPE_TO_FILE_STEM = {
    "setcover": "set_covering",
    "set_covering": "set_covering",
    "facilities": "facility_location",
    "facility_location": "facility_location",
    "cauctions": "combinatorial_auction",
    "combinatorial_auction": "combinatorial_auction",
    "indset": "mis",
    "mis": "mis",
}


LEAF_LEVELS = (
    "scale_bucket",
    "structure_bucket",
    "search_phase",
    "gap_level",
    "tree_shape",
)


@dataclass
class ExperienceConfig:
    """Configuration for the scheduling memory.

    The defaults intentionally favor correctness and interpretability over
    aggressive switching. All values are plain Python primitives so they can be
    serialized alongside the JSON index if needed later.
    """

    small_threshold: float = 1000.0
    medium_threshold: float = 5000.0
    large_threshold: float = 20000.0
    checkpoint_interval: int = 10
    episode_window: int = 10
    llm_min_interval: int = 40
    save_every_episodes: int = 5
    min_leaf_visits: int = 5
    min_rule_effective_samples: int = 2
    min_ucb_margin: float = 0.01
    ucb_c: float = 1.0
    switching_penalty: float = 0.01
    truncated_weight: float = 0.5
    gap_weight: float = 1.0
    bound_weight: float = 0.2
    prune_weight: float = 0.1
    node_penalty_weight: float = 1e-4
    time_penalty_weight: float = 1e-3
    lp_penalty_weight: float = 1e-5
    enable_callback_gap_metrics: bool = False
    enable_callback_bound_metrics: bool = False
    enable_callback_depth_metrics: bool = False
    enable_callback_candidate_metrics: bool = False


@dataclass
class UCBDecision:
    """Result of asking a leaf UCB table for a rule recommendation."""

    confident: bool
    rule: str = None
    reason: str = ""
    scores: dict = field(default_factory=dict)


def canonical_problem_type(problem_type):
    return PROBLEM_TYPE_TO_FILE_STEM.get(str(problem_type or "").strip().lower(), "unknown")


def experience_path_for_problem(problem_type, base_dir="experience_indexes"):
    return Path(base_dir) / f"{canonical_problem_type(problem_type)}.json"


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _nonnegative_delta(before, after):
    before = _safe_float(before, None)
    after = _safe_float(after, None)
    if before is None or after is None:
        return 0.0
    return max(0.0, after - before)


def compute_reward_components(pre_metrics, post_metrics, config=None):
    """Compute reward terms from two B&B metric snapshots.

    Missing metrics contribute zero instead of failing the solver callback. This
    keeps the first implementation usable even when a PySCIPOpt build does not
    expose optional counters such as open nodes or LP iterations.
    """

    config = config or ExperienceConfig()
    pre_metrics = pre_metrics or {}
    post_metrics = post_metrics or {}

    start_gap = _safe_float(pre_metrics.get("gap"), None)
    end_gap = _safe_float(post_metrics.get("gap"), None)
    gap_reward = 0.0
    if start_gap is not None and end_gap is not None:
        gap_reward = max(0.0, start_gap - end_gap)

    start_dual = _safe_float(pre_metrics.get("dual_bound"), None)
    end_dual = _safe_float(post_metrics.get("dual_bound"), None)
    bound_reward = 0.0
    if start_dual is not None and end_dual is not None:
        denom = max(1.0, abs(start_dual))
        bound_reward = max(0.0, (end_dual - start_dual) / denom)

    delta_nodes = _nonnegative_delta(pre_metrics.get("nnodes"), post_metrics.get("nnodes"))
    delta_cutoffs = _nonnegative_delta(pre_metrics.get("ncutoffs"), post_metrics.get("ncutoffs"))
    prune_reward = delta_cutoffs / max(delta_nodes, 1.0)

    delta_time = _nonnegative_delta(pre_metrics.get("stime"), post_metrics.get("stime"))
    delta_lps = _nonnegative_delta(pre_metrics.get("nlps"), post_metrics.get("nlps"))

    node_penalty = config.node_penalty_weight * delta_nodes
    time_penalty = config.time_penalty_weight * delta_time
    lp_iteration_penalty = config.lp_penalty_weight * delta_lps

    return {
        "gap_reward": config.gap_weight * gap_reward,
        "bound_reward": config.bound_weight * bound_reward,
        "prune_reward": config.prune_weight * prune_reward,
        "node_penalty": node_penalty,
        "time_penalty": time_penalty,
        "lp_iteration_penalty": lp_iteration_penalty,
    }


def compute_reward(pre_metrics, post_metrics, config=None):
    components = compute_reward_components(pre_metrics, post_metrics, config=config)
    return (
        components["gap_reward"]
        + components["bound_reward"]
        + components["prune_reward"]
        - components["node_penalty"]
        - components["time_penalty"]
        - components["lp_iteration_penalty"]
    )


class HierarchicalExperienceIndex:
    """Problem-type-local hierarchy whose leaves hold UCB rule tables."""

    def __init__(self, problem_type="unknown", config=None, tree=None):
        self.problem_type = canonical_problem_type(problem_type)
        self.config = config or ExperienceConfig()
        self.tree = tree if isinstance(tree, dict) else {}

    @classmethod
    def load(cls, path, problem_type="unknown", config=None):
        path = Path(path)
        if not path.exists():
            return cls(problem_type=problem_type, config=config)

        with path.open("r") as file:
            payload = json.load(file)

        return cls(
            problem_type=payload.get("problem_type", problem_type),
            config=config,
            tree=payload.get("tree", {}),
        )

    def save(self, path):
        """Persist the index with deterministic indentation for inspection."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "problem_type": self.problem_type,
            "levels": list(LEAF_LEVELS),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tree": self.tree,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(tmp_path, path)

    def build_leaf_key(self, state_summary):
        state_summary = state_summary or {}
        return (
            self._scale_bucket(state_summary),
            self._structure_bucket(state_summary),
            self._search_phase(state_summary),
            self._gap_level(state_summary),
            self._tree_shape(state_summary),
        )

    def get_leaf(self, leaf_key, create=True):
        node = self.tree
        for part in tuple(leaf_key):
            if create:
                node = node.setdefault(part, {})
            else:
                node = node.get(part)
                if node is None:
                    return None

        if create:
            self._ensure_leaf_shape(node)
        return node

    def record_recommendation(self, leaf_key, rule, source, adopted):
        leaf = self.get_leaf(leaf_key, create=True)
        entry = self._ensure_rule_entry(leaf, rule)
        if source == "llm":
            entry["llm_recommend_count"] += 1
        if adopted:
            entry["adopt_count"] += 1
        entry["last_updated"] = self._now()
        return entry

    def record_adoption(self, leaf_key, rule):
        """Record that the scheduler actually ran ``rule`` in this leaf.

        This is separate from ``record_recommendation`` because a single LLM
        decision has two facts: the LLM recommended a target rule, and the
        scheduler may or may not adopt it after anti-oscillation checks. Keeping
        them separate avoids double-counting LLM recommendations.
        """
        leaf = self.get_leaf(leaf_key, create=True)
        entry = self._ensure_rule_entry(leaf, rule)
        entry["adopt_count"] += 1
        entry["last_updated"] = self._now()
        return entry

    def record_passive_exposure(self, leaf_key):
        leaf = self.get_leaf(leaf_key, create=True)
        leaf["passive_exposure_count"] = leaf.get("passive_exposure_count", 0) + 1
        leaf["last_updated"] = self._now()

    def select_rule(self, leaf_key, current_active_rule=None):
        leaf = self.get_leaf(leaf_key, create=False)
        if not leaf or not leaf.get("rules"):
            return UCBDecision(False, reason="empty_leaf")

        total_visits = float(leaf.get("total_leaf_visits", 0.0))
        if total_visits < self.config.min_leaf_visits:
            return UCBDecision(False, reason="insufficient_leaf_visits")

        scores = {}
        for rule, entry in leaf.get("rules", {}).items():
            if entry.get("n_effective", 0.0) < self.config.min_rule_effective_samples:
                continue
            scores[rule] = self.ucb_score(rule, entry, total_visits, current_active_rule)

        if not scores:
            return UCBDecision(False, reason="insufficient_rule_samples")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_rule, best_score = ranked[0]
        if len(ranked) < 2:
            return UCBDecision(False, rule=best_rule, reason="single_rule_only", scores=scores)
        if len(ranked) > 1:
            margin = best_score - ranked[1][1]
            if margin < self.config.min_ucb_margin:
                return UCBDecision(False, rule=best_rule, reason="small_ucb_margin", scores=scores)

        return UCBDecision(True, rule=best_rule, reason="confident", scores=scores)

    def ucb_score(self, rule, entry, total_leaf_visits, current_active_rule=None):
        n_effective = float(entry.get("n_effective", 0.0))
        mean_reward = float(entry.get("mean_reward", 0.0))
        exploration = self.config.ucb_c * math.sqrt(
            math.log(float(total_leaf_visits) + 1.0) / (n_effective + 1.0)
        )
        penalty = 0.0
        if current_active_rule and normalize_rule(rule, fallback=None) != normalize_rule(current_active_rule, fallback=None):
            penalty = self.config.switching_penalty
        return mean_reward + exploration - penalty

    def update_rule_reward(self, leaf_key, rule, reward, weight=1.0, components=None):
        """Update weighted running reward statistics for a leaf/rule pair."""

        leaf = self.get_leaf(leaf_key, create=True)
        entry = self._ensure_rule_entry(leaf, rule)
        weight = max(0.0, float(weight))
        reward = float(reward)
        old_n = float(entry.get("n_effective", 0.0))
        new_n = old_n + weight
        if new_n <= 0:
            return entry

        old_mean = float(entry.get("mean_reward", 0.0))
        new_mean = old_mean + (weight / new_n) * (reward - old_mean)

        # Store a population-style variance estimate. It is enough for
        # inspection and future confidence heuristics; UCB currently uses mean.
        old_var = float(entry.get("reward_variance", 0.0))
        if old_n <= 0:
            new_var = 0.0
        else:
            old_m2 = old_var * old_n
            new_m2 = old_m2 + weight * (reward - old_mean) * (reward - new_mean)
            new_var = max(0.0, new_m2 / new_n)

        entry["n_effective"] = new_n
        entry["mean_reward"] = new_mean
        entry["reward_variance"] = new_var
        entry["last_reward"] = reward
        entry["success_count"] += 1 if reward > 0 else 0
        entry["exposure_count"] += 1
        entry["last_updated"] = self._now()

        for component_name in (
            "gap_reward",
            "bound_reward",
            "prune_reward",
            "node_penalty",
            "time_penalty",
        ):
            if components and component_name in components:
                mean_key = "mean_" + component_name
                old_component = float(entry.get(mean_key, 0.0))
                entry[mean_key] = old_component + (weight / new_n) * (
                    float(components[component_name]) - old_component
                )

        leaf["total_leaf_visits"] = float(leaf.get("total_leaf_visits", 0.0)) + weight
        leaf["last_updated"] = self._now()
        return entry

    def _scale_bucket(self, state):
        num_vars = _safe_float(state.get("num_variables"), 0.0) or 0.0
        num_constraints = _safe_float(state.get("num_constraints"), 0.0) or 0.0
        num_nonzeros = _safe_float(state.get("num_nonzeros"), None)
        size_terms = [num_vars, num_constraints]
        if num_nonzeros is not None:
            size_terms.append(num_nonzeros / 10.0)
        size_score = max(size_terms)

        if size_score < self.config.small_threshold:
            return "small"
        if size_score < self.config.medium_threshold:
            return "medium"
        if size_score < self.config.large_threshold:
            return "large"
        return "huge"

    def _structure_bucket(self, state):
        binary_ratio = _safe_float(state.get("binary_variable_ratio"), None)
        density = _safe_float(state.get("matrix_density"), None)
        if binary_ratio is None or density is None:
            return "unknown"
        if binary_ratio >= 0.9 and density < 0.01:
            return "sparse_binary"
        if binary_ratio >= 0.9 and density >= 0.01:
            return "dense_binary"
        if binary_ratio < 0.9 and density < 0.01:
            return "mixed_sparse"
        return "mixed_dense"

    def _search_phase(self, state):
        explored_nodes = _safe_int(state.get("explored_nodes"), 0)
        if explored_nodes < 50:
            return "initial"

        elapsed_time_ratio = _safe_float(state.get("elapsed_time_ratio"), None)
        if elapsed_time_ratio is not None:
            if elapsed_time_ratio < 0.25:
                return "early"
            if elapsed_time_ratio < 0.75:
                return "middle"
            return "late"

        if explored_nodes < 500:
            return "early"
        if explored_nodes < 5000:
            return "middle"
        return "late"

    def _gap_level(self, state):
        gap = _safe_float(state.get("current_gap", state.get("gap")), None)
        if gap is None:
            return "unknown"
        if gap > 0.20:
            return "high"
        if gap > 0.05:
            return "medium"
        if gap > 0.005:
            return "low"
        return "very_low"

    def _tree_shape(self, state):
        explored_nodes = _safe_int(state.get("explored_nodes"), 0)
        if explored_nodes < 50:
            return "initial"

        open_nodes = _safe_float(state.get("open_nodes"), None)
        if open_nodes is not None:
            branching_ratio = open_nodes / max(float(explored_nodes), 1.0)
            if branching_ratio > 0.5:
                return "bushy"

        current_depth = _safe_float(state.get("current_depth"), None)
        max_depth = _safe_float(state.get("max_depth"), None)
        if current_depth is not None and max_depth is not None and max_depth > 0:
            if current_depth > 0.7 * max_depth:
                return "deep"

        # Simple fallback when open nodes and max depth are unavailable.
        if current_depth is None:
            return "unknown"
        if current_depth < 10:
            return "shallow"
        if current_depth >= 50:
            return "deep"
        return "balanced"

    def _ensure_leaf_shape(self, leaf):
        leaf.setdefault("total_leaf_visits", 0.0)
        leaf.setdefault("passive_exposure_count", 0)
        leaf.setdefault("rules", {})
        leaf.setdefault("last_updated", None)

    def _ensure_rule_entry(self, leaf, rule):
        self._ensure_leaf_shape(leaf)
        normalized = normalize_rule(rule, fallback=FALLBACK_RULE)
        rules = leaf["rules"]
        if normalized not in rules:
            rules[normalized] = {
                "llm_recommend_count": 0,
                "adopt_count": 0,
                "exposure_count": 0,
                "n_effective": 0.0,
                "mean_reward": 0.0,
                "reward_variance": 0.0,
                "last_reward": 0.0,
                "success_count": 0,
                "mean_gap_reward": 0.0,
                "mean_bound_reward": 0.0,
                "mean_prune_reward": 0.0,
                "mean_node_penalty": 0.0,
                "mean_time_penalty": 0.0,
                "last_updated": None,
            }
        return rules[normalized]

    def _now(self):
        return time.strftime("%Y-%m-%dT%H:%M:%S")
