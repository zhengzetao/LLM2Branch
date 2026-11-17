def calculate_relative_primal_gap(model):
    primal = model.getPrimalbound()
    dual = model.getDualbound()
    if abs(primal) < 1e-9:
        return float("inf")
    return abs(primal - dual) / max(1.0, abs(primal))


def collect_solver_metrics(model, brancher, walltime, proctime):
    rel_primal_gap = calculate_relative_primal_gap(model)
    return {
        "nnodes": model.getNNodes(),
        "nlps": model.getNLPs(),
        "stime": model.getSolvingTime(),
        "gap": model.getGap(),
        "status": model.getStatus(),
        "ndomchgs": brancher.ndomchgs,
        "ncutoffs": brancher.ncutoffs,
        "rpg": rel_primal_gap,
        "walltime": walltime,
        "proctime": proctime,
        "llm_query_time": getattr(brancher, "total_llm_query_time", 0.0),
    }


def print_solver_metrics(policy, instance, metrics):
    print("=== Solver Result Metrics ===")
    print(f"Policy: {policy['type']}:{policy['name']}")
    print(f"Seed: {policy['seed']}")
    print(f"Instance: {instance['path']}")
    print(f"Number of Nodes: {metrics['nnodes']}")
    print(f"Number of LP Solves: {metrics['nlps']}")
    print(f"SCIP Solving Time: {metrics['stime']:.2f} s")
    print(f"LLM Query Time: {metrics['llm_query_time']:.2f} s")
    print(f"Wall Time: {metrics['walltime']:.2f} s")
    print(f"Process Time: {metrics['proctime']:.2f} s")
    print(f"Optimality Gap: {metrics['gap']:.6f}")
    print(f"Relative Primal Gap: {metrics['rpg']:.6f}")
    print(f"Solving Status: {metrics['status']}")
    print(f"Number of Domain Changes: {metrics['ndomchgs']}")
    print(f"Number of Cutoffs: {metrics['ncutoffs']}")
    print(f"Fair Node Count: {metrics['nnodes'] + 2 * (metrics['ndomchgs'] + metrics['ncutoffs'])}")
    print("=" * 30)


def build_csv_row(policy, instance, metrics):
    return {
        "policy": f"{policy['type']}:{policy['name']}",
        "seed": policy["seed"],
        "type": instance["type"],
        "instance": instance["path"],
        "nnodes": metrics["nnodes"],
        "nlps": metrics["nlps"],
        "stime": metrics["stime"],
        "rpg": metrics["rpg"],
        "gap": metrics["gap"],
        "status": metrics["status"],
        "ndomchgs": metrics["ndomchgs"],
        "ncutoffs": metrics["ncutoffs"],
        "walltime": metrics["walltime"],
        "proctime": metrics["proctime"],
        "llm_query_time": metrics["llm_query_time"],
    }


def build_summary_item(policy, instance, metrics):
    return {
        "policy_type": policy["type"],
        "policy_name": policy["name"],
        "seed": policy["seed"],
        "instance_type": instance["type"],
        "instance_path": instance["path"],
        "nnodes": float(metrics["nnodes"]),
        "nlps": float(metrics["nlps"]),
        "stime": float(metrics["stime"]),
        "gap": float(metrics["gap"]),
        "rpg": float(metrics["rpg"]),
        "status": str(metrics["status"]),
        "ndomchgs": float(metrics["ndomchgs"]),
        "ncutoffs": float(metrics["ncutoffs"]),
        "walltime": float(metrics["walltime"]),
        "proctime": float(metrics["proctime"]),
        "llm_query_time": float(metrics["llm_query_time"]),
    }
