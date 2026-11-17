TIME_LIMIT = 3600
SEEDS = [0]
OTHER_MODELS = ["extratrees_gcnn_agg", "lambdamart_khalil", "svmrank_khalil"]
LLM_MODELS = ["llm"]
INTERNAL_BRANCHERS = ["relpscost"]

RESULT_FIELDNAMES = [
    "policy",
    "seed",
    "type",
    "instance",
    "nnodes",
    "nlps",
    "stime",
    "gap",
    "status",
    "rpg",
    "ndomchgs",
    "ncutoffs",
    "walltime",
    "proctime",
    "llm_query_time",
]


DEFAULT_INSTANCE_SIZE = "small"
DEFAULT_INSTANCE_LIMIT = 10

# Central registry for benchmark instance families. The tuple stores
# (path_template, available_count). Runtime --limit can select a smaller prefix.
INSTANCE_SPECS = {
    "setcover": {
        "small": ("data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i}.lp", 100),
        "medium": ("data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i}.lp", 100),
        "big": ("data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i}.lp", 100),
    },
    "cauctions": {
        "small": ("data/instances/cauctions/transfer_100_500/instance_{i}.lp", 100),
        "medium": ("data/instances/cauctions/transfer_200_1000/instance_{i}.lp", 100),
        "big": ("data/instances/cauctions/transfer_300_1500/instance_{i}.lp", 100),
    },
    "facilities": {
        "small": ("data/instances/facilities/transfer_100_100_5/instance_{i}.lp", 100),
        "medium": ("data/instances/facilities/transfer_200_100_5/instance_{i}.lp", 100),
        "big": ("data/instances/facilities/transfer_400_100_5/instance_{i}.lp", 100),
    },
    "indset": {
        "small": ("data/instances/indset/transfer_500_4/instance_{i}.lp", 100),
        "medium": ("data/instances/indset/transfer_1000_4/instance_{i}.lp", 100),
        "big": ("data/instances/indset/transfer_1500_4/instance_{i}.lp", 100),
    },
}


def build_instances(problem, size=DEFAULT_INSTANCE_SIZE, limit=DEFAULT_INSTANCE_LIMIT):
    """Build benchmark instance paths for one problem type and scale.

    ``size`` defaults to small so running ``python 07_llm-evaluate.py cauctions``
    remains simple and predictable. ``limit`` selects the first N instances from
    that family, capped by the number available in the registry.
    """
    if problem not in INSTANCE_SPECS:
        raise NotImplementedError("unknown problem: %s" % problem)

    size = size or DEFAULT_INSTANCE_SIZE
    if size not in INSTANCE_SPECS[problem]:
        available = ", ".join(sorted(INSTANCE_SPECS[problem].keys()))
        raise ValueError("unknown size '%s' for %s; available: %s" % (size, problem, available))

    template, available_count = INSTANCE_SPECS[problem][size]
    if limit is None:
        selected_count = available_count
    else:
        selected_count = max(0, min(int(limit), available_count))

    return [
        {
            "type": size,
            "path": template.format(i=i + 1),
        }
        for i in range(selected_count)
    ]


def build_branching_policies(llm_agent, seeds=SEEDS):
    policies = []
    for brancher in INTERNAL_BRANCHERS:
        for seed in seeds:
            policies.append({
                "type": "internal",
                "name": brancher,
                "seed": seed,
            })

    for model in LLM_MODELS:
        for seed in seeds:
            policies.append({
                "type": "llm-dynamic",
                "name": model,
                "model": llm_agent,
                "seed": seed,
            })
    return policies


def load_ml_competitor_policies(policies, svmrank_module, pickle_module):
    for policy in policies:
        if policy["type"] != "ml-competitor":
            continue
        try:
            with open(f"{policy['model']}/normalization.pkl", "rb") as file:
                policy["feat_shift"], policy["feat_scale"] = pickle_module.load(file)
        except Exception:
            policy["feat_shift"], policy["feat_scale"] = 0, 1

        with open(f"{policy['model']}/feat_specs.pkl", "rb") as file:
            policy["feat_specs"] = pickle_module.load(file)

        if policy["name"].startswith("svmrank"):
            policy["model"] = svmrank_module.Model().read(f"{policy['model']}/model.txt")
        else:
            with open(f"{policy['model']}/model.pkl", "rb") as file:
                policy["model"] = pickle_module.load(file)
