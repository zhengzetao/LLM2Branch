import datetime
import numpy as np
import scipy.sparse as sp
import pyscipopt as scip
from collections import defaultdict
import pickle
import gzip
import math
import csv
import time
import pdb
import re

def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def record_step_result(problem_type, instance_path, step_records):
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    m = re.search(r'([\\/][^\\/]+)\.lp$', instance_path)  # 匹配最后一个文件名（含前导分隔符）
    name = m.group(0)[1:-3] if m else None       # -> 'instance_26'

    step_fieldnames = [
        'problem_type_and_index', 'step', 'current_policy', 'nnodes', 'nlps',
        'stime', 'gap', 'status', 'ndomchgs', 'ncutoffs'
    ]
    step_result_file = f"{name}_steps_{timestamp}.csv"
    # with open(f'experiment_results/{step_result_file}', 'w', newline='') as f:
    #     csv.DictWriter(f, fieldnames=step_fieldnames).writeheader()
    with open(f'experiment_results/{step_result_file}', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=step_fieldnames)
        writer.writeheader()
        for step_rec in step_records:
            step_rec.update({
                'problem_type_and_index': problem_type
            })
            writer.writerow(step_rec)

def init_scip_params(model, seed, heuristics=True, presolving=True, separating=True, conflict=True):

    seed = seed % 2147483648  # SCIP seed range

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)

    # no restart
    model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable separating (cuts)
    if not separating:
        model.setIntParam('separating/maxroundsroot', 0)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)


def extract_state(model, buffer=None):
    """
    Compute a bipartite graph representation of the solver. In this
    representation, the variables and constraints of the MILP are the
    left- and right-hand side nodes, and an edge links two nodes iff the
    variable is involved in the constraint. Both the nodes and edges carry
    features.

    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    buffer : dict
        A buffer to avoid re-extracting redundant information from the solver
        each time.
    Returns
    -------
    variable_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the variable nodes in the bipartite graph.
    edge_features : dictionary of type ('names': list, 'indices': np.ndarray, 'values': np.ndarray}
        The features associated with the edges in the bipartite graph.
        This is given as a sparse matrix in COO format.
    constraint_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the constraint nodes in the bipartite graph.
    """
    if buffer is None or model.getNNodes() == 1:
        buffer = {}

    # update state from buffer if any
    s = model.getState(buffer['scip_state'] if 'scip_state' in buffer else None)
    buffer['scip_state'] = s

    if 'state' in buffer:
        obj_norm = buffer['state']['obj_norm']
    else:
        obj_norm = np.linalg.norm(s['col']['coefs'])
        obj_norm = 1 if obj_norm <= 0 else obj_norm

    row_norms = s['row']['norms']
    row_norms[row_norms == 0] = 1

    # Column features
    n_cols = len(s['col']['types'])

    if 'state' in buffer:
        col_feats = buffer['state']['col_feats']
    else:
        col_feats = {}
        col_feats['type'] = np.zeros((n_cols, 4))  # BINARY INTEGER IMPLINT CONTINUOUS
        col_feats['type'][np.arange(n_cols), s['col']['types']] = 1
        col_feats['coef_normalized'] = s['col']['coefs'].reshape(-1, 1) / obj_norm

    col_feats['has_lb'] = ~np.isnan(s['col']['lbs']).reshape(-1, 1)
    col_feats['has_ub'] = ~np.isnan(s['col']['ubs']).reshape(-1, 1)
    col_feats['sol_is_at_lb'] = s['col']['sol_is_at_lb'].reshape(-1, 1)
    col_feats['sol_is_at_ub'] = s['col']['sol_is_at_ub'].reshape(-1, 1)
    col_feats['sol_frac'] = s['col']['solfracs'].reshape(-1, 1)
    col_feats['sol_frac'][s['col']['types'] == 3] = 0  # continuous have no fractionality
    col_feats['basis_status'] = np.zeros((n_cols, 4))  # LOWER BASIC UPPER ZERO
    col_feats['basis_status'][np.arange(n_cols), s['col']['basestats']] = 1
    col_feats['reduced_cost'] = s['col']['redcosts'].reshape(-1, 1) / obj_norm
    col_feats['age'] = s['col']['ages'].reshape(-1, 1) / (s['stats']['nlps'] + 5)
    col_feats['sol_val'] = s['col']['solvals'].reshape(-1, 1)
    col_feats['inc_val'] = s['col']['incvals'].reshape(-1, 1)
    col_feats['avg_inc_val'] = s['col']['avgincvals'].reshape(-1, 1)

    col_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in col_feats.items()]
    col_feat_names = [n for names in col_feat_names for n in names]
    col_feat_vals = np.concatenate(list(col_feats.values()), axis=-1)

    variable_features = {
        'names': col_feat_names,
        'values': col_feat_vals,}

    # Row features

    if 'state' in buffer:
        row_feats = buffer['state']['row_feats']
        has_lhs = buffer['state']['has_lhs']
        has_rhs = buffer['state']['has_rhs']
    else:
        row_feats = {}
        has_lhs = np.nonzero(~np.isnan(s['row']['lhss']))[0]
        has_rhs = np.nonzero(~np.isnan(s['row']['rhss']))[0]
        row_feats['obj_cosine_similarity'] = np.concatenate((
            -s['row']['objcossims'][has_lhs],
            +s['row']['objcossims'][has_rhs])).reshape(-1, 1)
        row_feats['bias'] = np.concatenate((
            -(s['row']['lhss'] / row_norms)[has_lhs],
            +(s['row']['rhss'] / row_norms)[has_rhs])).reshape(-1, 1)

    row_feats['is_tight'] = np.concatenate((
        s['row']['is_at_lhs'][has_lhs],
        s['row']['is_at_rhs'][has_rhs])).reshape(-1, 1)

    row_feats['age'] = np.concatenate((
        s['row']['ages'][has_lhs],
        s['row']['ages'][has_rhs])).reshape(-1, 1) / (s['stats']['nlps'] + 5)

    # # redundant with is_tight
    # tmp = s['row']['basestats']  # LOWER BASIC UPPER ZERO
    # tmp[s['row']['lhss'] == s['row']['rhss']] = 4  # LOWER == UPPER for equality constraints
    # tmp_l = tmp[has_lhs]
    # tmp_l[tmp_l == 2] = 1  # LHS UPPER -> BASIC
    # tmp_l[tmp_l == 4] = 2  # EQU UPPER -> UPPER
    # tmp_l[tmp_l == 0] = 2  # LHS LOWER -> UPPER
    # tmp_r = tmp[has_rhs]
    # tmp_r[tmp_r == 0] = 1  # RHS LOWER -> BASIC
    # tmp_r[tmp_r == 4] = 2  # EQU LOWER -> UPPER
    # tmp = np.concatenate((tmp_l, tmp_r)) - 1  # BASIC UPPER ZERO
    # row_feats['basis_status'] = np.zeros((len(has_lhs) + len(has_rhs), 3))
    # row_feats['basis_status'][np.arange(len(has_lhs) + len(has_rhs)), tmp] = 1

    tmp = s['row']['dualsols'] / (row_norms * obj_norm)
    row_feats['dualsol_val_normalized'] = np.concatenate((
            -tmp[has_lhs],
            +tmp[has_rhs])).reshape(-1, 1)

    row_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in row_feats.items()]
    row_feat_names = [n for names in row_feat_names for n in names]
    row_feat_vals = np.concatenate(list(row_feats.values()), axis=-1)

    constraint_features = {
        'names': row_feat_names,
        'values': row_feat_vals,}

    # Edge features
    if 'state' in buffer:
        edge_row_idxs = buffer['state']['edge_row_idxs']
        edge_col_idxs = buffer['state']['edge_col_idxs']
        edge_feats = buffer['state']['edge_feats']
    else:
        coef_matrix = sp.csr_matrix(
            (s['nzrcoef']['vals'] / row_norms[s['nzrcoef']['rowidxs']],
            (s['nzrcoef']['rowidxs'], s['nzrcoef']['colidxs'])),
            shape=(len(s['row']['nnzrs']), len(s['col']['types'])))
        coef_matrix = sp.vstack((
            -coef_matrix[has_lhs, :],
            coef_matrix[has_rhs, :])).tocoo(copy=False)

        edge_row_idxs, edge_col_idxs = coef_matrix.row, coef_matrix.col
        edge_feats = {}

        edge_feats['coef_normalized'] = coef_matrix.data.reshape(-1, 1)

    edge_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in edge_feats.items()]
    edge_feat_names = [n for names in edge_feat_names for n in names]
    edge_feat_indices = np.vstack([edge_row_idxs, edge_col_idxs])
    edge_feat_vals = np.concatenate(list(edge_feats.values()), axis=-1)

    edge_features = {
        'names': edge_feat_names,
        'indices': edge_feat_indices,
        'values': edge_feat_vals,}

    if 'state' not in buffer:
        buffer['state'] = {
            'obj_norm': obj_norm,
            'col_feats': col_feats,
            'row_feats': row_feats,
            'has_lhs': has_lhs,
            'has_rhs': has_rhs,
            'edge_row_idxs': edge_row_idxs,
            'edge_col_idxs': edge_col_idxs,
            'edge_feats': edge_feats,
        }

    return constraint_features, edge_features, variable_features

def estimate_nnz(model):
    nnz = 0
    for con in model.getConss():
        try:
            vars_in_con = model.getValsLinear(con).keys()
            nnz += len(vars_in_con)
        except Exception:
            # 如果不是线性约束（比如指示/非线性约束），跳过
            continue
    return nnz

def extract_problem_info(model):
    all_vars = model.getVars()
    num_vars = len(all_vars)
    num_bin_vars = sum(1 for v in all_vars if v.vtype() == "BINARY")
    num_int_vars = sum(1 for v in all_vars if v.vtype() == "INTEGER")
    num_cont_vars = sum(1 for v in all_vars if v.vtype() == "CONTINUOUS")

    conss = model.getConss()
    num_conss = len(conss)
    has_indicator = any("indicator" in con.name.lower() for con in model.getConss())

    # 判断是否是带整数变量的 MIP
    is_mip = (num_int_vars + num_bin_vars) > 0

    # 是否有 quadratic objective
    has_quadratic_obj = model.getObjective().degree() >= 2

    # 稀疏度估计
    total_nnz = estimate_nnz(model)
    avg_nnz_per_row = total_nnz / num_conss if num_conss > 0 else 0

    return {
         "total_number_of_variables": num_vars,
        "number_of_binary_variables": num_bin_vars,
        "number_of_integer_variables": num_int_vars,
        "number_of_continuous_variables": num_cont_vars,
        "total_number_of_constraints": num_conss,
        "average_nonzeros_per_constraint": avg_nnz_per_row,
        "contains_indicator_constraints": has_indicator,
        "has_quadratic_objective_function": has_quadratic_obj,
        "is_mixed_integer_program": is_mip
    }

def collect_tree_state(model, ndomchgs=0, ncutoffs=0):
    node = model.getCurrentNode()
    depth = node.getDepth() if node is not None else 0

    try:
        n_nodes = model.getNNodes()
    except Exception:
        n_nodes = 0

    # 界与 gap（更稳：考虑无穷与除零）
    try:
        pb = model.getPrimalbound()
        db = model.getDualbound()
        if any(map(lambda x: x is None or math.isinf(x) or math.isnan(x), [pb, db])):
            gap = None
        else:
            denom = max(1e-9, abs(pb))
            gap = max(0.0, (pb - db) / denom)
    except Exception:
        pb, db, gap = None, None, None

    cutoff_ratio = (ncutoffs or 0) / float(1 + n_nodes)
    domchg_ratio = (ndomchgs or 0) / float(1 + n_nodes)

    # 候选与“均匀熵”
    try:
        cands, *_ = model.getPseudoBranchCands()
        num_cands = len(cands)
    except Exception:
        cands, num_cands = None, 0

    # 均匀分布下的熵：-∑ p log p = log(N)
    entropy = float(np.log(num_cands)) if num_cands > 0 else 0.0

    return {
        "depth": depth,
        "nodes_explored": n_nodes,
        "primal_bound": pb,
        "dual_bound": db,
        "gap": gap,
        "cutoff_ratio": cutoff_ratio,
        "domchg_ratio": domchg_ratio,
        "num_branch_candidates": num_cands,
        "entropy": entropy,
    }

def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed


def compute_extended_variable_features(state, candidates):
    """
    Utility to extract variable features only from a bipartite state representation.

    Parameters
    ----------
    state : dict
        A bipartite state representation.
    candidates: list of ints
        List of candidate variables for which to compute features (given as indexes).

    Returns
    -------
    variable_states : np.array
        The resulting variable states.
    """
    constraint_features, edge_features, variable_features = state
    constraint_features = constraint_features['values']
    edge_indices = edge_features['indices']
    edge_features = edge_features['values']
    variable_features = variable_features['values']

    cand_states = np.zeros((
        len(candidates),
        variable_features.shape[1] + 3*(edge_features.shape[1] + constraint_features.shape[1]),
    ))

    # re-order edges according to variable index
    edge_ordering = edge_indices[1].argsort()
    edge_indices = edge_indices[:, edge_ordering]
    edge_features = edge_features[edge_ordering]

    # gather (ordered) neighbourhood features
    nbr_feats = np.concatenate([
        edge_features,
        constraint_features[edge_indices[0]]
    ], axis=1)

    # split neighborhood features by variable, along with the corresponding variable
    var_cuts = np.diff(edge_indices[1]).nonzero()[0]+1
    nbr_feats = np.split(nbr_feats, var_cuts)
    nbr_vars = np.split(edge_indices[1], var_cuts)
    assert all([all(vs[0] == vs) for vs in nbr_vars])
    nbr_vars = [vs[0] for vs in nbr_vars]

    # process candidate variable neighborhoods only
    for var, nbr_id, cand_id in zip(*np.intersect1d(nbr_vars, candidates, return_indices=True)):
        cand_states[cand_id, :] = np.concatenate([
            variable_features[var, :],
            nbr_feats[nbr_id].min(axis=0),
            nbr_feats[nbr_id].mean(axis=0),
            nbr_feats[nbr_id].max(axis=0)])

    cand_states[np.isnan(cand_states)] = 0

    return cand_states


def extract_khalil_variable_features(model, candidates, root_buffer):
    """
    Extract features following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.

    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    candidates : list of pyscipopt.scip.Variable's
        A list of variables for which to compute the variable features.
    root_buffer : dict
        A buffer to avoid re-extracting redundant root node information (None to deactivate buffering).

    Returns
    -------
    variable_features : 2D np.ndarray
        The features associated with the candidate variables.
    """
    # update state from state_buffer if any
    scip_state = model.getKhalilState(root_buffer, candidates)

    variable_feature_names = sorted(scip_state)
    variable_features = np.stack([scip_state[feature_name] for feature_name in variable_feature_names], axis=1)

    return variable_features

def extract_variable_features_for_llm(model, candidates, root_buffer):
    """
    Extract part of features following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming for llm.

    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    candidates : list of pyscipopt.scip.Variable's
        A list of variables for which to compute the variable features.
    root_buffer : dict
        A buffer to avoid re-extracting redundant root node information (None to deactivate buffering).

    Returns
    -------
    variable_features : 2D np.ndarray
        The features associated with the candidate variables.
    """
    # update state from state_buffer if any
    filter_features = ['coefs', 'nnzrs', 'root_cdeg_mean', 'root_pcoefs_mean', 'root_ncoefs_mean', 'solfracs', 'ps_up',
    'ps_down', 'nb_up_infeas', 'nb_down_infeas', 'cdeg_mean', 'cdeg_mean_ratio', 'prhs_ratio_max', 'acons_mean1', 'acons_max1']
    scip_state = model.getKhalilState(root_buffer, candidates)
    # variable_feature_names = sorted(scip_state)
    variable_features = np.stack([scip_state[feature_name] for feature_name in filter_features], axis=1)

    return variable_features


def preprocess_variable_features(features, interaction_augmentation, normalization):
    """
    Features preprocessing following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.

    Parameters
    ----------
    features : 2D np.ndarray
        The candidate variable features to preprocess.
    interaction_augmentation : bool
        Whether to augment features with 2-degree interactions (useful for linear models such as SVMs).
    normalization : bool
        Wether to normalize features in [0, 1] (i.e., query-based normalization).

    Returns
    -------
    variable_features : 2D np.ndarray
        The preprocessed variable features.
    """
    # 2-degree polynomial feature augmentation
    if interaction_augmentation:
        interactions = (
            np.expand_dims(features, axis=-1) * \
            np.expand_dims(features, axis=-2)
        ).reshape((features.shape[0], -1))
        features = np.concatenate([features, interactions], axis=1)

    # query-based normalization in [0, 1]
    if normalization:
        features -= features.min(axis=0, keepdims=True)
        max_val = features.max(axis=0, keepdims=True)
        max_val[max_val == 0] = 1
        features /= max_val

    return features


def load_flat_samples(filename, feat_type, label_type, augment_feats, normalize_feats):
    with gzip.open(filename, 'rb') as file:
        sample = pickle.load(file)

    state, khalil_state, best_cand, cands, cand_scores = sample['data']

    cands = np.array(cands)
    cand_scores = np.array(cand_scores)

    cand_states = []
    if feat_type in ('all', 'gcnn_agg'):
        cand_states.append(compute_extended_variable_features(state, cands))
    if feat_type in ('all', 'khalil'):
        cand_states.append(khalil_state)
    cand_states = np.concatenate(cand_states, axis=1)

    best_cand_idx = np.where(cands == best_cand)[0][0]

    # feature preprocessing
    cand_states = preprocess_variable_features(cand_states, interaction_augmentation=augment_feats, normalization=normalize_feats)

    if label_type == 'scores':
        cand_labels = cand_scores

    elif label_type == 'ranks':
        cand_labels = np.empty(len(cand_scores), dtype=int)
        cand_labels[cand_scores.argsort()] = np.arange(len(cand_scores))

    elif label_type == 'bipartite_ranks':
        # scores quantile discretization as in
        # Khalil et al. (2016) Learning to Branch in Mixed Integer Programming
        cand_labels = np.empty(len(cand_scores), dtype=int)
        cand_labels[cand_scores >= 0.8 * cand_scores.max()] = 1
        cand_labels[cand_scores < 0.8 * cand_scores.max()] = 0

    else:
        raise ValueError(f"Invalid label type: '{label_type}'")

    return cand_states, cand_labels, best_cand_idx

# def result_summary(results, csvfile=None):
#     # ====== 汇总：按 (policy_type, instance_type) 取均值 ======
#     numeric_keys = ['nnodes', 'nlps', 'stime', 'gap', 'ndomchgs', 'ncutoffs', 'walltime', 'proctime']
#     group_sums = defaultdict(lambda: {k: 0.0 for k in numeric_keys})
#     group_counts = defaultdict(int)
#     group_solved = defaultdict(int)  # 可选：统计“optimal”比例

#     for row in results:
#         key = (row['policy_type'], row['instance_type'])
#         for k in numeric_keys:
#             group_sums[key][k] += row[k]
#         group_counts[key] += 1
#         if row['status'].lower() in ('optimal', 'solved', 'optimal/solved'):
#             group_solved[key] += 1

#     summary_rows = []
#     print("\n==== Averages by (policy_type, instance_type) ====")
#     for key, sums in group_sums.items():
#         count = group_counts[key]
#         avg = {k: (sums[k] / count if count else float('nan')) for k in numeric_keys}
#         solved_ratio = group_solved[key] / count if count else 0.0
#         policy_type, instance_type = key
#         summary_rows.append({
#             'policy_type': policy_type,
#             'instance_type': instance_type,
#             'n': count,
#             **{f"avg_{k}": avg[k] for k in numeric_keys},
#             'solved_ratio': solved_ratio,
#         })
#     # 直接写到同一个文件末尾（如果传入了 csvfile）
#     if csvfile and summary_rows:
#         # 先写分隔行和标题
#         raw_writer = csv.writer(csvfile)
#         raw_writer.writerow([])
#         raw_writer.writerow(["==== SUMMARY (policy_type, instance_type) ===="])

#         # 用新的 DictWriter（表头不同）
#         sum_fields = list(summary_rows[0].keys())
#         sum_writer = csv.DictWriter(csvfile, fieldnames=sum_fields)
#         sum_writer.writeheader()
#         sum_writer.writerows(summary_rows)
#         csvfile.flush()
#     return summary_rows


def result_summary(results, csvfile=None):
    numeric_keys = ['nnodes', 'nlps', 'stime', 'llm_query_time', 'gap', 'ndomchgs', 'rpg',
                    'ncutoffs', 'walltime', 'proctime']
    group_values = defaultdict(lambda: {k: [] for k in numeric_keys})
    group_counts = defaultdict(int)
    group_solved = defaultdict(int)

    # 先收集所有数值
    for row in results:
        key = (row['policy_type'], row['instance_type'])
        for k in numeric_keys:
            group_values[key][k].append(float(row[k]))
        group_counts[key] += 1
        if str(row['status']).lower() in ('optimal', 'solved', 'optimal/solved'):
            group_solved[key] += 1

    summary_rows = []
    print("\n==== Trimmed Averages (drop min & max) by (policy_type, instance_type) ====")
    for key, values in group_values.items():
        count = group_counts[key]
        avg = {}
        for k, arr in values.items():
            if len(arr) > 2:
                arr_sorted = sorted(arr)
                trimmed = arr_sorted[1:-1]  # 去掉 min 和 max
                avg[k] = sum(trimmed) / len(trimmed)
            elif arr:  # 如果样本数太少，退化为普通均值
                avg[k] = sum(arr) / len(arr)
            else:
                avg[k] = float('nan')

        solved_ratio = group_solved[key] / count if count else 0.0
        policy_type, instance_type = key
        summary_rows.append({
            'policy_type': policy_type,
            'instance_type': instance_type,
            'n': count,
            **{f"trimmed_avg_{k}": avg[k] for k in numeric_keys},
            'solved_ratio': solved_ratio,
        })

    # 写入 CSV
    if csvfile and summary_rows:
        raw_writer = csv.writer(csvfile)
        raw_writer.writerow([])
        raw_writer.writerow(["==== TRIMMED SUMMARY (policy_type, instance_type) ===="])
        sum_fields = list(summary_rows[0].keys())
        sum_writer = csv.DictWriter(csvfile, fieldnames=sum_fields)
        sum_writer.writeheader()
        sum_writer.writerows(summary_rows)
        csvfile.flush()

    return summary_rows


def dynamic_min_hold_nodes(num_vars, num_constraints,
                           base=50, a=30, b=30, lo=50, hi=2000):
    """
    根据实例规模（变量/约束数）自适应地确定“至少持有多少节点”。
    对数缩放，避免小实例过大或大实例过小。
    """
    scaled = base + a * math.log1p(max(1, num_vars)) + b * math.log1p(max(1, num_constraints))
    return int(max(lo, min(hi, scaled)))

class RuleSwitchController:
    """
    负责防抖：最小持有(min_hold)、冷却窗(cooldown)、连续确认(hysteresis)、停滞触发(stagnation)。
    用法：
      - 每步调用 update_progress(step, gap)
      - 收到 LLM 决策后调用 allow_switch(...)
      - 真正切换后调用 commit_switch(...)
    """
    def __init__(self, num_vars, num_constraints,
                 consecutive_required=2,       # 连续K次相同建议才切换
                 cooldown_factor=0.8,          # 冷却窗 = cooldown_factor * min_hold
                 stagnation_factor=0.6,        # 停滞触发阈值 = stagnation_factor * min_hold
                 min_gap_improve=1e-3):        # 认为“gap有改进”的最小下降
        self.num_vars = num_vars
        self.num_constraints = num_constraints
        self.min_hold = dynamic_min_hold_nodes(num_vars, num_constraints)
        # self.cooldown = int(self.min_hold * cooldown_factor)
        self.cooldown = int(2000)
        self.stagnation_nodes = int(800)
        # self.stagnation_nodes = int(self.min_hold * stagnation_factor)
        self.min_gap_improve = float(min_gap_improve)

        self.last_switch_step = -10**9
        self.last_rule = None
        self.rule_last_used_at = {}     # rule -> last step used
        self.pending_target = None      # 等待连续确认的目标rule
        self.pending_count = 0          # 连续同意计数

        self.best_gap = float('inf')
        self.last_improve_step = 0
        self.consecutive_required = int(consecutive_required)

    def update_progress(self, step, gap):
        if gap is None:
            return
        # 只有当 gap 明显下降才认为“有改进”
        if gap + 1e-12 < self.best_gap - self.min_gap_improve:
            self.best_gap = gap
            self.last_improve_step = step

    def allow_switch(self, step, current_rule, llm_next_rule, llm_switch_flag):
        """
        返回 (should_switch, target_rule)。
        - step: 当前累计节点（可用 model.getNNodes() 或你的 step 计数）
        - current_rule: 当前在用的 rule
        - llm_next_rule/llm_switch_flag: LLM 的建议
        """
        # LLM 不建议切换，清空待确认
        if not llm_switch_flag:
            self.pending_target = None
            self.pending_count = 0
            return False, current_rule

        # 建议切换到同名，无意义
        if llm_next_rule == current_rule:
            self.pending_target = None
            self.pending_count = 0
            return False, current_rule

        # 1) 最小持有
        # if step - self.last_switch_step < self.min_hold:
        #     return False, current_rule

        # 2) 冷却窗：禁止短期来回切
        last_used_target = self.rule_last_used_at.get(llm_next_rule, -10**9)
        if step - last_used_target < self.cooldown:
            return False, current_rule

        # 3) 停滞触发：gap 长时间无明显改善才允许切
        if step - self.last_improve_step < self.stagnation_nodes:
            return False, current_rule

        # 4) 连续确认（hysteresis）
        # if self.pending_target == llm_next_rule:
        #     self.pending_count += 1
        # else:
        #     self.pending_target = llm_next_rule
        #     self.pending_count = 1

        # if self.pending_count < self.consecutive_required:
        #     return False, current_rule

        # 满足所有条件，可以切换
        print('step:',step,'last_used_target:',last_used_target,'last_improved_step,',self.last_improve_step)
        return True, llm_next_rule

    def commit_switch(self, step, new_rule):
        self.last_switch_step = step
        self.last_rule = new_rule
        self.rule_last_used_at[new_rule] = step
        self.pending_target = None
        self.pending_count = 0


# from collections import deque

# class LLMQueryScheduler:
#     """
#     Maintain rolling metrics from SCIP and decide when to trigger an LLM
#     based on trends (cutoff drop, gap stall, cost-quality tradeoff, etc.).
#     """

#     def __init__(
#         self,
#         win: int = 50,            # 滑动窗口长度
#         N_min: int = 100,          # 触发的最小步间隔
#         N_max: int = 200,          # 超过则强制评估一次
#         tau_cutoff_drop: float = 0.15,  # 剪枝率相对下降触发阈值
#         stall_steps: int = 12,     # 判定gap停滞的最近步数
#         small_cands: int = 6,      # 候选很少的阈值（倾向轻量规则）
#         pc_rely_min: int = 4       # 伪代价“可靠性”近似阈值
#     ):
#         self.win = win
#         self.hist = deque(maxlen=win)

#         self.N_min = N_min
#         self.N_max = N_max
#         self.TAU_CUTOFF_DROP = tau_cutoff_drop
#         self.STALL_STEPS = stall_steps
#         self.SMALL_CANDS = small_cands
#         self.PC_RELY_MIN = pc_rely_min

#     def reset(self):
#         self.hist.clear()

#     def sample(self, model, step_counter: int) -> dict:
#         """
#         采样当前节点/模型的关键指标并加入滑动窗口。
#         注意：这里的 pc_reliability 是近似值（用候选数代替）。
#         如有真实的 ps_up/ps_down 计数，可在此替换。
#         """
#         depth = model.getCurrentNode().getDepth()
#         n_nodes = max(1, model.getNNodes())
#         # cutoff_ratio = model.getNCutoffNodes() / n_nodes
#         gap = max(0.0, min(1.0, model.getGap()))
#         cands, *_ = model.getPseudoBranchCands()
#         cand_cnt = len(cands)

#         # 伪代价“可靠性”近似（可替换为真实统计）
#         pc_reliability = min(10, cand_cnt)

#         # 单位节点 LP 密度（代价感知）
#         nlps = model.getNLPs()
#         lp_density = nlps / n_nodes

#         snap = dict(
#             step=step_counter,
#             depth=depth,
#             # cutoff=cutoff_ratio,
#             gap=gap,
#             cand_cnt=cand_cnt,
#             pc_rel=pc_reliability,
#             lp_dens=lp_density,
#         )
#         self.hist.append(snap)
#         return snap

#     def should_query(
#         self,
#         step_counter: int,
#         last_decision_step: int,
#         start_step: int
#     ) -> bool:
#         """
#         基于滑动窗口内的趋势/阈值 + 最小/最大间隔，判断是否触发 LLM。
#         """
#         # 冷启动门槛
#         if step_counter < start_step:
#             return False

#         # 最小/最大间隔约束
#         since_last = step_counter - last_decision_step
#         if since_last < self.N_min:
#             return False

#         # 窗口不足：仅在达到最大间隔时强制评估
#         if len(self.hist) < self.win:
#             return since_last >= self.N_max

#         H = list(self.hist)
#         pre = H[:-1]
#         curr = H[-1]

#         # 剪枝率相对下降
#         # mean_cut_prev = sum(x["cutoff"] for x in pre) / len(pre)
#         # cut_drop = max(0.0, mean_cut_prev - curr["cutoff"])
#         # rel_drop = cut_drop / max(1e-6, mean_cut_prev)

#         # gap 连续停滞
#         tail = H[-min(self.STALL_STEPS, len(H)):]
#         gap_span = max(x["gap"] for x in tail) - min(x["gap"] for x in tail)
#         gap_stalled = gap_span <= 1e-6

#         # 伪代价已“可靠”但单位代价变差（lp_dens↑ 且 cutoff↓）
#         pc_reliable = curr["pc_rel"] >= self.PC_RELY_MIN
#         mean_lp_prev = sum(x["lp_dens"] for x in pre) / len(pre)
#         # cost_up_quality_down = pc_reliable and (rel_drop > self.TAU_CUTOFF_DROP) and (curr["lp_dens"] >= mean_lp_prev)

#         # 候选很少（深层/局部化 → 倾向轻量规则）
#         few_candidates = curr["cand_cnt"] <= self.SMALL_CANDS

#         return (
#             # rel_drop > self.TAU_CUTOFF_DROP  # 剪枝率显著下降
#             gap_stalled                   # gap 连续停滞
#             # or cost_up_quality_down          # 代价上升且剪枝变差
#             or few_candidates                # 深层局部化
#             or since_last >= self.N_max      # 达到最大间隔强评估
#         )

from collections import deque

class LLMQueryScheduler:
    """
    Maintain rolling metrics from SCIP and decide when to trigger an LLM
    with a one-time forced check at the root node, then metric-based checks.
    """

    def __init__(
        self,
        win: int = 50,             # 滑动窗口长度
        N_min: int = 100,          # 触发的最小步间隔
        N_max: int = 200,          # 超过则强制评估一次
        tau_cutoff_drop: float = 0.15,  # 预留：剪枝率相对下降阈值（此版本未启用）
        stall_steps: int = 12,      # 判定 gap 停滞的最近步数
        small_cands: int = 6,       # 候选很少阈值（倾向轻量规则）
        pc_rely_min: int = 4,       # 伪代价“可靠性”近似阈值
        root_force_depth: int = 0   # ≤该深度的一次性强制触发（默认仅 root=0）
    ):
        self.win = win
        self.hist = deque(maxlen=win)

        self.N_min = N_min
        self.N_max = N_max
        self.TAU_CUTOFF_DROP = tau_cutoff_drop
        self.STALL_STEPS = stall_steps
        self.SMALL_CANDS = small_cands
        self.PC_RELY_MIN = pc_rely_min

        # root 节点一次性强制判断
        self.root_force_depth = root_force_depth
        self._root_checked = False

    def reset(self):
        self.hist.clear()
        self._root_checked = False

    def sample(self, model, step_counter: int) -> dict:
        """
        采样当前节点/模型的关键指标并加入滑动窗口。
        注意：这里的 pc_reliability 是近似值（用候选数代替）。
        如有真实的 ps_up/ps_down 计数，可在此替换。
        """
        node = model.getCurrentNode()
        depth = node.getDepth() if node is not None else 0

        n_nodes = max(1, model.getNNodes())
        gap = max(0.0, min(1.0, model.getGap()))
        cands, *_ = model.getPseudoBranchCands()
        cand_cnt = len(cands)

        # 伪代价“可靠性”近似（可替换为真实统计）
        pc_reliability = min(10, cand_cnt)

        # 单位节点 LP 密度（代价感知）
        nlps = model.getNLPs()
        lp_density = nlps / n_nodes

        snap = dict(
            step=step_counter,
            depth=depth,
            gap=gap,
            cand_cnt=cand_cnt,
            pc_rel=pc_reliability,
            lp_dens=lp_density,
        )
        self.hist.append(snap)
        return snap

    def should_query(
        self,
        step_counter: int,
        last_decision_step: int,
        start_step: int
    ) -> bool:
        """
        先在 root（或 depth ≤ root_force_depth）强制触发一次；
        之后基于滑动窗口趋势 + 最小/最大间隔判断是否触发 LLM。
        """
        # --- 1) root 强制触发（只触发一次） ---
        if self.hist:
            curr = self.hist[-1]
            if (curr["depth"] <= self.root_force_depth) and (not self._root_checked):
                self._root_checked = True
                return True

        # --- 2) 常规门槛与节流 ---
        if step_counter < start_step:
            return False

        since_last = step_counter - last_decision_step
        if since_last < self.N_min:
            return False

        if len(self.hist) < self.win:
            return since_last >= self.N_max

        # --- 3) 指标趋势判断 ---
        H = list(self.hist)
        pre = H[:-1]
        curr = H[-1]

        # gap 连续停滞
        tail = H[-min(self.STALL_STEPS, len(H)):]
        gap_span = max(x["gap"] for x in tail) - min(x["gap"] for x in tail)
        gap_stalled = gap_span <= 1e-6

        # 伪代价已“可靠”且单位代价恶化（lp_dens↑）
        pc_reliable = curr["pc_rel"] >= self.PC_RELY_MIN
        mean_lp_prev = sum(x["lp_dens"] for x in pre) / len(pre)
        cost_up = curr["lp_dens"] >= mean_lp_prev  # 简化的代价上升信号

        # 候选很少（深层/局部化 → 倾向轻量规则）
        few_candidates = curr["cand_cnt"] <= self.SMALL_CANDS

        return (
            gap_stalled              # gap 连续停滞
            or (pc_reliable and cost_up)  # 代价上升且已具备可靠伪代价
            or few_candidates        # 深层局部化
            or since_last >= self.N_max
        )
