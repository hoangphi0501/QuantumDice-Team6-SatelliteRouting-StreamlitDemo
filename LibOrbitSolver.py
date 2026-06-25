import numpy as np
from math import ceil
from sklearn.cluster import KMeans
from LibNetwork import *
from LibBenchmark_Checking import *
from LibQUBO import *
from orbit.models import AdvancedSolverParams, SolverParams
from orbit.problems import BoltzmannProblem, IsingProblem, QuboProblem
from orbit.solvers import LocalSolver
from orbit.toolkit.converters import ising_to_qubo, qubo_to_ising
from orbit.toolkit.energy import get_ising_energy, get_qubo_energy
from orbit.toolkit.states import binary_to_spin, spin_to_binary


def standardize_features(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / std


def ensure_location_shape(Loc, expected_n, name="Loc"):
    """
    Ensure location matrix has shape:

        (num_nodes, coordinate_dimension)

    If input is transposed, this function transposes it.
    """
    Loc = np.asarray(Loc, dtype=float)

    if Loc.shape[0] == expected_n:
        return Loc

    if Loc.ndim == 2 and Loc.shape[1] == expected_n:
        return Loc.T

    raise ValueError(
        f"{name} has invalid shape {Loc.shape}. "
        f"Expected first or second dimension to be {expected_n}."
    )


def make_dp_dt_features(Loc_DP, Loc_DT):
    """
    Build KMeans features for each DP/DT pair.

    Feature vector:
        [DP location, DT location, midpoint]
    """
    Loc_DP = np.asarray(Loc_DP, dtype=float)
    Loc_DT = np.asarray(Loc_DT, dtype=float)

    midpoint = 0.5 * (Loc_DP + Loc_DT)

    features = np.hstack([
        Loc_DP,
        Loc_DT,
        midpoint,
    ])

    return standardize_features(features)


def split_large_cluster(cluster, max_cluster_size=5):
    cluster = list(cluster)

    return [
        cluster[i:i + max_cluster_size]
        for i in range(0, len(cluster), max_cluster_size)
    ]


def cluster_dp_dt_pairs(Loc_DP, Loc_DT, max_cluster_size=5, random_state=1):
    """
    Cluster DP/DT pairs using KMeans.

    Guarantee:
        every cluster has size <= max_cluster_size.
    """
    num_DT = len(Loc_DP)

    if num_DT <= max_cluster_size:
        return [list(range(num_DT))]

    n_clusters = int(ceil(num_DT / max_cluster_size))

    features = make_dp_dt_features(Loc_DP, Loc_DT)

    if KMeans is None:
        print("sklearn is not installed. Using sequential clustering.")

        return [
            list(range(i, min(i + max_cluster_size, num_DT)))
            for i in range(0, num_DT, max_cluster_size)
        ]

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=random_state,
    )

    labels = kmeans.fit_predict(features)

    clusters = []

    for k in range(n_clusters):
        cluster = [m for m in range(num_DT) if labels[m] == k]

        if len(cluster) == 0:
            continue

        if len(cluster) <= max_cluster_size:
            clusters.append(cluster)
        else:
            clusters.extend(split_large_cluster(cluster, max_cluster_size))

    clusters = [sorted(c) for c in clusters]
    clusters = sorted(clusters, key=lambda c: c[0])

    assert all(len(c) <= max_cluster_size for c in clusters)

    return clusters

def slice_sat_to_dt_matrix(A, cluster, num_SAT, name="matrix"):
    """
    Slice matrices whose logical shape is:

        (num_SAT, num_DT)

    Examples:
        DistDP2SAT
        DistSAT2DT
        DR_up
        DR_down

    If the input is accidentally shaped as (num_DT, num_SAT),
    this function transposes the sliced result.
    """
    A = np.asarray(A)

    cluster = list(cluster)
    max_m = max(cluster)

    if A.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {A.shape}")

    # Expected shape: (num_SAT, num_DT)
    if A.shape[0] == num_SAT and A.shape[1] > max_m:
        return A[:, cluster]

    # Alternative shape: (num_DT, num_SAT)
    if A.shape[1] == num_SAT and A.shape[0] > max_m:
        return A[cluster, :].T

    raise ValueError(
        f"{name} has invalid shape {A.shape}. "
        f"Expected shape like (num_SAT, num_DT) or (num_DT, num_SAT)."
    )


def check_sat_sat_matrix(A, num_SAT, name="matrix"):
    A = np.asarray(A)

    if A.shape != (num_SAT, num_SAT):
        raise ValueError(
            f"{name} must have shape ({num_SAT}, {num_SAT}), got {A.shape}"
        )

    return A

def local_var_to_global(var, local_to_global):
    """
    Convert local-cluster variables back to global DP indices.

    Local:
        ('xp', local_m, (local_m, s))
        ('xs', local_m, (s, i))

    Global:
        ('xp', global_m, (global_m, s))
        ('xs', global_m, (s, i))
    """
    name = var[0]
    local_m = var[1]
    edge = var[2]

    global_m = local_to_global[local_m]

    if name == "xp":
        return (name, global_m, (global_m, edge[1]))

    if name == "xs":
        return (name, global_m, edge)

    return (name, global_m, edge)


def map_solution_local_to_global(solution_local, cluster):
    local_to_global = {
        local_m: global_m
        for local_m, global_m in enumerate(cluster)
    }

    solution_global = {}

    for var, value in solution_local.items():
        if value == 1:
            global_var = local_var_to_global(var, local_to_global)
            solution_global[global_var] = 1

    return solution_global

def solve_qubo_with_orbit(
    c,
    q,
    Q,
    variables,
    full_sweeps=1000,
    beta_start=0.5,
    beta_end=4.0,
    beta_step_interval=10,
    n_replicas=300,
    n_processes=2,
):
    c_mat, q_vec, Q_mat, variables, var_to_idx = build_qubo_matrix(
        c=c,
        q=q,
        Q=Q,
        variables=variables,
    )

    Q_orbit = 0.5 * Q_mat.astype(float)
    np.fill_diagonal(Q_orbit, q_vec.astype(float))

    problem = QuboProblem(Q=Q_orbit)

    # print(f"Problem: {problem.n_pbits}-variable QUBO problem")

    params = SolverParams(
        full_sweeps=full_sweeps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_step_interval=beta_step_interval,
        advanced=AdvancedSolverParams(
            n_replicas=n_replicas,
            n_processes=n_processes,
            log_level="WARNING",
        ),
    )

    solver = LocalSolver()
    solver.connect()
    result = solver.solve(problem, params).result()
    solver.disconnect()

    best_x = np.asarray(result.min_state, dtype=int)
    best_energy_with_constant = float(c_mat + result.min_energy)

    solution_dict = {
        variables[i]: int(best_x[i])
        for i in range(len(variables))
    }

    return {
        "problem": problem,
        "result": result,
        "best_x": best_x,
        "best_energy_with_constant": best_energy_with_constant,
        "solution_dict": solution_dict,
        "variables": variables,
        "var_to_idx": var_to_idx,
        "Q_orbit": Q_orbit,
    }

# ============================================================
# Postprocess / repair clustered global solution
# ============================================================

def _sat_dt_value(A, sat, m, num_SAT=None, name="matrix"):
    """
    Safely read a matrix whose logical shape is either:

        (num_SAT, num_DT)

    or accidentally:

        (num_DT, num_SAT)
    """
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {A.shape}")

    if num_SAT is not None:
        if A.shape[0] == num_SAT:
            return A[sat, m]
        if A.shape[1] == num_SAT:
            return A[m, sat]

    # fallback
    return A[sat, m]


def _sat_sat_value(A, s, sp, name="matrix"):
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {A.shape}")

    return A[s, sp]


def repair_global_solution_two_layer(
    global_solution,
    variables_global,
    num_DT_global,
    Layer1,
    Layer2,
    DistDP2SAT,
    DistSAT2SAT,
    DistSAT2DT=None,
    num_SAT=None,
    use_dt_distance=True,
):
    """
    Repair the global clustered ORBIT solution.

    For each DP/DT pair m:
        1. Find selected Layer1 satellite s from ('xp', m, (m, s)).
        2. If no s is selected, choose the closest Layer1 satellite to DP m.
        3. If multiple s are selected, keep the closest one.
        4. Given selected s, find selected Layer2 satellite sp from ('xs', m, (s, sp)).
        5. If no sp is selected, choose the minimum-distance Layer2 satellite.
        6. If multiple sp are selected, keep the minimum-distance one.
        7. Remove all other xp/xs variables for packet m.

    Distance rule:
        s selection:
            min DistDP2SAT[s, m]

        sp selection:
            if DistSAT2DT is given and use_dt_distance=True:
                min DistSAT2SAT[s, sp] + DistSAT2DT[sp, m]
            else:
                min DistSAT2SAT[s, sp]

    Returns:
        repaired_solution : dict
            Dictionary containing only selected variables with value 1.

        repair_report : list of dict
            One row per DP/DT pair explaining what was repaired.
    """

    variable_set = set(variables_global)
    Layer1 = list(Layer1)
    Layer2 = list(Layer2)

    repaired_solution = {}
    repair_report = []

    for m in range(num_DT_global):

        # ------------------------------------------------------------
        # Step 1: repair xp selection
        # ------------------------------------------------------------
        xp_candidates = [
            s for s in Layer1
            if ("xp", m, (m, s)) in variable_set
        ]

        if len(xp_candidates) == 0:
            raise ValueError(
                f"No xp candidates found for DP m={m}. "
                "Check variables_global / Layer1."
            )

        selected_s_list = [
            s for s in xp_candidates
            if int(global_solution.get(("xp", m, (m, s)), 0)) == 1
        ]

        if len(selected_s_list) == 1:
            chosen_s = selected_s_list[0]
            xp_action = "kept"
        else:
            chosen_s = min(
                xp_candidates,
                key=lambda s: _sat_dt_value(
                    DistDP2SAT,
                    sat=s,
                    m=m,
                    num_SAT=num_SAT,
                    name="DistDP2SAT",
                ),
            )

            if len(selected_s_list) == 0:
                xp_action = "added_missing"
            else:
                xp_action = "reduced_multiple"

        repaired_solution[("xp", m, (m, chosen_s))] = 1

        # ------------------------------------------------------------
        # Step 2: repair xs selection given chosen_s
        # ------------------------------------------------------------
        xs_candidates = [
            sp for sp in Layer2
            if ("xs", m, (chosen_s, sp)) in variable_set
        ]

        if len(xs_candidates) == 0:
            raise ValueError(
                f"No xs candidates found for m={m}, chosen_s={chosen_s}. "
                "Check variables_global / Layer2."
            )

        selected_sp_list = [
            sp for sp in xs_candidates
            if int(global_solution.get(("xs", m, (chosen_s, sp)), 0)) == 1
        ]

        def sp_cost(sp):
            cost = _sat_sat_value(
                DistSAT2SAT,
                s=chosen_s,
                sp=sp,
                name="DistSAT2SAT",
            )

            if DistSAT2DT is not None and use_dt_distance:
                cost += _sat_dt_value(
                    DistSAT2DT,
                    sat=sp,
                    m=m,
                    num_SAT=num_SAT,
                    name="DistSAT2DT",
                )

            return cost

        if len(selected_sp_list) == 1:
            chosen_sp = selected_sp_list[0]
            xs_action = "kept"
        else:
            chosen_sp = min(xs_candidates, key=sp_cost)

            if len(selected_sp_list) == 0:
                xs_action = "added_missing"
            else:
                xs_action = "reduced_multiple"

        repaired_solution[("xs", m, (chosen_s, chosen_sp))] = 1

        repair_report.append({
            "m": m,
            "chosen_s_layer1": chosen_s,
            "chosen_sp_layer2": chosen_sp,
            "num_selected_s_before": len(selected_s_list),
            "num_selected_sp_before": len(selected_sp_list),
            "xp_action": xp_action,
            "xs_action": xs_action,
        })

    return repaired_solution, repair_report

def solve_clustered_orbit_routing_from_inputs(
    Loc_DP,
    Loc_DT,
    Layer1,
    Layer2,
    num_SAT,
    DistDP2SAT,
    DistSAT2DT,
    DistSAT2SAT,
    DR_up,
    DR_down,
    DR_isl,
    Im=1e9,
    Lrelay=1e-4,
    Wt=1,
    Wc=1,
    alpha_TS=0.1,
    alpha_SS=0.1,
    lam1=5,
    lam2=5,
    max_cluster_size=5,
    random_state=1,
    full_sweeps=1000,
    beta_start=0.5,
    beta_end=4.0,
    beta_step_interval=10,
    n_replicas=300,
    n_processes=2,
):
    """
    Solve large-scale DP/DT routing by clustering DP/DT pairs.

    This function assumes that all physical-layer quantities have already
    been computed outside the function.

    Inputs:
        DistDP2SAT:  shape (num_SAT, num_DT)
        DistSAT2DT:  shape (num_SAT, num_DT)
        DistSAT2SAT: shape (num_SAT, num_SAT)
        DR_up:       shape (num_SAT, num_DT)
        DR_down:     shape (num_SAT, num_DT)
        DR_isl:      shape (num_SAT, num_SAT)

    Each cluster has at most max_cluster_size DP/DT pairs.
    """

    Loc_DP = ensure_location_shape(Loc_DP, expected_n=len(Loc_DP), name="Loc_DP")
    Loc_DT = ensure_location_shape(Loc_DT, expected_n=len(Loc_DT), name="Loc_DT")

    num_DT_global = len(Loc_DP)

    DistSAT2SAT = check_sat_sat_matrix(DistSAT2SAT, num_SAT, name="DistSAT2SAT")
    DR_isl = check_sat_sat_matrix(DR_isl, num_SAT, name="DR_isl")

    clusters = cluster_dp_dt_pairs(
        Loc_DP=Loc_DP,
        Loc_DT=Loc_DT,
        max_cluster_size=max_cluster_size,
        random_state=random_state,
    )

    # print("\n=== DP/DT Clusters ===")
    # for k, cluster in enumerate(clusters):
    #     print(f"Cluster {k}: global DP indices = {cluster}, size = {len(cluster)}")

    all_cluster_results = []
    global_solution = {}

    # ------------------------------------------------------------
    # Build global variable ordering for reconstructed min_state
    # ------------------------------------------------------------
    Link_DP2SAT_global, Link_SAT2SAT_global, Link_SAT2DT_global = Func_2LayerLinks(
        num_DT_global,
        Layer1,
        Layer2,
    )

    Index_DP2SAT_global, Index_SAT2SAT_global = Func_VariableIndex(
        num_DT_global,
        Link_DP2SAT_global,
        Link_SAT2SAT_global,
        Link_SAT2DT_global,
    )

    variables_global = Index_DP2SAT_global + Index_SAT2SAT_global

    for cluster_id, cluster in enumerate(clusters):
        # print("\n" + "=" * 70)
        # print(f"Solving cluster {cluster_id}")
        # print(f"Global DP/DT indices: {cluster}")
        # print("=" * 70)

        cluster = list(cluster)
        num_DT_cluster = len(cluster)

        # ------------------------------------------------------------
        # Slice precomputed global matrices for this cluster
        # ------------------------------------------------------------
        DistDP2SAT_c = slice_sat_to_dt_matrix(
            DistDP2SAT,
            cluster,
            num_SAT,
            name="DistDP2SAT",
        )

        DistSAT2DT_c = slice_sat_to_dt_matrix(
            DistSAT2DT,
            cluster,
            num_SAT,
            name="DistSAT2DT",
        )

        DR_up_c = slice_sat_to_dt_matrix(
            DR_up,
            cluster,
            num_SAT,
            name="DR_up",
        )

        DR_down_c = slice_sat_to_dt_matrix(
            DR_down,
            cluster,
            num_SAT,
            name="DR_down",
        )

        # ------------------------------------------------------------
        # Build local variables
        # Local DP indices: 0, 1, ..., num_DT_cluster - 1
        # ------------------------------------------------------------
        Link_DP2SAT_c, Link_SAT2SAT_c, Link_SAT2DT_c = Func_2LayerLinks(
            num_DT_cluster,
            Layer1,
            Layer2,
        )

        Index_DP2SAT_c, Index_SAT2SAT_c = Func_VariableIndex(
            num_DT_cluster,
            Link_DP2SAT_c,
            Link_SAT2SAT_c,
            Link_SAT2DT_c,
        )

        variables_c = Index_DP2SAT_c + Index_SAT2SAT_c

        # print(f"Cluster {cluster_id} local variables: {len(variables_c)}")

        # ------------------------------------------------------------
        # Build local QUBO
        # ------------------------------------------------------------
        qubo_c = QUBOModel()

        qubo_c.func_tcomDpS(
            Wt,
            Index_DP2SAT_c,
            DistDP2SAT_c,
            DR_up_c,
            alpha_TS,
            Im,
        )

        qubo_c.func_tcomSS(
            Wt,
            Index_SAT2SAT_c,
            DistSAT2SAT,
            DR_isl,
            alpha_SS,
            Im,
        )

        qubo_c.func_tcomSDt(
            Wt,
            Layer2,
            Index_SAT2SAT_c,
            DistSAT2DT_c,
            DR_down_c,
            Im,
        )

        qubo_c.func_trel(
            Wt,
            num_SAT,
            Index_DP2SAT_c,
            Index_SAT2SAT_c,
            Lrelay,
        )

        qubo_c.func_linkCongestion(
            Wc,
            Layer1,
            Layer2,
            Index_SAT2SAT_c,
        )

        qubo_c.cons_begin(
            num_DT_cluster,
            Index_DP2SAT_c,
            lam1,
        )

        qubo_c.cons_equal(
            num_DT_cluster,
            Layer1,
            Index_DP2SAT_c,
            Index_SAT2SAT_c,
            lam2,
        )

        c_c, q_c, Q_c = qubo_c.get_qubo()

        # ------------------------------------------------------------
        # Solve local QUBO with ORBIT
        # ------------------------------------------------------------
        orbit_output_c = solve_qubo_with_orbit(
            c=c_c,
            q=q_c,
            Q=Q_c,
            variables=variables_c,
            full_sweeps=full_sweeps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_step_interval=beta_step_interval,
            n_replicas=n_replicas,
            n_processes=n_processes,
        )

        solution_local = orbit_output_c["solution_dict"]

        selected_local = {
            var: value
            for var, value in solution_local.items()
            if value == 1
        }

        selected_global = map_solution_local_to_global(
            solution_local=solution_local,
            cluster=cluster,
        )

        global_solution.update(selected_global)

        # print(
        #     f"Cluster {cluster_id} best energy with constant: "
        #     f"{orbit_output_c['best_energy_with_constant']:.6f}"
        # )

        # print("Selected local variables:")
        # for var in sorted(selected_local):
        #     print(" ", var)

        # print("Selected global variables:")
        # for var in sorted(selected_global):
        #     print(" ", var)

        cluster_result = {
            "cluster_id": cluster_id,
            "cluster": cluster,
            "num_DT_cluster": num_DT_cluster,
            "variables": variables_c,
            "solution_local": solution_local,
            "selected_local": selected_local,
            "selected_global": selected_global,
            "best_x": orbit_output_c["best_x"],
            "best_energy_with_constant": orbit_output_c["best_energy_with_constant"],
            "Q_orbit": orbit_output_c["Q_orbit"],
            "result": orbit_output_c["result"],
        }

        all_cluster_results.append(cluster_result)

    # print("\n" + "=" * 70)
    # print("Finished clustered ORBIT routing")
    # print("=" * 70)

    # print("\nGlobal selected variables:")
    # for var in sorted(global_solution):
    #     print(" ", var)

    # ------------------------------------------------------------
    # Postprocess / repair global solution
    # ------------------------------------------------------------
    global_solution_raw = dict(global_solution)

    global_solution, repair_report = repair_global_solution_two_layer(
        global_solution=global_solution_raw,
        variables_global=variables_global,
        num_DT_global=num_DT_global,
        Layer1=Layer1,
        Layer2=Layer2,
        DistDP2SAT=DistDP2SAT,
        DistSAT2SAT=DistSAT2SAT,
        DistSAT2DT=DistSAT2DT,
        num_SAT=num_SAT,
        use_dt_distance=True,
    )

    # ------------------------------------------------------------
    # Reconstruct global min_state after repair
    # ------------------------------------------------------------
    min_state = np.zeros(len(variables_global), dtype=int)

    for i, var in enumerate(variables_global):
        min_state[i] = int(global_solution.get(var, 0))

    # Raw cluster-level ORBIT min_states
    cluster_min_states = [
        cluster_result["best_x"]
        for cluster_result in all_cluster_results
    ]

    return {
        "clusters": clusters,
        "cluster_results": all_cluster_results,
        "global_solution": global_solution,
        "min_state": min_state,
        "variables_global": variables_global,
        "cluster_min_states": cluster_min_states,
    }