import numpy as np
import torch

# Convert your dictionary QUBO into matrix form
def build_qubo_matrix(c, q, Q, variables=None):
    """
    Convert QUBO dictionaries into matrix form.
    """
    if variables is None:
        var_set = set()

        for var in q:
            var_set.add(var)

        for var_i, var_j in Q:
            var_set.add(var_i)
            var_set.add(var_j)

        variables = sorted(var_set, key=str)

    var_to_idx = {var: i for i, var in enumerate(variables)}
    n = len(variables)

    q_vec = np.zeros(n)
    Q_mat = np.zeros((n, n))

    # Linear terms
    for var, coeff in q.items():
        i = var_to_idx[var]
        q_vec[i] += coeff

    # Quadratic terms
    for (var_i, var_j), coeff in Q.items():
        i = var_to_idx[var_i]
        j = var_to_idx[var_j]

        # Store symmetrically.
        # obj will be computed using i < j, so this is only for easy lookup.
        Q_mat[i, j] += coeff
        Q_mat[j, i] += coeff

    return c, q_vec, Q_mat, variables, var_to_idx

def qubo_obj_torch(X, c, q, Q):
    """
    Compute QUBO obj for many chains in parallel.

    X shape:
        [num_chains, n]

    q shape:
        [n]

    Q shape:
        [n, n]

    Objective:
        E = c + q^T x + 0.5 x^T Q x
    """

    linear = X @ q
    quadratic = 0.5 * torch.sum((X @ Q) * X, dim=1)

    return c + linear + quadratic

#######################################################################################
# Proposed Probabilistic Optimization Method: GPU mini-batch p-bit Gibbs solver #
#######################################################################################

def pbit_gibbs_gpu_minibatch(
    c,
    q_vec,
    Q_mat,
    num_chains=50,
    num_steps=500,
    batch_size=16,
    T_start=5.0,
    T_end=0.05,
    seed=42,
    device=None,
):
    """
    GPU mini-batch p-bit Gibbs solver.

    Main idea:
        - Run many chains in parallel on GPU.
        - At each step, update a mini-batch of variables.
        - Each variable update uses:

              P(x_i = 1) = sigmoid(-ΔE_i / T)

        - ΔE_i = E(x_i=1) - E(x_i=0)

    This is useful for large QUBOs.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("GPU:", torch.cuda.get_device_name(0))

    torch.manual_seed(seed)

    q = torch.tensor(q_vec, dtype=torch.float32, device=device)
    Q = torch.tensor(Q_mat, dtype=torch.float32, device=device)
    c = torch.tensor(float(c), dtype=torch.float32, device=device)

    n = q.shape[0]

    # X contains many independent chains
    X = torch.randint(
        low=0,
        high=2,
        size=(num_chains, n),
        dtype=torch.float32,
        device=device,
    )

    energies = qubo_obj_torch(X, c, q, Q)

    best_idx = torch.argmin(energies)
    best_E = energies[best_idx].item()
    best_x = X[best_idx].clone()

    for step in range(num_steps):

        # Annealing temperature
        frac = step / max(1, num_steps - 1)
        T = T_start * (T_end / T_start) ** frac

        # Choose mini-batch variables
        batch = torch.randperm(n, device=device)[:min(batch_size, n)]

        # Sequentially update variables inside the mini-batch.
        # This is more stable than updating all selected bits simultaneously.
        for i in batch:

            i_int = int(i.item())

            # ΔE_i = q_i + sum_{j != i} Q_ij x_j
            dE = q[i_int] + X @ Q[:, i_int]

            # Remove possible self contribution
            dE = dE - Q[i_int, i_int] * X[:, i_int]

            # p-bit update probability
            p_one = torch.sigmoid(-dE / T)

            # Sample new value for x_i in all chains
            new_values = torch.bernoulli(p_one)

            X[:, i_int] = new_values

        # Evaluate obj after mini-batch update
        energies = qubo_obj_torch(X, c, q, Q)

        # Track best solution
        current_best_idx = torch.argmin(energies)
        current_best_E = energies[current_best_idx].item()

        if current_best_E < best_E:
            best_E = current_best_E
            best_x = X[current_best_idx].clone()

    return {
        "best_x": best_x.detach().cpu().numpy().astype(int),
        "best_obj": best_E,
        "final_solutions": X.detach().cpu().numpy().astype(int),
        "final_energies": energies.detach().cpu().numpy(),
        "device": device,
    }
    
def decode_solution(x, variables, threshold=0.5):
    """
    Convert binary vector back to dictionary:

        variable -> 0/1

    Example:
        ('xp', 0, (0, 0)) -> 1
        ('xs', 0, (0, 2)) -> 0
    """

    return {
        variables[i]: int(x[i] > threshold)
        for i in range(len(variables))
    }
    
def print_selected_routes(solution_dict):
    """
    Print only selected variables with value 1.
    """

    print("\n=== Selected Variables x = 1 ===")

    for var, value in solution_dict.items():
        if value == 1:
            print(var)

###################################################
# Benchmark 1: Brute force checker for small QUBO #
###################################################
def brute_force_qubo(c, q_vec, Q_mat):
    """
    Exact brute force solver.
    """
    n = len(q_vec)
    if n > 25:
        raise ValueError("Brute force is too expensive for n > 25.")

    best_E = np.inf
    best_x = None
    for state in range(2 ** n):
        x = np.array([(state >> i) & 1 for i in range(n)], dtype=int)
        x = np.asarray(x)
        linear = np.dot(q_vec, x)
        quadratic = 0.5 * x @ Q_mat @ x
        E = c + linear + quadratic

        if E < best_E:
            best_E = E
            best_x = x.copy()

    return best_x, best_E

###################################################
# Benchmark 2: Genetic Algorithm (GA) #
###################################################
def qubo_obj_numpy(x, c, q_vec, Q_mat):
    """
    Single-solution QUBO obj.

    E = c + q^T x + 0.5 x^T Q x
    """
    x = np.asarray(x)
    return c + np.dot(q_vec, x) + 0.5 * x @ Q_mat @ x

def qubo_obj_population(X, c, q_vec, Q_mat):
    """
    Population QUBO obj.
    X shape: [population_size, n]
    """
    linear = X @ q_vec
    quadratic = 0.5 * np.sum((X @ Q_mat) * X, axis=1)
    return c + linear + quadratic

def ga_qubo_solver(
    c,
    q_vec,
    Q_mat,
    pop_size=200,
    num_generations=500,
    crossover_rate=0.9,
    mutation_rate=None,
    elite_frac=0.05,
    tournament_k=3,
    seed=42,
    initial_population=None,
):
    """
    CPU Genetic Algorithm for QUBO minimization.

    obj:
        E = c + q^T x + 0.5 x^T Q x
    """

    rng = np.random.default_rng(seed)
    n = len(q_vec)

    if mutation_rate is None:
        mutation_rate = 1.0 / n

    elite_count = max(1, int(elite_frac * pop_size))

    # ------------------------------------------------------------
    # Initialize population
    # ------------------------------------------------------------
    if initial_population is not None:
        initial_population = np.asarray(initial_population).astype(int)

        if initial_population.ndim == 1:
            initial_population = initial_population.reshape(1, -1)

        if initial_population.shape[1] != n:
            raise ValueError("initial_population has wrong number of variables.")

        if initial_population.shape[0] >= pop_size:
            population = initial_population[:pop_size].copy()
        else:
            random_part = rng.integers(
                0,
                2,
                size=(pop_size - initial_population.shape[0], n),
                dtype=int,
            )
            population = np.vstack([initial_population, random_part])
    else:
        population = rng.integers(0, 2, size=(pop_size, n), dtype=int)

    energies = qubo_obj_population(population, c, q_vec, Q_mat)

    best_idx = np.argmin(energies)
    best_x = population[best_idx].copy()
    best_obj = energies[best_idx]

    history = []

    # ------------------------------------------------------------
    # Tournament selection
    # ------------------------------------------------------------
    def tournament_select():
        candidate_idx = rng.integers(0, pop_size, size=tournament_k)
        best_candidate = candidate_idx[np.argmin(energies[candidate_idx])]
        return population[best_candidate].copy()

    # ------------------------------------------------------------
    # Main GA loop
    # ------------------------------------------------------------
    for gen in range(num_generations):

        # Sort by obj
        sorted_idx = np.argsort(energies)
        elites = population[sorted_idx[:elite_count]].copy()

        new_population = [elites[i] for i in range(elite_count)]

        while len(new_population) < pop_size:
            parent_1 = tournament_select()
            parent_2 = tournament_select()

            # Uniform crossover
            if rng.random() < crossover_rate:
                mask = rng.random(n) < 0.5
                child = np.where(mask, parent_1, parent_2)
            else:
                child = parent_1.copy()

            # Mutation
            mutation_mask = rng.random(n) < mutation_rate
            child[mutation_mask] = 1 - child[mutation_mask]

            new_population.append(child)

        population = np.array(new_population, dtype=int)

        energies = qubo_obj_population(population, c, q_vec, Q_mat)

        current_best_idx = np.argmin(energies)
        current_best_obj = energies[current_best_idx]

        if current_best_obj < best_obj:
            best_obj = current_best_obj
            best_x = population[current_best_idx].copy()

        history.append(best_obj)

    return {
        "best_x": best_x,
        "best_obj": float(best_obj),
        "final_population": population,
        "final_energies": energies,
        "history": history,
    }
    
def qubo_energy_dict(c, q, Q, x_dict):
    energy = float(c)
    for var, coef in q.items():
        energy += coef * x_dict.get(var, 0)
    for (var_i, var_j), coef in Q.items():
        energy += coef * x_dict.get(var_i, 0) * x_dict.get(var_j, 0)
    return energy

def find_unique_variable(index_set, condition, name):
    matched = [item for item in index_set if condition(item)]
    if len(matched) != 1:
        raise ValueError(f"{name}: expected 1 matched variable, got {len(matched)}. Matched = {matched}")
    return matched[0]

###########################################################
# Benchmark 3: From each DP, choose the nearest node #
###########################################################
def nearest_layer_benchmark(num_DT, Layer1, Layer2, Index_DP2SAT, Index_SAT2SAT, DistDP2SAT, DistSAT2SAT, variables):
    x_dict = {var: 0 for var in variables}
    routes = []

    for m in range(num_DT):
        s1 = min(Layer1, key=lambda s: DistDP2SAT[s, m])

        xp_var = find_unique_variable(
            Index_DP2SAT,
            lambda item: item[1] == m and item[2][1] == s1,
            name=f"DP {m} to Layer1 SAT {s1}"
        )

        s2 = min(Layer2, key=lambda s: DistSAT2SAT[s1, s])

        xs_var = find_unique_variable(
            Index_SAT2SAT,
            lambda item: item[1] == m and item[2][0] == s1 and item[2][1] == s2,
            name=f"Layer1 SAT {s1} to Layer2 SAT {s2} for demand {m}"
        )

        x_dict[xp_var] = 1
        x_dict[xs_var] = 1

        routes.append({
            "demand": m,
            "Layer1_sat": s1,
            "Layer2_sat": s2,
            "xp_var": xp_var,
            "xs_var": xs_var,
            "DP_to_Layer1_distance": DistDP2SAT[s1, m],
            "Layer1_to_Layer2_distance": DistSAT2SAT[s1, s2],
        })

    x_vec = np.array([x_dict[var] for var in variables], dtype=int)
    return x_vec, x_dict, routes

def safe_link_time(data_size, data_rate):
    if data_rate <= 0:
        return np.inf
    return data_size / data_rate

#################################################################################
# Benchmark 4: From each DP, choose the nodes with the minimum latency  #
#################################################################################
def minimum_latency_route_benchmark(num_DT, Layer1, Layer2, Index_DP2SAT, Index_SAT2SAT, DistDP2SAT, DistSAT2SAT, DistSAT2DT, DR_up, DR_isl, DR_down, Im, Lrelay, variables, include_propagation=True, c_light=3e8):
    x_dict = {var: 0 for var in variables}
    routes = []

    for m in range(num_DT):
        best_latency = np.inf
        best_s1 = None
        best_s2 = None
        best_parts = None

        for s1 in Layer1:
            for s2 in Layer2:
                t_up_tx = safe_link_time(Im, DR_up[s1, m])
                t_isl_tx = safe_link_time(Im, DR_isl[s1, s2])
                t_down_tx = safe_link_time(Im, DR_down[s2, m])

                if include_propagation:
                    t_up_prop = DistDP2SAT[s1, m] / c_light
                    t_isl_prop = DistSAT2SAT[s1, s2] / c_light
                    t_down_prop = DistSAT2DT[s2, m] / c_light
                else:
                    t_up_prop = 0.0
                    t_isl_prop = 0.0
                    t_down_prop = 0.0

                t_up = t_up_tx + t_up_prop
                t_isl = t_isl_tx + t_isl_prop
                t_down = t_down_tx + t_down_prop

                total_latency = t_up + t_isl + t_down + Lrelay

                if total_latency < best_latency:
                    best_latency = total_latency
                    best_s1 = s1
                    best_s2 = s2
                    best_parts = {
                        "t_up_tx": t_up_tx,
                        "t_isl_tx": t_isl_tx,
                        "t_down_tx": t_down_tx,
                        "t_up_prop": t_up_prop,
                        "t_isl_prop": t_isl_prop,
                        "t_down_prop": t_down_prop,
                        "t_up": t_up,
                        "t_isl": t_isl,
                        "t_down": t_down,
                        "t_relay": Lrelay,
                        "total_latency": total_latency,
                    }

        xp_var = find_unique_variable(
            Index_DP2SAT,
            lambda item: item[1] == m and item[2][1] == best_s1,
            name=f"DP {m} to Layer1 SAT {best_s1}"
        )

        xs_var = find_unique_variable(
            Index_SAT2SAT,
            lambda item: item[1] == m and item[2][0] == best_s1 and item[2][1] == best_s2,
            name=f"Layer1 SAT {best_s1} to Layer2 SAT {best_s2} for demand {m}"
        )

        x_dict[xp_var] = 1
        x_dict[xs_var] = 1

        routes.append({
            "demand": m,
            "Layer1_sat": best_s1,
            "Layer2_sat": best_s2,
            "xp_var": xp_var,
            "xs_var": xs_var,
            **best_parts,
        })

    x_vec = np.array([x_dict[var] for var in variables], dtype=int)

    return x_vec, x_dict, routes

def compute_feasibility_probability(samples, variables, num_DT, Layer1, Index_DP2SAT, Index_SAT2SAT):
    samples = np.asarray(samples)
    
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.shape[1] != len(variables):
        raise ValueError(f"samples has {samples.shape[1]} pbits, but variables has {len(variables)} variables")

    feasible_list = []
    begin_ok_list = []
    equal_ok_list = []

    for sample in samples:
        x = {var: int(value) for var, value in zip(variables, sample)}

        begin_ok = all(
            sum(x[item] for item in Index_DP2SAT if item[1] == m) == 1
            for m in range(num_DT)
        )

        equal_ok = all(
            sum(x[item] for item in Index_DP2SAT if item[1] == m and item[2][1] == s)
            ==
            sum(x[item] for item in Index_SAT2SAT if item[1] == m and item[2][0] == s)
            for m in range(num_DT)
            for s in Layer1
        )

        feasible = begin_ok and equal_ok

        feasible_list.append(feasible)
        begin_ok_list.append(begin_ok)
        equal_ok_list.append(equal_ok)

    report = {
        "num_samples": len(samples),
        "num_feasible": int(np.sum(feasible_list)),
        "p_feasible": float(np.mean(feasible_list)),
        "p_begin_feasible": float(np.mean(begin_ok_list)),
        "p_equal_feasible": float(np.mean(equal_ok_list)),
    }
    return report