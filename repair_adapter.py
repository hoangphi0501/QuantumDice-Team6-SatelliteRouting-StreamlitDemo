import numpy as np


REPAIR_SOURCE = "local_copy"

try:
    from LibOrbitSolver import repair_global_solution_two_layer as _repair_from_lib

    repair_global_solution_two_layer = _repair_from_lib
    REPAIR_SOURCE = "LibOrbitSolver.repair_global_solution_two_layer"
except Exception:

    def _sat_dt_value(A, sat, m, num_SAT=None, name="matrix"):
        A = np.asarray(A)
        if A.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {A.shape}")
        if num_SAT is not None:
            if A.shape[0] == num_SAT:
                return A[sat, m]
            if A.shape[1] == num_SAT:
                return A[m, sat]
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
        variable_set = set(variables_global)
        Layer1 = list(Layer1)
        Layer2 = list(Layer2)
        repaired_solution = {}
        repair_report = []

        for m in range(num_DT_global):
            xp_candidates = [
                s for s in Layer1 if ("xp", m, (m, s)) in variable_set
            ]
            if len(xp_candidates) == 0:
                raise ValueError(f"No xp candidates found for DP m={m}.")

            selected_s_list = [
                s
                for s in xp_candidates
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
                xp_action = "added_missing" if len(selected_s_list) == 0 else "reduced_multiple"

            repaired_solution[("xp", m, (m, chosen_s))] = 1

            xs_candidates = [
                sp for sp in Layer2 if ("xs", m, (chosen_s, sp)) in variable_set
            ]
            if len(xs_candidates) == 0:
                raise ValueError(f"No xs candidates found for m={m}, chosen_s={chosen_s}.")

            selected_sp_list = [
                sp
                for sp in xs_candidates
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
                xs_action = "added_missing" if len(selected_sp_list) == 0 else "reduced_multiple"

            repaired_solution[("xs", m, (chosen_s, chosen_sp))] = 1
            repair_report.append(
                {
                    "m": m,
                    "chosen_s_layer1": chosen_s,
                    "chosen_sp_layer2": chosen_sp,
                    "num_selected_s_before": len(selected_s_list),
                    "num_selected_sp_before": len(selected_sp_list),
                    "xp_action": xp_action,
                    "xs_action": xs_action,
                }
            )

        return repaired_solution, repair_report
