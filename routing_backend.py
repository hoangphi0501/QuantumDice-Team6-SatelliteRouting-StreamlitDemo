import time

import numpy as np

from satellite_catalog import RE_KM, latlon_to_xyz_km


from LibBeamforming import Func_BeamDP2SAT, Func_BeamSAT2DT, Func_BeamSAT2SAT
from LibBenchmark_Checking import (
    build_qubo_matrix,
    compute_feasibility_probability,
    pbit_gibbs_gpu_minibatch,
    qubo_energy_dict,
)
from LibChannel import Func_ChlGS2SAT, Func_ChlSAT2SAT
from LibDataRate import Func_DataRateDP2SAT, Func_DataRateSAT2DT, Func_DataRateSAT2SAT
from LibNetwork import Func_2LayerLinks, Func_VariableIndex
from LibQUBO import QUBOModel
from repair_adapter import REPAIR_SOURCE, repair_global_solution_two_layer


def _locations_from_inputs(pairs_df, selected_sats):
    loc_dp = np.array(
        [
            latlon_to_xyz_km(row.dp_lat, row.dp_lon, RE_KM)
            for row in pairs_df.itertuples(index=False)
        ],
        dtype=float,
    )
    loc_dt = np.array(
        [
            latlon_to_xyz_km(row.dt_lat, row.dt_lon, RE_KM)
            for row in pairs_df.itertuples(index=False)
        ],
        dtype=float,
    )
    loc_sat = np.array(
        [
            latlon_to_xyz_km(sat["lat"], sat["lon"], RE_KM + float(sat["alt_km"]))
            for sat in selected_sats
        ],
        dtype=float,
    )
    return loc_sat, loc_dp, loc_dt


def _dict_to_vector(solution_dict, variables):
    return np.array([int(solution_dict.get(var, 0)) for var in variables], dtype=int)


def _solution_routes(solution_dict, variables, num_dt):
    routes = []
    for m in range(num_dt):
        xp_vars = [
            var
            for var in variables
            if var[0] == "xp" and var[1] == m and int(solution_dict.get(var, 0)) == 1
        ]
        xs_vars = [
            var
            for var in variables
            if var[0] == "xs" and var[1] == m and int(solution_dict.get(var, 0)) == 1
        ]
        if not xp_vars or not xs_vars:
            continue
        s1 = int(xp_vars[0][2][1])
        s2 = int(xs_vars[0][2][1])
        routes.append({"m": m, "layer1": s1, "layer2": s2})
    return routes


def build_and_solve_gibbs_cpu(
    pairs_df,
    selected_sats,
    seed=42,
    num_chains=80,
    num_steps=800,
    batch_size=8,
    t_start=5.0,
    t_end=0.01,
    wt=1.0,
    wc=1.0,
    lam1=20.0,
    lam2=20.0,
):
    if len(selected_sats) != 5:
        raise ValueError("Expected exactly 5 selected satellites.")

    np.random.seed(int(seed))
    t0 = time.time()

    num_dt = len(pairs_df)
    num_sat = 5
    layer1 = [0, 1]
    layer2 = [2, 3, 4]

    nt_s, nr_s = 128, 16
    nt_g, nr_g = 128, 16
    fw = 12e9
    bww = 240e6
    noise_variance_dbm = -174 + 10 * np.log10(bww)
    sq_sigma_w = 10 ** ((noise_variance_dbm - 30) / 10)
    p_dt_max = 10
    p_sat_max = 5
    im = 1e9
    lrelay = 1e-4
    alpha_ts = 0.1
    alpha_ss = 0.1

    loc_sat, loc_dp, loc_dt = _locations_from_inputs(pairs_df, selected_sats)

    chl_sat2sat, dist_sat2sat = Func_ChlSAT2SAT(nt_s, nr_s, loc_sat, fw)
    chl_dp2sat, chl_sat2dt, dist_sat2dt, dist_dp2sat = Func_ChlGS2SAT(
        nt_s,
        nr_s,
        nt_g,
        nr_g,
        loc_sat,
        loc_dp,
        loc_dt,
        fw,
    )

    beam_t_dp2sat, beam_r_dp2sat = Func_BeamDP2SAT(chl_dp2sat)
    beam_t_sat2dt, beam_r_sat2dt = Func_BeamSAT2DT(chl_sat2dt)
    beam_t_sat2sat, beam_r_sat2sat = Func_BeamSAT2SAT(chl_sat2sat)

    p_dp2sat = p_dt_max * np.ones((num_sat, num_dt)) / num_sat
    p_sat2dt = p_sat_max * np.ones((num_sat, num_dt)) / max(1, num_dt)
    p_sat2sat = p_sat_max * np.ones((num_sat, num_sat)) / num_sat

    dr_up = Func_DataRateDP2SAT(
        beam_t_dp2sat,
        beam_r_dp2sat,
        chl_dp2sat,
        p_dp2sat,
        bww,
        sq_sigma_w,
    )
    dr_down = Func_DataRateSAT2DT(
        beam_t_sat2dt,
        beam_r_sat2dt,
        chl_sat2dt,
        p_sat2dt,
        bww,
        sq_sigma_w,
    )
    dr_isl = Func_DataRateSAT2SAT(
        beam_t_sat2sat,
        beam_r_sat2sat,
        chl_sat2sat,
        p_sat2sat,
        bww,
        sq_sigma_w,
    )

    link_dp2sat, link_sat2sat, link_sat2dt = Func_2LayerLinks(num_dt, layer1, layer2)
    index_dp2sat, index_sat2sat = Func_VariableIndex(
        num_dt,
        link_dp2sat,
        link_sat2sat,
        link_sat2dt,
    )
    variables = index_dp2sat + index_sat2sat

    qubo = QUBOModel()
    qubo.func_tcomDpS(wt, index_dp2sat, dist_dp2sat, dr_up, alpha_ts, im)
    qubo.func_tcomSS(wt, index_sat2sat, dist_sat2sat, dr_isl, alpha_ss, im)
    qubo.func_tcomSDt(wt, layer2, index_sat2sat, dist_sat2dt, dr_down, im)
    qubo.func_trel(wt, num_sat, index_dp2sat, index_sat2sat, lrelay)
    qubo.func_linkCongestion(wc, layer1, layer2, index_sat2sat)
    qubo.cons_begin(num_dt, index_dp2sat, lam1)
    qubo.cons_equal(num_dt, layer1, index_dp2sat, index_sat2sat, lam2)

    c, q, q_pairs = qubo.get_qubo()
    c_mat, q_vec, q_mat, variables, _ = build_qubo_matrix(
        c=c,
        q=q,
        Q=q_pairs,
        variables=variables,
    )

    gibbs_result = pbit_gibbs_gpu_minibatch(
        c=c_mat,
        q_vec=q_vec,
        Q_mat=q_mat,
        num_chains=int(num_chains),
        num_steps=int(num_steps),
        batch_size=int(batch_size),
        T_start=float(t_start),
        T_end=float(t_end),
        seed=int(seed),
        device="cpu",
    )

    raw_x = np.asarray(gibbs_result["best_x"], dtype=int)
    raw_solution = {variables[i]: int(raw_x[i]) for i in range(len(variables))}
    raw_feasibility = compute_feasibility_probability(
        raw_x,
        variables,
        num_dt,
        layer1,
        index_dp2sat,
        index_sat2sat,
    )

    repaired_solution, repair_report = repair_global_solution_two_layer(
        global_solution=dict(raw_solution),
        variables_global=variables,
        num_DT_global=num_dt,
        Layer1=layer1,
        Layer2=layer2,
        DistDP2SAT=dist_dp2sat,
        DistSAT2SAT=dist_sat2sat,
        DistSAT2DT=dist_sat2dt,
        num_SAT=num_sat,
        use_dt_distance=True,
    )
    repaired_x = _dict_to_vector(repaired_solution, variables)
    repaired_feasibility = compute_feasibility_probability(
        repaired_x,
        variables,
        num_dt,
        layer1,
        index_dp2sat,
        index_sat2sat,
    )

    total_latency, latency_breakdown = qubo.compute_total_latency(
        repaired_x,
        variables,
        index_dp2sat,
        index_sat2sat,
        layer2,
        dist_dp2sat,
        dist_sat2sat,
        dist_sat2dt,
        dr_up,
        dr_isl,
        dr_down,
        alpha_ts,
        alpha_ss,
        im,
        lrelay,
    )
    congestion = qubo.compute_linkCongestion(
        repaired_x,
        variables,
        layer1,
        layer2,
        index_sat2sat,
    )

    return {
        "num_dt": num_dt,
        "num_sat": num_sat,
        "n_vars": len(variables),
        "variables": variables,
        "raw_x": raw_x,
        "repaired_x": repaired_x,
        "raw_solution": raw_solution,
        "repaired_solution": repaired_solution,
        "routes": _solution_routes(repaired_solution, variables, num_dt),
        "raw_best_obj": float(gibbs_result["best_obj"]),
        "repaired_obj": float(qubo_energy_dict(c, q, q_pairs, repaired_solution)),
        "raw_feasibility": raw_feasibility,
        "repaired_feasibility": repaired_feasibility,
        "repair_report": repair_report,
        "repair_source": REPAIR_SOURCE,
        "total_latency": float(total_latency),
        "latency_breakdown": latency_breakdown,
        "congestion": float(congestion),
        "runtime_sec": time.time() - t0,
        "device": gibbs_result["device"],
        "distance_matrices": {
            "DistDP2SAT": dist_dp2sat,
            "DistSAT2SAT": dist_sat2sat,
            "DistSAT2DT": dist_sat2dt,
        },
    }
