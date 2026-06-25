from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from itertools import combinations
from typing import Dict, Tuple, Hashable, Any, Literal, List

# ----------------------------
# Types
# ----------------------------
Package = Hashable
Node = Hashable
Link = Tuple[Node, Node]
KIndex = Hashable

VarType = Literal["xp", "xs"]
VarKey = Tuple[VarType, Hashable, Hashable]
# ("x", m, (i,j)) -> x_{m,(i,j)}

class QUBOModel:
    """
    QUBO:
      Obj(x) = constant + Σ q[v] x + Σ_{u<v} Q[u,v] x x
    where x ∈ {0,1} for ALL variable types ("x").
    """

    def __init__(self):
        self.constant: float = 0.0
        self.linear: Dict[VarKey, float] = defaultdict(float)
        self.quadratic: Dict[Tuple[VarKey, VarKey], float] = defaultdict(float)

    # ---------------------------
    # Bookkeeping: canonical pair
    # ---------------------------
    @staticmethod
    def _ordered_pair(a: VarKey, b: VarKey) -> Tuple[VarKey, VarKey]:
        # ensures (a,b) and (b,a) map to the same dictionary key
        return (a, b) if a < b else (b, a)

    # ---------------------------
    # Add terms
    # ---------------------------
    def add_constant(self, c: float) -> None:
        self.constant += float(c)

    def add_linear(self, v: VarKey, coeff: float) -> None:
        self.linear[v] += float(coeff)

    def add_quadratic(self, v1: VarKey, v2: VarKey, coeff: float) -> None:
        """
        Add coeff * x_{v1} * x_{v2}.
        If v1==v2, it becomes a linear term since x^2=x for binary.
        """
        coeff = float(coeff)
        if v1 == v2:
            self.linear[v1] += coeff
            return
        u, v = self._ordered_pair(v1, v2)
        self.quadratic[(u, v)] += coeff

    # --------------------------------------
    # Add squared-linear: λ (c + Σ a_v x_v)^2
    # --------------------------------------
    def add_squared_linear(self, a: Dict[VarKey, float], c: float = 0.0, lam: float = 1.0) -> None:
        """
        Expand λ (c + Σ a_v x_v)^2 into constant + linear + quadratic.
        Works with mixed variables: ("x",m,e), ("v",m,k), ("u",j,k), etc.
        """
        lam = float(lam)
        c = float(c)

        # constant
        self.constant += lam * (c ** 2)

        # linear: λ(2 c a + a^2) x
        for var, av in a.items():
            av = float(av)
            self.linear[var] += lam * (2.0 * c * av + av * av)

        # quadratic: λ(2 a_i a_j) x_i x_j
        keys = list(a.keys())
        for i, j in combinations(keys, 2):
            val = lam * (2.0 * float(a[i]) * float(a[j]))
            u, v = self._ordered_pair(i, j)
            self.quadratic[(u, v)] += val

    # Latency com m,(m,s)
    def func_tcomDpS(self,Wt,Index_DP2SAT,DistDP2SAT,DR_up,alpha_TS,Im):
        # tcomDpS_mms
        for var in Index_DP2SAT:
            _, m, link = var
            s = link[1]
            self.linear[var] += Wt*(Im/DR_up[s,m] + DistDP2SAT[s,m]/3e8)
            for var2 in Index_DP2SAT:
                _, m2, link2 = var2
                if m2 != m and link2[1] == s:
                    u, v = self._ordered_pair(var, var2)
                    self.quadratic[(u, v)] += Wt*(Im/DR_up[s,m]*alpha_TS)
    
    # Latency com m,(s,sp)
    def func_tcomSS(self,Wt,Index_SAT2SAT,DistSAT2SAT,DR_isl,alpha_SS,Im):
        # tcomSS_mssp
        for var in Index_SAT2SAT:
            _, m, link = var
            s, sp = link
            self.linear[var] += Wt*(Im/DR_isl[s,sp] + DistSAT2SAT[s,sp]/3e8)
            for var2 in Index_SAT2SAT:
                _, m2, link2 = var2
                if m2 != m and link2[0] != s and link2[1] == sp:
                    u, v = self._ordered_pair(var, var2)
                    self.quadratic[(u, v)] += Wt*(Im/DR_isl[s,sp]*alpha_SS)
    
    # Latency com m,(s,m)
    def func_tcomSDt(self,Wt,Layer2,Index_SAT2SAT,DistSAT2DT,DR_down,Im):
        # tcomSS_msm
        for var in Index_SAT2SAT:
            _, m, link = var
            s, sp = link
            if sp in Layer2:
                self.linear[var] += Wt*(Im/DR_down[sp,m] + DistSAT2DT[sp,m]/3e8)
    
    # Latency processing
    def func_trel(self,Wt,num_SAT,Index_DP2SAT,Index_SAT2SAT,Lrelay):
        for s in range(num_SAT):
            for var in Index_DP2SAT:
                self.linear[var] += Wt*Lrelay
            for var in Index_SAT2SAT:
                self.linear[var] += 2*Wt*Lrelay
    
    def func_linkCongestion(self,Wc,Layer1,Layer2,Index_SAT2SAT):
        for s in Layer1:
            for sp in Layer2:
                a1 = {
                    item: 1
                    for item in Index_SAT2SAT
                    if item[2][0] == s and item[2][1] == sp
                }
                self.add_squared_linear(
                    a=a1,
                    c=0,
                    lam=Wc,
                )
            
    # Constraint begin
    def cons_begin(self,num_DT,Index_DP2SAT,lam):
        for m in range(num_DT):
            a = {
                item: 1
                for item in Index_DP2SAT
                if item[1] == m
            }
            self.add_squared_linear(
                a=a, # merged 2 dictionaries
                c=-1,
                lam=lam,
            )
            
    # Constraint equal
    def cons_equal(self,num_DT,Layer1,Index_DP2SAT,Index_SAT2SAT,lam):
        for m in range(num_DT):
            for s in Layer1:
                a1 = {
                    item: 1
                    for item in Index_DP2SAT
                    if item[2][1] == s and item[1] == m
                }
                a2 = {
                    item: -1
                    for item in Index_SAT2SAT
                    if item[2][0] == s  and item[1] == m
                }
                self.add_squared_linear(
                    a=a1|a2, # merged 2 dictionaries
                    c=0,
                    lam=lam, # depend on the latency
                )

    # Export QUBO
    def get_qubo(self) -> Tuple[float, Dict[VarKey, float], Dict[Tuple[VarKey, VarKey], float]]:
        return self.constant, dict(self.linear), dict(self.quadratic)

    # -------------------------------------------------------
    # Convert bitstring or dict solution to solution dict
    # -------------------------------------------------------
    @staticmethod
    def _to_solution_dict(solution, variables=None) -> Dict[VarKey, int]:
        # Case 1: already decoded dictionary
        if isinstance(solution, dict):
            return {
                var: int(value)
                for var, value in solution.items()
            }

        # Case 2: bitstring / list / numpy array / torch tensor
        if variables is None:
            raise ValueError(
                "variables must be provided when solution is a bitstring."
            )
        if len(solution) != len(variables):
            raise ValueError(
                f"Bitstring length {len(solution)} does not match "
                f"number of variables {len(variables)}."
            )
        return {
            variables[i]: int(solution[i])
            for i in range(len(variables))
        }
        
    # -------------------------------------------------------
    # Compute raw total latency after solving QUBO
    # -------------------------------------------------------
    def compute_total_latency(
        self,
        solution,
        variables=None,
        Index_DP2SAT=None,
        Index_SAT2SAT=None,
        Layer2=None,
        DistDP2SAT=None,
        DistSAT2SAT=None,
        DistSAT2DT=None,
        DR_up=None,
        DR_isl=None,
        DR_down=None,
        alpha_TS: float = 0.0,
        alpha_SS: float = 0.0,
        Im: float = 1.0,
        Lrelay: float = 0.0,
        c_light: float = 3e8,
        return_breakdown: bool = True,
    ):
        solution_dict = self._to_solution_dict(solution, variables)

        latency_dp2sat = 0.0
        latency_sat2sat = 0.0
        latency_sat2dt = 0.0
        latency_relay = 0.0

        selected_dp2sat = [
            var for var in Index_DP2SAT
            if int(solution_dict.get(var, 0)) == 1
        ]

        selected_sat2sat = [
            var for var in Index_SAT2SAT
            if int(solution_dict.get(var, 0)) == 1
        ]

        # 1. DP -> SAT latency
        for var in selected_dp2sat:
            _, m, link = var
            s = link[1]

            tx_time = Im / DR_up[s, m]
            prop_time = DistDP2SAT[s, m] / c_light

            num_other_same_sat = sum(
                1
                for var2 in selected_dp2sat
                if var2 != var
                and var2[1] != m
                and var2[2][1] == s
            )

            congestion_time = alpha_TS * tx_time * num_other_same_sat

            latency_dp2sat += tx_time + prop_time + congestion_time

        # 2. SAT -> SAT latency
        for var in selected_sat2sat:
            _, m, link = var
            s, sp = link

            tx_time = Im / DR_isl[s, sp]
            prop_time = DistSAT2SAT[s, sp] / c_light

            num_other_same_receiver = sum(
                1
                for var2 in selected_sat2sat
                if var2 != var
                and var2[1] != m
                and var2[2][0] != s
                and var2[2][1] == sp
            )

            congestion_time = alpha_SS * tx_time * num_other_same_receiver

            latency_sat2sat += tx_time + prop_time + congestion_time

        # 3. SAT -> DT downlink latency
        for var in selected_sat2sat:
            _, m, link = var
            s, sp = link

            if sp in Layer2:
                tx_time = Im / DR_down[sp, m]
                prop_time = DistSAT2DT[sp, m] / c_light

                latency_sat2dt += tx_time + prop_time

        # 4. Relay processing latency
        for var in selected_dp2sat:
            latency_relay += Lrelay

        for var in selected_sat2sat:
            latency_relay += 2.0 * Lrelay

        total_latency = (
            latency_dp2sat
            + latency_sat2sat
            + latency_sat2dt
            + latency_relay
        )

        breakdown = {
            "dp2sat_latency": latency_dp2sat,
            "sat2sat_latency": latency_sat2sat,
            "sat2dt_latency": latency_sat2dt,
            "relay_latency": latency_relay,
            "total_latency": total_latency,
        }
        if return_breakdown:
            return total_latency, breakdown
        return total_latency
    
    
    def compute_linkCongestion(self,solution,variables,Layer1,Layer2,Index_SAT2SAT):
        solution_dict = self._to_solution_dict(solution, variables)
        congestion = 0
        for s in Layer1:
            for sp in Layer2:
                congestion_ssp = 0
                for item in Index_SAT2SAT:
                    if item[2][0] == s and item[2][1] == sp:
                        congestion_ssp += int(solution_dict.get(item, 0))
                congestion += congestion_ssp**2
        return congestion

