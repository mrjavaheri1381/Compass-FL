"""
exact: the true optimum by depth-first branch-and-bound.

Sets are enumerated in lexicographic index order with an ``IncrementalState`` (add on the way down, remove on the way
back), so every leaf costs O(1). A partial set C (|C| = s, need = n - s more members from the remaining indices R) is
pruned when an upper bound on the utility of every completion is below the incumbent. U is monotone in every
component, so bounding every component separately gives a valid bound:

    Phi_comp  <= min(f_min(C), need-th largest f in R) / f_ref
    Phi_comm  <= 1 - max(L_max(C), need-th smallest L in R) / L_max
    Phi_fair  <= (fair(C) + top-need fair in R) / n,   same for Phi_strag
    P_final   <= P + top-need sigma_v(C) in R + C(need, 2) max_{a<b in R} S_ab      -> Phi_coh upper bound
    P_final   >= P + bottom-need sigma_v(C) in R + C(need, 2) min_{a<b in R} S_ab   -> Phi_repr upper bound
    R_final   <= R + top-need r in R                                                   (Phi_repr = R / sqrt((n + 2P) net))

An optional warm start (e.g. greedy_swap) gives a strong incumbent from the beginning. Ties are broken by the
lexicographically smallest id tuple, as in brute force. If the evaluation or time budget runs out, the result is not
certified and ``bound`` is a valid upper bound on the optimum (the maximum of the incumbent and of the bounds of the
unexplored subtrees), so the optimality gap stays certified.
"""
import logging
import math
from typing import List, Optional

import numpy as np

from dlsim.compass.core.utility import (IncrementalState, SelectionProblem, components_from_aggregates,
                                        utility_from_components)
from dlsim.compass.solvers.base import (Budget, BudgetExhausted, Counter, Solver, SolverRefused, SolverResult,
                                        better, register)

logger = logging.getLogger("ExactSolver")


@register("exact")
class ExactSolver(Solver):

    def __init__(self, max_combinations: int = 2_000_000, on_too_large: str = "error", warm_start: Optional[str] = None,
                 **options) -> None:
        super().__init__(**options)
        self.max_combinations = max_combinations
        self.on_too_large = on_too_large
        self.warm_start = warm_start

    def solve(self, problem: SelectionProblem, budget: Budget, rng: np.random.Generator) -> SolverResult:
        k, n = problem.size, problem.n
        if k < n:
            raise ValueError("Neighborhood smaller than n")
        combinations = math.comb(k, n)
        if combinations > self.max_combinations:
            message = "exact: C(%d, %d) = %d subsets exceeds max_combinations = %d" % (
                k, n, combinations, self.max_combinations)
            if self.on_too_large == "warn":
                logger.warning(message + " -- falling back to greedy_swap")
                from dlsim.compass.solvers.greedy_swap import GreedySwapSolver
                return GreedySwapSolver(**self.options).solve(problem, budget, rng)
            raise SolverRefused(message)

        counter = Counter(budget)
        self.problem = problem
        self.counter = counter
        self.best_u = -math.inf
        self.best: Optional[List[int]] = None
        self.frontier_ub = -math.inf
        self._precompute()
        if self.warm_start:
            from dlsim.compass.solvers import create_solver
            start = create_solver(self.warm_start).__class__(**self.options).solve(problem, Budget(), rng)
            self.best_u, self.best = start.utility, sorted(start.members)
            counter.evaluations += start.evaluations
        state = IncrementalState(problem)
        complete = True
        try:
            self._dfs(state, 0)
        except BudgetExhausted:
            complete = False
        if self.best is None:
            self.best = list(range(n))
        bound = self.best_u if complete else max(self.best_u, self.frontier_ub)
        return self.finish(problem, self.best, counter, certificate="optimal" if complete else None, bound=bound)

    # ------------------------------------------------------------------ bounds

    def _precompute(self) -> None:
        p = self.problem
        k = p.size
        S = p.S.copy()
        np.fill_diagonal(S, np.nan)
        # Max / min off-diagonal similarity among the indices >= i
        self.smax = np.zeros(k + 1)
        self.smin = np.zeros(k + 1)
        for i in range(k - 2, -1, -1):
            block = S[i:, i:]
            self.smax[i] = np.nanmax(block)
            self.smin[i] = np.nanmin(block)

    @staticmethod
    def _top(values: np.ndarray, need: int) -> float:
        return float(np.partition(values, len(values) - need)[len(values) - need:].sum())

    @staticmethod
    def _bottom(values: np.ndarray, need: int) -> float:
        return float(np.partition(values, need - 1)[:need].sum())

    def _bound(self, state: IncrementalState, start: int) -> float:
        p = self.problem
        n, s = p.n, state.size
        need = n - s
        if need == 0:
            return state.utility()
        m = p.size - start
        f_rest, L_rest = p.f[start:], p.L[start:]
        kth_f = float(np.partition(f_rest, m - need)[m - need])
        kth_L = float(np.partition(L_rest, need - 1)[need - 1])
        min_f = min(state.min_f[0], kth_f) if s else kth_f
        max_L = max(state.max_L[0], kth_L) if s else kth_L
        fair = state.fair_sum + self._top(p.fair[start:], need)
        q = state.q_sum + self._top(p.Q[start:], need)
        sigma = state.sigma[start:]
        pairs = need * (need - 1) / 2.0
        P_ub = state.P + self._top(sigma, need) + pairs * (self.smax[start] if pairs else 0.0)
        P_lb = state.P + self._bottom(sigma, need) + pairs * (self.smin[start] if pairs else 0.0)
        R_ub = state.R + self._top(p.r[start:], need)
        params = p.params
        comps = components_from_aggregates(params, n, min_f, max_L, P_ub, R_ub, fair, q, p.net_sum)
        # Cohesion uses the largest possible P, representativeness the smallest
        rep = components_from_aggregates(params, n, min_f, max_L, max(P_lb, 0.0), R_ub, fair, q, p.net_sum)["repr"]
        data = 1.0
        if params.w_c > 0:
            data *= comps["coh"] ** params.w_c
        if params.w_r > 0:
            data *= rep ** params.w_r
        comps["data"] = min(1.0, max(params.eps, data))
        return utility_from_components(params, comps)

    def _prunable(self, bound: float) -> bool:
        return self.best is not None and bound < self.best_u - 1e-12 * max(1.0, abs(self.best_u))

    # ------------------------------------------------------------------ search

    def _dfs(self, state: IncrementalState, start: int) -> None:
        p = self.problem
        need = p.n - state.size
        last = p.size - need + 1
        v = start
        while v < last:
            try:
                self.counter.tick()
                state.add(v)
                try:
                    if state.size == p.n:
                        value = state.utility()
                        if better(value, state.members, self.best_u, self.best):
                            self.best_u, self.best = value, sorted(state.members)
                    elif not self._prunable(self._bound(state, v + 1)):
                        self._dfs(state, v + 1)
                finally:
                    state.remove(v)
            except BudgetExhausted:
                # Bound every subtree of this level that is not fully explored (v included, conservatively)
                for u in range(v, last):
                    state.add(u)
                    self.frontier_ub = max(self.frontier_ub, self._bound(state, u + 1))
                    state.remove(u)
                raise
            v += 1
