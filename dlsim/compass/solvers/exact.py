"""
exact: the true optimum by depth-first enumeration with branch-and-bound.

Sets are enumerated in lexicographic index order. A partial set C (|C| = s) is pruned when an upper bound on the
utility of every completion is below the incumbent. The bound uses the monotonicity of U in every component:

    Phi_comp  <= min(f_min(C), (n-s)-th largest f among the remaining) / f_ref
    Phi_comm  <= 1 - max(L_max(C), (n-s)-th smallest L among the remaining) / L_max
    Phi_fair  <= (fair(C) + top-(n-s) fair among the remaining) / n,   same for Phi_strag
    Phi_data  <= 1  (Phi_coh, Phi_repr <= 1)

Because enumeration is lexicographic and only strict improvements replace the incumbent, ties are broken by node id
exactly as in brute force. Instances with more than ``max_combinations`` subsets are refused (or, with
``on_too_large = warn``, solved by greedy_swap with a warning).
"""
import logging
import math
from typing import List, Optional

import numpy as np

from dlsim.compass.core.utility import (SelectionProblem, components_from_aggregates, evaluate,
                                        utility_from_components)
from dlsim.compass.solvers.base import (Budget, BudgetExhausted, Counter, Solver, SolverRefused, SolverResult,
                                        better, register)

logger = logging.getLogger("ExactSolver")


@register("exact")
class ExactSolver(Solver):

    def __init__(self, max_combinations: int = 2_000_000, on_too_large: str = "error", **options) -> None:
        super().__init__(**options)
        self.max_combinations = max_combinations
        self.on_too_large = on_too_large

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
        self.best_u = -math.inf
        self.best: Optional[List[int]] = None
        complete = True
        try:
            self._dfs(problem, [], 0, counter)
        except BudgetExhausted:
            complete = False
        if self.best is None:
            self.best = list(range(n))
        return self.finish(problem, self.best, counter, certificate="optimal" if complete else None,
                           bound=self.best_u if complete else None)

    def _bound(self, problem: SelectionProblem, chosen: List[int], start: int) -> float:
        n, s = problem.n, len(chosen)
        need = n - s
        rest = range(start, problem.size)
        f_rest = sorted((problem.f[i] for i in rest), reverse=True)
        L_rest = sorted(problem.L[i] for i in rest)
        fair_rest = sorted((problem.fair[i] for i in rest), reverse=True)
        q_rest = sorted((problem.Q[i] for i in rest), reverse=True)
        min_f = min([problem.f[i] for i in chosen] + ([f_rest[need - 1]] if need else []))
        max_L = max([problem.L[i] for i in chosen] + ([L_rest[need - 1]] if need else []))
        fair = sum(problem.fair[i] for i in chosen) + sum(fair_rest[:need])
        q = sum(problem.Q[i] for i in chosen) + sum(q_rest[:need])
        comps = components_from_aggregates(problem.params, n, min_f, max_L, 0.0, 0.0, fair, q, problem.net_sum)
        comps["data"] = 1.0
        return utility_from_components(problem.params, comps)

    def _dfs(self, problem: SelectionProblem, chosen: List[int], start: int, counter: Counter) -> None:
        n = problem.n
        if len(chosen) == n:
            counter.tick()
            value = evaluate(problem, chosen).utility
            if better(value, chosen, self.best_u, self.best):
                self.best_u, self.best = value, list(chosen)
            return
        need = n - len(chosen)
        for v in range(start, problem.size - need + 1):
            candidate = chosen + [v]
            if self.best is not None and len(candidate) < n:
                counter.tick()
                bound = self._bound(problem, candidate, v + 1)
                if bound < self.best_u - 1e-12 * max(1.0, abs(self.best_u)):
                    continue
            self._dfs(problem, candidate, v + 1, counter)
