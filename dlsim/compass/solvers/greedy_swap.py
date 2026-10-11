"""
greedy_swap: the paper's local solver (Algorithm 1).

Seed with each of the R_s pairs of highest U({i, j}), grow greedily to n, improve with best-improvement 1-swaps for at
most I_max passes, and keep the best result. Every candidate evaluation is O(1) through ``IncrementalState``.
Deterministic: ties are broken by node id (= index order of the problem).
"""
from typing import List, Optional

import numpy as np

from dlsim.compass.core.utility import IncrementalState, SelectionProblem
from dlsim.compass.solvers.base import (Budget, BudgetExhausted, Counter, Solver, SolverResult, best_swap, better,
                                        greedy_complete, register)


def local_search(problem: SelectionProblem, state: IncrementalState, counter: Counter, max_passes: int) -> int:
    passes = 0
    current = state.utility()
    while passes < max_passes:
        u, v, value = best_swap(problem, state, counter)
        if u is None or value <= current:
            break
        state.swap(u, v)
        current = value
        passes += 1
    return passes


@register("greedy_swap")
class GreedySwapSolver(Solver):

    def __init__(self, R_s: int = 3, I_max: int = 50, **options) -> None:
        super().__init__(**options)
        self.R_s = R_s
        self.I_max = I_max

    def solve(self, problem: SelectionProblem, budget: Budget, rng: np.random.Generator) -> SolverResult:
        counter = Counter(budget)
        if problem.size < problem.n:
            raise ValueError("Neighborhood smaller than n")
        if problem.size == problem.n:
            return self.finish(problem, range(problem.size), counter)

        best_members: Optional[List[int]] = None
        best_u = -np.inf
        state: Optional[IncrementalState] = None
        try:
            # Rank the seed pairs by U({i, j}) -- O(k^2) evaluations
            pairs = []
            single = [IncrementalState(problem, [i]) for i in range(problem.size)]
            for i in range(problem.size):
                for j in range(i + 1, problem.size):
                    counter.tick()
                    pairs.append((-single[i].utility_if_added(j), i, j))
            pairs.sort()
            for _, i, j in pairs[:max(1, self.R_s)]:
                state = IncrementalState(problem, [i, j])
                greedy_complete(problem, state, counter)
                local_search(problem, state, counter, self.I_max)
                value = state.utility()
                if better(value, state.members, best_u, best_members):
                    best_u, best_members = value, list(state.members)
                state = None
        except BudgetExhausted:
            if state is not None and state.size == problem.n and \
                    better(state.utility(), state.members, best_u, best_members):
                best_members = list(state.members)
            if best_members is None:
                best_members = self._fill(problem, state)
        return self.finish(problem, best_members, counter)

    @staticmethod
    def _fill(problem: SelectionProblem, state: Optional[IncrementalState]) -> List[int]:
        members = list(state.members) if state is not None else []
        for v in range(problem.size):
            if len(members) >= problem.n:
                break
            if v not in members:
                members.append(v)
        return members
