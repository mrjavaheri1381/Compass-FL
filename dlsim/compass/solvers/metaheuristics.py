"""
Metaheuristic solvers for the solver-agnostic comparison: tabu search, simulated annealing, a genetic algorithm
(the existing SPMA memetic engine, which keeps the set size fixed), large neighborhood search, and a uniformly random
feasible set. All are seeded through the ``rng`` they receive and respect the evaluation/time budget.
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from dlsim.compass.core.utility import IncrementalState, SelectionProblem, evaluate
from dlsim.compass.solvers.base import (Budget, BudgetExhausted, Counter, Solver, SolverResult, best_swap, better,
                                        greedy_complete, random_members, register)


def _greedy_start(problem: SelectionProblem, counter: Counter) -> IncrementalState:
    """
    Greedy construction from the empty set (the first pick is the best singleton).
    """
    state = IncrementalState(problem)
    greedy_complete(problem, state, counter)
    return state


@register("tabu")
class TabuSolver(Solver):
    """
    Tabu search over 1-swaps. A node that left (entered) the set may not re-enter (leave) it for ``tenure`` iterations,
    unless the move beats the best set found (aspiration). Stops after ``iterations`` moves or ``patience`` moves
    without improvement.
    """

    def __init__(self, iterations: int = 200, tenure: int = 5, patience: int = 50, **options) -> None:
        super().__init__(**options)
        self.iterations, self.tenure, self.patience = iterations, tenure, patience

    def solve(self, problem: SelectionProblem, budget: Budget, rng: np.random.Generator) -> SolverResult:
        counter = Counter(budget)
        best: Optional[List[int]] = None
        best_u = -math.inf
        try:
            state = _greedy_start(problem, counter)
            best, best_u = list(state.members), state.utility()
            tabu_until: Dict[int, int] = {}
            since_improvement = 0
            for iteration in range(self.iterations):
                if problem.size == problem.n:
                    break

                def forbidden(u: int, v: int) -> bool:
                    if tabu_until.get(u, -1) < iteration and tabu_until.get(v, -1) < iteration:
                        return False
                    return state.utility_if_swapped(u, v) <= best_u  # Aspiration

                u, v, value = best_swap(problem, state, counter, forbidden)
                if u is None:
                    break
                state.swap(u, v)
                tabu_until[u] = iteration + self.tenure
                tabu_until[v] = iteration + self.tenure
                if better(value, state.members, best_u, best):
                    best, best_u = list(state.members), value
                    since_improvement = 0
                else:
                    since_improvement += 1
                    if since_improvement >= self.patience:
                        break
        except BudgetExhausted:
            pass
        if best is None:
            best = random_members(problem, rng)
        return self.finish(problem, best, counter)


@register("simulated_annealing")
class AnnealingSolver(Solver):
    """
    Simulated annealing on log U with random 1-swap moves and geometric cooling, starting from a random set.
    """

    def __init__(self, iterations: int = 2000, t0: float = 0.1, cooling: float = 0.995, **options) -> None:
        super().__init__(**options)
        self.iterations, self.t0, self.cooling = iterations, t0, cooling

    def solve(self, problem: SelectionProblem, budget: Budget, rng: np.random.Generator) -> SolverResult:
        counter = Counter(budget)
        state = IncrementalState(problem, random_members(problem, rng))
        current = state.utility()
        best, best_u = list(state.members), current
        temperature = self.t0
        try:
            for _ in range(self.iterations):
                if problem.size == problem.n:
                    break
                u = state.members[int(rng.integers(problem.n))]
                outside = np.flatnonzero(~state.in_set)
                v = int(outside[int(rng.integers(len(outside)))])
                counter.tick()
                value = state.utility_if_swapped(u, v)
                delta = math.log(max(value, 1e-300)) - math.log(max(current, 1e-300))
                if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-12)):
                    state.swap(u, v)
                    current = value
                    if better(current, state.members, best_u, best):
                        best, best_u = list(state.members), current
                temperature *= self.cooling
        except BudgetExhausted:
            pass
        return self.finish(problem, best, counter)


@register("genetic")
class GeneticSolver(Solver):
    """
    Genetic algorithm with a fixed-size set encoding: the existing Subset-Preserving Memetic Algorithm
    (``dlsim/compass/solvers/spma.py``). Its crossover and mutation keep exactly n selected genes, so the repair step is the
    identity; the budget is enforced by the evaluation counter.
    """

    def __init__(self, population: int = 20, generations: int = 20, elitism: int = 2, mutation_rate: float = 0.3,
                 local_search_depth: int = 3, tournament_size: int = 3, **options) -> None:
        super().__init__(**options)
        self.params = dict(population_size=population, generations=generations, elitism_count=elitism,
                           mutation_rate=mutation_rate, local_search_depth=local_search_depth,
                           tournament_size=tournament_size)

    def solve(self, problem: SelectionProblem, budget: Budget, rng: np.random.Generator) -> SolverResult:
        from random import Random
        from dlsim.compass.solvers.spma import FitnessEvaluator, SPMASolver

        counter = Counter(budget)
        best: Dict[str, object] = {"u": -math.inf, "members": None}

        class Evaluator(FitnessEvaluator):
            def evaluate(self, chromosome):
                counter.tick()
                members = chromosome.selected()
                value = evaluate(problem, members).utility
                if better(value, members, best["u"], best["members"]):
                    best["u"], best["members"] = value, members
                return value

        engine = SPMASolver(rng=Random(int(rng.integers(2 ** 63))), **self.params)
        try:
            engine.solve(problem.size, problem.n, Evaluator())
        except BudgetExhausted:
            pass
        members = best["members"] if best["members"] is not None else random_members(problem, rng)
        return self.finish(problem, members, counter)


@register("lns")
class LNSSolver(Solver):
    """
    Large neighborhood search: start from greedy_swap's construction, repeatedly destroy q random members and rebuild
    greedily (or exactly over the freed slots), keeping the result if it is better.
    """

    def __init__(self, iterations: int = 100, destroy: int = 2, rebuild: str = "greedy", **options) -> None:
        super().__init__(**options)
        self.iterations, self.destroy, self.rebuild = iterations, destroy, rebuild

    def solve(self, problem: SelectionProblem, budget: Budget, rng: np.random.Generator) -> SolverResult:
        counter = Counter(budget)
        best: Optional[List[int]] = None
        best_u = -math.inf
        try:
            state = _greedy_start(problem, counter)
            best, best_u = sorted(state.members), state.utility()
            q = max(1, min(self.destroy, problem.n))
            for _ in range(self.iterations):
                if problem.size == problem.n:
                    break
                keep = sorted(best)
                removed = set(int(x) for x in rng.choice(keep, size=q, replace=False))
                kept = [v for v in keep if v not in removed]
                candidate = self._rebuild(problem, kept, counter)
                counter.tick()
                value = evaluate(problem, candidate).utility
                if better(value, candidate, best_u, best):
                    best, best_u = sorted(candidate), value
        except BudgetExhausted:
            pass
        if best is None:
            best = random_members(problem, rng)
        return self.finish(problem, best, counter)

    def _rebuild(self, problem: SelectionProblem, kept: List[int], counter: Counter) -> List[int]:
        if self.rebuild == "exact":
            import itertools
            free = [v for v in range(problem.size) if v not in kept]
            need = problem.n - len(kept)
            best, best_u = None, -math.inf
            for extra in itertools.combinations(free, need):
                counter.tick()
                members = kept + list(extra)
                value = evaluate(problem, members).utility
                if better(value, members, best_u, best):
                    best, best_u = members, value
            return best
        state = IncrementalState(problem, kept)
        greedy_complete(problem, state, counter)
        return list(state.members)


@register("random")
class RandomSolver(Solver):
    """
    A uniformly random feasible set (sanity baseline; with the full protocol this is the compass_random selector).
    """

    def solve(self, problem: SelectionProblem, budget: Budget, rng: np.random.Generator) -> SolverResult:
        counter = Counter(budget)
        counter.evaluations = 1
        return self.finish(problem, random_members(problem, rng), counter)
