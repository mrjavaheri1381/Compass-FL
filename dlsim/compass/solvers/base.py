"""
The solver interface (paper Sec. 4.5: the framework is solver-agnostic).

A solver receives a ``SelectionProblem`` (the neighborhood, n, the utility parameters and precomputed arrays), a
``Budget`` and a numpy ``Generator``, and returns a ``SolverResult``. Every solver evaluates candidate sets with the
same utility (``dlsim.compass.core.utility``). Adding a solver requires only a new module that defines a class
decorated with ``@register("name")`` and an import line in ``dlsim/compass/solvers/__init__.py``.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Type

import numpy as np

from dlsim.compass.core.utility import Evaluation, IncrementalState, SelectionProblem, evaluate


@dataclass
class Budget:
    max_evals: int = 0        # 0 = unlimited
    max_time: float = 0.0     # Wall-clock seconds, 0 = unlimited (nondeterministic when > 0)


@dataclass
class SolverResult:
    members: List[int]                    # Indices into the problem
    utility: float
    components: Dict[str, float]
    evaluations: int
    wall_time: float
    certificate: Optional[str] = None     # "optimal" when proven
    bound: Optional[float] = None         # Upper bound on the optimum, when known

    def ids(self, problem: SelectionProblem) -> List:
        return [problem.ids[i] for i in self.members]


class BudgetExhausted(Exception):
    pass


class SolverRefused(Exception):
    """
    Raised when a solver declines an instance (e.g. ``exact`` above its size limit).
    """


class Counter:
    """
    Counts utility evaluations and enforces the budget.
    """

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.evaluations = 0
        self.start = time.perf_counter()

    def tick(self, count: int = 1) -> None:
        self.evaluations += count
        if self.budget.max_evals and self.evaluations > self.budget.max_evals:
            raise BudgetExhausted()
        if self.budget.max_time and time.perf_counter() - self.start > self.budget.max_time:
            raise BudgetExhausted()

    def exhausted(self) -> bool:
        if self.budget.max_evals and self.evaluations >= self.budget.max_evals:
            return True
        return bool(self.budget.max_time and time.perf_counter() - self.start > self.budget.max_time)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start


class Solver:
    name = "base"

    def __init__(self, **options) -> None:
        self.options = options

    def solve(self, problem: SelectionProblem, budget: Budget, rng: np.random.Generator) -> SolverResult:
        raise NotImplementedError

    @staticmethod
    def finish(problem: SelectionProblem, members: Sequence[int], counter: Counter,
               certificate: Optional[str] = None, bound: Optional[float] = None) -> SolverResult:
        members = sorted(members)
        evaluation: Evaluation = evaluate(problem, members)
        return SolverResult(members, evaluation.utility, evaluation.components, counter.evaluations, counter.elapsed,
                            certificate, bound)


_REGISTRY: Dict[str, Type[Solver]] = {}


def register(name: str) -> Callable[[Type[Solver]], Type[Solver]]:
    def decorator(cls: Type[Solver]) -> Type[Solver]:
        if name in _REGISTRY:
            raise ValueError("Solver %s registered twice" % name)
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return decorator


def registry() -> Dict[str, Type[Solver]]:
    return dict(_REGISTRY)


# ---------------------------------------------------------------------- shared building blocks

def better(u_new: float, members_new: Sequence[int], u_old: float, members_old: Optional[Sequence[int]]) -> bool:
    """
    Strictly higher utility wins; equal utility is broken by the lexicographically smaller sorted id tuple.
    """
    if members_old is None:
        return True
    if u_new > u_old:
        return True
    return u_new == u_old and tuple(sorted(members_new)) < tuple(sorted(members_old))


def greedy_complete(problem: SelectionProblem, state: IncrementalState, counter: Counter) -> None:
    """
    Grow ``state`` to size n by repeatedly adding the neighbor that maximizes U (ties: smaller index).
    """
    while state.size < problem.n:
        best_v, best_u = None, -np.inf
        for v in range(problem.size):
            if state.in_set[v]:
                continue
            counter.tick()
            u = state.utility_if_added(v)
            if u > best_u:
                best_u, best_v = u, v
        state.add(best_v)


def best_swap(problem: SelectionProblem, state: IncrementalState, counter: Counter, forbidden=None):
    """
    (u*, v*, U') of the best 1-swap; ties broken by (u, v) index order. ``forbidden(u, v)`` excludes moves.
    """
    best = (None, None, -np.inf)
    for u in sorted(state.members):
        for v in range(problem.size):
            if state.in_set[v] or (forbidden is not None and forbidden(u, v)):
                continue
            counter.tick()
            value = state.utility_if_swapped(u, v)
            if value > best[2]:
                best = (u, v, value)
    return best


def random_members(problem: SelectionProblem, rng: np.random.Generator) -> List[int]:
    return sorted(int(i) for i in rng.choice(problem.size, size=problem.n, replace=False))
