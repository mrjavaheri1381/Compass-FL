"""
Solver registry. To add a solver: write a module with a ``Solver`` subclass decorated with ``@register("name")`` and
import it below; it is then selectable with ``solver.name=<name>``.
"""
from typing import List

from dlsim.compass.solvers.base import (Budget, BudgetExhausted, Solver, SolverRefused, SolverResult, register,
                                        registry)
from dlsim.compass.solvers import greedy_swap, exact, metaheuristics  # noqa: F401  (registration side effects)


def available_solvers() -> List[str]:
    return sorted(registry())


def create_solver(name: str, solver_config=None) -> Solver:
    """
    Instantiate a solver by name with its options from ``config.solver``.
    """
    cls = registry().get(name)
    if cls is None:
        raise ValueError("Unknown solver %s (available: %s)" % (name, available_solvers()))
    options = {}
    if solver_config is not None:
        import dataclasses
        options = {"R_s": solver_config.R_s, "I_max": solver_config.I_max}
        section = {"exact": solver_config.exact, "tabu": solver_config.tabu,
                   "simulated_annealing": solver_config.annealing, "genetic": solver_config.genetic,
                   "lns": solver_config.lns}.get(name)
        if section is not None:
            options.update(dataclasses.asdict(section))
    return cls(**options)


def budget_from_config(solver_config) -> Budget:
    return Budget(max_evals=solver_config.budget.max_evals, max_time=solver_config.budget.max_time)
