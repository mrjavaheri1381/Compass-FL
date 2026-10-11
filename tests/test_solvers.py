import itertools

import numpy as np
import pytest

from dlsim.compass.core.utility import evaluate
from dlsim.compass.solvers import Budget, SolverRefused, available_solvers, create_solver
from dlsim.compass.solvers.greedy_swap import GreedySwapSolver
from tests.helpers import random_problem


def brute_force(problem):
    best, best_u = None, -1.0
    for combo in itertools.combinations(range(problem.size), problem.n):
        u = evaluate(problem, combo).utility
        if u > best_u:
            best, best_u = list(combo), u
    return best, best_u


CASES = [(seed, k, n, form) for seed in range(6) for k, n in [(6, 2), (9, 3), (12, 5)]
         for form in ("multiplicative", "additive")]


@pytest.mark.parametrize("seed,k,n,form", CASES)
def test_exact_equals_brute_force_and_dominates(seed, k, n, form):
    """Test 3: exact == brute force; no solver beats exact; report greedy_swap's optimality gap."""
    problem, _, _ = random_problem(seed, k=k, n=n, form=form)
    best, best_u = brute_force(problem)
    exact = create_solver("exact").solve(problem, Budget(), np.random.default_rng(0))
    assert exact.utility == pytest.approx(best_u, rel=1e-12)
    assert exact.members == best
    assert exact.certificate == "optimal"
    for name in available_solvers():
        result = create_solver(name).solve(problem, Budget(), np.random.default_rng(seed))
        assert len(result.members) == n and len(set(result.members)) == n
        assert result.utility <= exact.utility * (1 + 1e-12)
        assert result.utility == pytest.approx(evaluate(problem, result.members).utility)
    greedy = create_solver("greedy_swap").solve(problem, Budget(), np.random.default_rng(0))
    gap = 1 - greedy.utility / exact.utility
    print("greedy_swap optimality gap (seed=%d, k=%d, n=%d, %s): %.4f" % (seed, k, n, form, gap))
    assert gap >= -1e-12


def test_greedy_swap_is_deterministic_and_incremental():
    """Test 2 (solver part): the solver's reported utility equals a full recomputation."""
    problem, _, _ = random_problem(7, k=14, n=5)
    a = GreedySwapSolver(R_s=3, I_max=20).solve(problem, Budget(), np.random.default_rng(1))
    b = GreedySwapSolver(R_s=3, I_max=20).solve(problem, Budget(), np.random.default_rng(99))
    assert a.members == b.members and a.utility == b.utility
    assert a.utility == evaluate(problem, a.members).utility


def test_exact_refuses_large_instances():
    problem, _, _ = random_problem(1, k=20, n=8)
    with pytest.raises(SolverRefused):
        create_solver("exact").__class__(max_combinations=1000).solve(problem, Budget(), np.random.default_rng(0))


def test_budget_is_respected():
    problem, _, _ = random_problem(2, k=14, n=5)
    for name in ("greedy_swap", "tabu", "simulated_annealing", "genetic", "lns"):
        result = create_solver(name).solve(problem, Budget(max_evals=50), np.random.default_rng(0))
        assert result.evaluations <= 51
        assert len(result.members) == problem.n


def test_seeded_metaheuristics_reproducible():
    problem, _, _ = random_problem(3, k=14, n=5)
    for name in ("simulated_annealing", "genetic", "lns", "random", "tabu"):
        a = create_solver(name).solve(problem, Budget(), np.random.default_rng(5))
        b = create_solver(name).solve(problem, Budget(), np.random.default_rng(5))
        assert a.members == b.members
