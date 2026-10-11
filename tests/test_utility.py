import itertools
import math

import numpy as np
import pytest

from dlsim.compass.config import ConfigError, load_config, validate_weights
from dlsim.compass.core.utility import IncrementalState, UtilityParams, evaluate
from tests.helpers import random_problem


@pytest.mark.parametrize("seed", range(20))
def test_repr_equals_cosine_of_means(seed):
    """Test 1: Phi_repr from S and r equals the cosine of the mean normalized vectors."""
    problem, Y, idx = random_problem(seed, k=12, n=5)
    rng = np.random.default_rng(seed + 100)
    members = sorted(rng.choice(problem.size, size=int(rng.integers(1, 9)), replace=False).tolist())
    mean_c = Y[[idx[i] for i in members]].mean(axis=0)
    mean_v = Y.mean(axis=0)
    expected = float(mean_c @ mean_v / (np.linalg.norm(mean_c) * np.linalg.norm(mean_v)))
    got = evaluate(problem, members).components["repr"]
    assert got == pytest.approx(min(1.0, max(problem.params.eps, expected)), abs=1e-12)


def test_components_match_definitions():
    problem, _, _ = random_problem(3, k=8, n=4)
    members = [0, 2, 5, 7]
    comps = evaluate(problem, members).components
    p = problem.params
    assert comps["comp"] == pytest.approx(min(1, problem.f[members].min() / p.f_ref))
    assert comps["comm"] == pytest.approx(1 - problem.L[members].max() / p.L_max)
    pairs = [problem.S[i, j] for i, j in itertools.combinations(members, 2)]
    assert comps["coh"] == pytest.approx(sum(pairs) / len(pairs))
    assert comps["fair"] == pytest.approx(np.mean(np.minimum(problem.stale[members], p.dtau_cap) / p.dtau_cap))
    assert comps["strag"] == pytest.approx(problem.Q[members].mean())
    assert comps["data"] == pytest.approx(comps["coh"] ** p.w_c * comps["repr"] ** p.w_r)
    u = evaluate(problem, members).utility
    w = p.weight_map
    assert u == pytest.approx(math.prod(comps[m] ** w[m] for m in w))


def test_additive_form_and_wc_zero():
    problem, _, _ = random_problem(4, form="additive", w_c=0.0, w_r=1.0)
    ev = evaluate(problem, [0, 1, 2, 3])
    assert ev.components["data"] == pytest.approx(ev.components["repr"])
    w = problem.params.weight_map
    assert ev.utility == pytest.approx(sum(w[m] * ev.components[m] for m in w))


def test_components_are_clipped():
    problem, _, _ = random_problem(5)
    problem.L[:] = problem.params.L_max  # Phi_comm = 0 before clipping
    problem.f[:] = 1e6                   # Phi_comp > 1 before clipping
    comps = evaluate(problem, [0, 1]).components
    assert comps["comm"] == problem.params.eps
    assert comps["comp"] == 1.0
    assert all(problem.params.eps <= v <= 1 for v in comps.values())


@pytest.mark.parametrize("seed", range(10))
def test_incremental_equals_full(seed):
    """Test 2 (bookkeeping part): every add and every swap evaluated in O(1) equals full recomputation."""
    problem, _, _ = random_problem(seed, k=11, n=5)
    rng = np.random.default_rng(seed)
    order = rng.permutation(problem.size).tolist()
    state = IncrementalState(problem)
    for v in order[:problem.n]:
        expected = evaluate(problem, state.members + [v]).utility
        assert state.utility_if_added(v) == pytest.approx(expected, rel=1e-12)
        state.add(v)
        assert state.utility() == pytest.approx(evaluate(problem, state.members).utility, rel=1e-12)
    for _ in range(40):
        u = state.members[int(rng.integers(problem.n))]
        outside = [v for v in range(problem.size) if not state.in_set[v]]
        v = outside[int(rng.integers(len(outside)))]
        swapped = [x for x in state.members if x != u] + [v]
        assert state.utility_if_swapped(u, v) == pytest.approx(evaluate(problem, swapped).utility, rel=1e-12)
        state.swap(u, v)
        assert state.utility() == pytest.approx(evaluate(problem, state.members).utility, rel=1e-12)


def test_weight_validation_fails_loudly():
    with pytest.raises(ConfigError):
        validate_weights({"comp": 0.0, "comm": 0.25, "data": 0.25, "fair": 0.25, "strag": 0.25})
    with pytest.raises(ConfigError):
        validate_weights({"comp": 0.3, "comm": 0.3, "data": 0.3, "fair": 0.3, "strag": 0.3})
    with pytest.raises(ConfigError):
        load_config(overrides=["utility.weights.comm=0.5"])
    with pytest.raises(ConfigError):
        load_config(overrides=["utility.w_c=0", "utility.w_r=0"])
    with pytest.raises(ConfigError):
        load_config(overrides=["utility.no_such_key=1"])
    config = load_config(overrides=["utility.w_c=0", "solver=tabu", "weights.comm=0.3", "weights.comp=0.1", "seed=3"])
    assert config.utility.w_c == 0 and config.solver.name == "tabu" and config.seed == 3
