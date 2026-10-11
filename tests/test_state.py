import math

import numpy as np
import pytest

from dlsim.compass.core import estimators as est
from dlsim.compass.core.view import ClassPrior, PopulationView, StateRecord

E, C, Z = 2, 1.0, 1_000_000.0
ALPHA, GAMMA, BETA = 0.3, 0.1, 0.2


def make_view():
    view = PopulationView({0: ClassPrior(f=10.0, B=1e5, sigma=5.0), 1: ClassPrior(f=40.0, B=1e6, sigma=1.0)})
    for node in range(4):
        view.add_node(node, kappa=node % 2, D=500)
    return view


def timestamps(D, f, B, t_ack=100.0):
    t_first = t_ack + E * D * C / f
    return est.Measurement(t_ack, t_first, t_first + Z / B)


def test_q_update_formula_and_non_participants():
    """Test 4: Q update incl. lateness = inf; non-participants unchanged."""
    view = make_view()
    view.apply_record(StateRecord(0, 1, lateness=0.0), E, C, Z, ALPHA, GAMMA, BETA)
    assert view.get(0).Q == pytest.approx((1 - ALPHA) * 1.0 + ALPHA * 1.0)
    view.apply_record(StateRecord(0, 2, lateness=7.5), E, C, Z, ALPHA, GAMMA, BETA)
    assert view.get(0).Q == pytest.approx((1 - ALPHA) * 1.0 + ALPHA * math.exp(-GAMMA * 7.5))
    q_before = view.get(0).Q
    view.apply_record(StateRecord(0, 3, lateness=math.inf), E, C, Z, ALPHA, GAMMA, BETA)
    assert view.get(0).Q == pytest.approx((1 - ALPHA) * q_before)
    for node in (1, 2, 3):
        assert view.get(node).Q == 1.0 and view.get(node).m == 0


def test_first_measurement_replaces_prior_and_converges():
    """Test 5 (estimator part): beta_1 = 1 replaces the prior; noiseless measurements converge to the truth."""
    view = make_view()
    true_f, true_B = 23.0, 345_678.0
    for round in range(1, 30):
        record = StateRecord(0, round, 0.0, timestamps(500, true_f, true_B))
        view.apply_record(record, E, C, Z, ALPHA, GAMMA, BETA)
        if round == 1:
            assert view.get(0).f == pytest.approx(true_f) and view.get(0).B == pytest.approx(true_B)
    assert view.get(0).f == pytest.approx(true_f) and view.get(0).B == pytest.approx(true_B)
    assert view.get(0).m == 29


def test_sigma_converges_to_fluctuation_std():
    """Test 11: sigma converges to the std of the injected fluctuation; censored samples leave B (and sigma) unchanged."""
    view = make_view()
    rng = np.random.default_rng(0)
    base_f, base_B, cv = 20.0, 5e5, 0.15
    errors = []
    beta_small = 0.02
    for round in range(1, 4000):
        factor = rng.lognormal(-0.5 * math.log(1 + cv ** 2), math.sqrt(math.log(1 + cv ** 2)))
        m = timestamps(500, base_f * factor, base_B)
        view.apply_record(StateRecord(0, round, 0.0, m), E, C, Z, ALPHA, GAMMA, beta_small)
        errors.append(m.delta_hat)
    # The prediction error is dominated by the fluctuation of the compute time; compare with its empirical std.
    expected = float(np.std(errors))
    assert view.get(0).sigma == pytest.approx(expected, rel=0.2)

    state = view.get(0)
    B_before, sigma_before, f_before = state.B, state.sigma, state.f
    censored = est.Measurement(10.0, 10.0 + E * 500 / 50.0, 10.0 + E * 500 / 50.0 + Z / 1e3, censored=True)
    view.apply_record(StateRecord(0, 5000, lateness=30.0, measurement=censored), E, C, Z, ALPHA, GAMMA, beta_small)
    assert state.B == B_before and state.sigma == sigma_before
    assert state.f != f_before                     # f is still updated: T_cmp is unaffected
    assert state.Q == pytest.approx((1 - ALPHA) * 1.0 + ALPHA * 1.0, rel=1e-6) or state.Q > 0.99  # never late


def test_stale_records_are_discarded_not_reordered():
    view = make_view()
    view.apply_record(StateRecord(1, 5, 0.0, timestamps(500, 30.0, 2e6)), E, C, Z, ALPHA, GAMMA, BETA)
    snapshot = (view.get(1).f, view.get(1).Q, view.get(1).m)
    status = view.apply_record(StateRecord(1, 4, math.inf), E, C, Z, ALPHA, GAMMA, BETA)
    assert status == "stale"
    assert (view.get(1).f, view.get(1).Q, view.get(1).m) == snapshot
    assert view.apply_record(StateRecord(99, 6, 0.0), E, C, Z, ALPHA, GAMMA, BETA) == "unknown"


def test_snapshot_is_immutable_copy():
    view = make_view()
    snap = view.snapshot(round=3)
    view.apply_record(StateRecord(0, 3, math.inf), E, C, Z, ALPHA, GAMMA, BETA)
    view.mark_participated([0], 3)
    assert snap.get(0).Q == 1.0 and snap.get(0).tau == 0
    assert view.get(0).Q < 1.0 and view.get(0).tau == 3


def test_new_node_gets_priors():
    view = make_view()
    state = view.get(1)
    assert (state.tau, state.Q, state.m, state.f, state.B, state.sigma) == (0, 1.0, 0, 40.0, 1e6, 1.0)


def test_contention_aware_prediction():
    assert est.contention_rate(1e6, 2e6, 4) == 5e5
    assert est.contention_rate(1e5, 2e6, 4) == 1e5
    assert est.predict_completion(E, 500, C, Z, 25.0, 1e6, upload_rate=5e5) == pytest.approx(E * 500 / 25.0 + 2.0)
