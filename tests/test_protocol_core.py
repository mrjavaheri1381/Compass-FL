import math

import numpy as np
import pytest

from dlsim.compass.core import hashing
from dlsim.compass.core.context import Proposal, SelectionContext, rank, verify
from dlsim.compass.core.neighborhood import neighborhood, neighborhood_or_decline
from dlsim.compass.core.similarity import assemble, estimate_network_sum
from dlsim.compass.core.utility import UtilityParams, evaluate
from dlsim.compass.core.view import ClassPrior, PopulationView, StateRecord
from dlsim.compass.solvers import Budget, create_solver


def test_neighborhood_ties_by_id_and_decline():
    """Test 6."""
    rtt = {5: 0.1, 3: 0.1, 9: 0.05, 1: 0.2, 7: 0.5, 2: 0.1}
    pool = [1, 2, 3, 5, 7, 9, 0]
    assert neighborhood(0, pool, rtt, L_max=0.4, k_nb=3) == [9, 2, 3]       # 2, 3, 5 tie at 0.1 -> ids
    assert neighborhood(0, pool, rtt, L_max=0.4, k_nb=10) == [9, 2, 3, 5, 1]  # 7 unreachable (>= L_max)
    assert neighborhood(0, pool, rtt, L_max=0.1, k_nb=10) == [9]             # strict L < L_max
    assert neighborhood_or_decline(0, pool, rtt, L_max=0.1, k_nb=10, n=2) is None
    assert neighborhood_or_decline(0, pool, rtt, L_max=0.4, k_nb=3, n=3) == [9, 2, 3]


def make_context(seed=0, M=8, n=3):
    rng = np.random.default_rng(seed)
    pool = list(range(M))
    Y = rng.dirichlet([0.5] * 10, size=M)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    S = Y @ Y.T
    r = S.sum(axis=1) + 5.0
    params = UtilityParams.create(f_ref=30.0, L_max=0.4, dtau_cap=10.0)
    view = PopulationView({0: ClassPrior(20.0, 1e6, 1.0)})
    for node in pool:
        view.add_node(node, 0, 300)
    snap = view.snapshot(round=4)
    ctx = SelectionContext(round=4, k=0, snapshot_id=snap.snapshot_id, pool=pool, S=S, r=r,
                           f=rng.uniform(10, 50, M), B=np.full(M, 1e6), Q=rng.uniform(0.3, 1, M),
                           stale=rng.integers(1, 8, M).astype(float), net_sum=float(r.sum()) * 3, params=params,
                           n=n, k_nb=6, eligible=pool)
    return ctx, view, rng


def candidate_proposal(ctx, aggregator, rtt):
    members = neighborhood_or_decline(aggregator, ctx.pool, rtt, ctx.params.L_max, ctx.k_nb, ctx.n)
    problem = ctx.problem(aggregator, members, rtt)
    result = create_solver("greedy_swap").solve(problem, Budget(), np.random.default_rng(0))
    clients = result.ids(problem)
    return Proposal(aggregator, clients, result.utility, {c: rtt[c] for c in clients}, result.components,
                    ctx.snapshot_id)


def test_verifier_accepts_honest_and_rejects_tampering():
    """Test 7."""
    ctx, _, rng = make_context()
    rtt = {node: float(rng.uniform(0.01, 0.3)) for node in ctx.pool}
    proposal = candidate_proposal(ctx, 0, rtt)
    assert verify(ctx, proposal, 1e-9).ok

    tampered_u = Proposal(**{**proposal.__dict__, "utility": proposal.utility * 1.01})
    assert verify(ctx, tampered_u, 1e-9).reason == "utility_mismatch"

    lat = dict(proposal.latencies)
    bottleneck = max(lat, key=lat.get)                                 # Phi_comm depends on the largest RTT
    lat[bottleneck] = lat[bottleneck] * 0.5                            # claims a better link than measured
    assert verify(ctx, Proposal(**{**proposal.__dict__, "latencies": lat}), 1e-9).reason == "utility_mismatch"

    lat = dict(proposal.latencies)
    lat[proposal.clients[0]] = 1.0                                     # beyond L_max
    assert verify(ctx, Proposal(**{**proposal.__dict__, "latencies": lat}), 1e-9).reason == "invalid_latency"

    wrong_size = Proposal(**{**proposal.__dict__, "clients": proposal.clients[:-1]})
    assert verify(ctx, wrong_size, 1e-9).reason == "wrong_group_size"
    stale = Proposal(**{**proposal.__dict__, "snapshot_id": (99,)})
    assert verify(ctx, stale, 1e-9).reason == "snapshot_mismatch"


def test_snapshot_consistency_with_late_records():
    """Test 12: a late record applied to the live view during selection does not invalidate proposals."""
    ctx, view, rng = make_context(seed=3)
    rtt = {node: float(rng.uniform(0.01, 0.3)) for node in ctx.pool}
    proposal = candidate_proposal(ctx, 1, rtt)
    # A late-arrival record changes Q and f of a proposed client in the live view after the snapshot ...
    client = proposal.clients[0]
    view.apply_record(StateRecord(client, 3, math.inf), 1, 1.0, 1e6, 0.3, 0.1, 0.2)
    assert view.get(client).Q < 1.0
    # ... but the verifier works on the context built from the snapshot, so the proposal stays valid.
    assert verify(ctx, proposal, 1e-9).ok


def test_ranking_is_deterministic():
    a = Proposal(5, [1, 2], 0.5, {})
    b = Proposal(3, [1, 2], 0.5, {})
    c = Proposal(9, [1, 2], 0.7, {})
    assert [p.aggregator for p in rank([a, b, c])] == [9, 3, 5]


def test_pool_assembly_averages_and_flags():
    reports = {1: (3.0, {2: 0.50, 3: 0.20}), 2: (2.5, {1: 0.52, 3: 0.90}), 3: (2.2, {1: 0.20})}
    pool = assemble(reports, tolerance=0.01)
    assert pool.members == [1, 2, 3]
    assert pool.S[0, 1] == pytest.approx(0.51)        # average of both reports
    assert pool.S[1, 2] == pytest.approx(0.90)        # one-sided
    assert pool.flags and pool.flags[0][:2] == (1, 2)
    assert pool.one_sided == 1
    unknown = assemble({1: (1.0, {}), 2: (1.0, {3: 0.1}), 3: (1.0, {2: 0.1})}, 0.01)
    assert unknown.dropped == [1] and unknown.members == [2, 3]
    assert estimate_network_sum([2.0, 4.0], N=10) == pytest.approx(30.0)


def test_hash_pool_and_resample_index():
    nodes = [bytes([i]) * 4 for i in range(30)]
    first = hashing.pool(nodes, seed=7, M=5)
    again = hashing.pool(list(reversed(nodes)), seed=7, M=5)
    assert first == again and len(first) == 5
    assert hashing.pool(nodes, seed=7, M=5, k=1) != first
    assert hashing.pool(nodes, seed=8, M=5) != first


def test_fedcs_schedule():
    """Uploads are sequential: the second slow uploader no longer fits; the cap n is respected."""
    from dlsim.compass.selectors import fedcs_schedule
    compute = {b"a": 1.0, b"b": 1.0, b"c": 1.0, b"d": 5.0}
    upload = {b"a": 1.0, b"b": 1.0, b"c": 4.0, b"d": 1.0}
    # a, b: round time 3; adding c -> 7, adding d -> 6: d is next (6 <= 6.5), then c exceeds
    assert fedcs_schedule(compute, upload, n=4, t_round=6.5) == [b"a", b"b", b"d"]
    assert fedcs_schedule(compute, upload, n=2, t_round=100.0) == [b"a", b"b"]
    assert fedcs_schedule(compute, upload, n=4, t_round=1.5) == []
