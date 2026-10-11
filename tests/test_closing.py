import math

import pytest
import torch

from dlsim.compass.aggregation import fedavg
from dlsim.compass.closing import ClientInfo, ClosingParams, make_policy

E, C, Z = 1, 1.0, 1_000_000.0


def params(z=2.0, delta_dl=500.0, grace=30.0, contention=False, n_target=3):
    return ClosingParams(E=E, c=C, Z=Z, z=z, delta_dl=delta_dl, grace=grace, rho=0.9, contention_aware=contention,
                         B_down_aggregator=1e12, n_target=n_target)


CLIENTS = [ClientInfo(node, D=500, f=50.0, B=1e6, sigma=1.0) for node in range(3)]
TRUE_DURATION = 500 / 50.0 + Z / 1e6  # 11 s from the acknowledgement


def simulate(policy, durations, ack_delay=1.0):
    """
    Drive the policy with acknowledgements at ack_delay and arrivals at ack + duration (None = never).
    Returns the close time.
    """
    events = []
    for node, duration in durations.items():
        events.append((ack_delay, "ack", node))
        if duration is not None:
            events.append((ack_delay + duration, "arrival", node))
    events.sort()
    for time, kind, node in events:
        while policy.next_event() is not None and policy.next_event() < time:
            policy.advance(policy.next_event())
        policy.advance(time)
        if kind == "ack":
            policy.on_ack(node, time)
        else:
            policy.on_arrival(node, time - Z / 1e6, time, 0.1)
    while policy.next_event() is not None:
        policy.advance(policy.next_event())
    return policy.close_time


def test_predictive_no_false_stragglers_with_deterministic_speeds():
    """Test 10a."""
    policy = make_policy("predictive", CLIENTS, 0.0, params(z=2.0))
    close = simulate(policy, {0: TRUE_DURATION, 1: TRUE_DURATION, 2: TRUE_DURATION})
    assert sorted(policy.accepted()) == [0, 1, 2]
    assert close == pytest.approx(1.0 + TRUE_DURATION)
    assert all(policy.lateness(t) == 0.0 for t in policy.tracks.values())


def test_predictive_cuts_injected_slowdown_at_its_deadline():
    """Test 10b: the slowed-down client is a straggler at d_i and the round does not wait for it."""
    policy = make_policy("predictive", CLIENTS, 0.0, params(z=2.0))
    close = simulate(policy, {0: TRUE_DURATION, 1: TRUE_DURATION, 2: 4 * TRUE_DURATION})
    deadline = 1.0 + TRUE_DURATION + 2.0 * 1.0
    assert sorted(policy.accepted()) == [0, 1]
    assert policy.tracks[2].status == "straggler"
    assert close == pytest.approx(deadline)
    assert close < 1.0 + 4 * TRUE_DURATION
    # Its arrival (33 s after the deadline) is beyond the grace period -> lateness = inf, no measurement
    assert math.isinf(policy.lateness(policy.tracks[2])) and policy.measurement(policy.tracks[2]) is None


def test_late_arrival_within_grace_is_recorded():
    policy = make_policy("predictive", CLIENTS, 0.0, params(z=2.0, grace=60.0))
    simulate(policy, {0: TRUE_DURATION, 1: TRUE_DURATION, 2: 2 * TRUE_DURATION})
    track = policy.tracks[2]
    assert not track.aggregated
    assert policy.lateness(track) == pytest.approx(2 * TRUE_DURATION - TRUE_DURATION - 2.0)
    assert policy.measurement(track) is not None


def test_large_z_behaves_like_fixed_deadline():
    """Test 10c: as z -> inf, d_i' = dispatch + delta_dl for everybody."""
    durations = {0: TRUE_DURATION, 1: 30.0, 2: 300.0}
    predictive = make_policy("predictive", CLIENTS, 0.0, params(z=1e9, delta_dl=100.0))
    fixed = make_policy("fixed_deadline", CLIENTS, 0.0, params(delta_dl=100.0))
    assert simulate(predictive, durations) == pytest.approx(simulate(fixed, durations)) == pytest.approx(100.0)
    assert sorted(predictive.accepted()) == sorted(fixed.accepted()) == [0, 1]


def test_missing_ack_is_straggler_at_cap_with_infinite_lateness():
    policy = make_policy("predictive", CLIENTS, 0.0, params(delta_dl=100.0))
    for node in (0, 1):
        policy.on_ack(node, 1.0)
    for node in (0, 1):
        policy.advance(1.0 + TRUE_DURATION)
        policy.on_arrival(node, TRUE_DURATION, 1.0 + TRUE_DURATION, 0.1)
    assert not policy.closed
    policy.advance(100.0)
    assert policy.closed and policy.close_time == 100.0
    assert math.isinf(policy.lateness(policy.tracks[2]))


def test_wait_all_and_over_selection():
    durations = {0: TRUE_DURATION, 1: 30.0, 2: 60.0}
    wait_all = make_policy("wait_all", CLIENTS, 0.0, params())
    assert simulate(wait_all, durations) == pytest.approx(61.0)
    assert sorted(wait_all.accepted()) == [0, 1, 2]
    # wait_all aggregates late clients but their lateness still uses the predictive deadline (RQ3)
    assert wait_all.lateness(wait_all.tracks[2]) > 0
    over = make_policy("over_selection", CLIENTS, 0.0, params(n_target=2))
    assert simulate(over, durations) == pytest.approx(31.0)
    assert sorted(over.accepted()) == [0, 1]


def test_effective_deadline_is_capped():
    """RQ2: d_i' = min(d_i, dispatch + delta_dl)."""
    policy = make_policy("predictive", CLIENTS, 0.0, params(z=1000.0, delta_dl=50.0))
    policy.on_ack(0, 1.0)
    assert policy.tracks[0].raw_deadline > 50.0 and policy.tracks[0].deadline == 50.0


def test_contention_aware_deadlines():
    clients = [ClientInfo(node, D=500, f=50.0, B=1e6, sigma=0.0) for node in range(4)]
    p = params(z=1.0, contention=True)
    p.B_down_aggregator = 2e6                      # 4 overlapping uploads share 2 MB/s -> 0.5 MB/s each
    policy = make_policy("predictive", clients, 0.0, p)
    policy.on_ack(0, 1.0)
    assert policy.tracks[0].k == 1 and policy.tracks[0].deadline == pytest.approx(1.0 + 10.0 + 1.0)
    for node in (1, 2, 3):
        policy.on_ack(node, 1.0)
    assert policy.tracks[0].k == 4
    assert policy.tracks[0].deadline == pytest.approx(1.0 + 10.0 + Z / 0.5e6)
    policy.on_ack  # Non-overlapping windows do not count
    late = make_policy("predictive", clients[:2], 0.0, p)
    late.on_ack(0, 1.0)
    late.on_ack(1, 100.0)
    assert late.tracks[0].k == 1 and late.tracks[1].k == 1


def test_fedavg_weighted_by_dataset_size_and_empty_keeps_model():
    """Test 8."""
    models = []
    for value in (1.0, 4.0):
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(value)
        models.append(model)
    avg = fedavg(models, [300, 100], previous=None)
    for p in avg.parameters():
        assert torch.allclose(p, torch.full_like(p, (300 * 1.0 + 100 * 4.0) / 400))
    previous = models[1]
    kept = fedavg([], [], previous=previous)
    for a, b in zip(kept.parameters(), previous.parameters()):
        assert torch.equal(a, b)
