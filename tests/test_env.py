import asyncio
import math

import numpy as np
import pytest

from dlsim.compass.config import load_config
from dlsim.compass.core import estimators as est
from dlsim.compass.core.view import ClassPrior, PopulationView, StateRecord
from dlsim.compass.env.devices import DeviceModel
from dlsim.compass.env.latency import LatencyModel, clean_matrix
from dlsim.compass.env.network import FairNetwork

from simulation.discrete_loop import DiscreteLoop


def run(coro_factory):
    loop = DiscreteLoop()
    asyncio.set_event_loop(loop)
    result = {}

    async def main():
        result["value"] = await coro_factory()
        loop.stop()

    asyncio.ensure_future(main())
    try:
        loop.run_forever()
    finally:
        asyncio._set_running_loop(None)  # DiscreteLoop does not reset the running loop when it stops
    return result["value"]


def test_fair_sharing_of_aggregator_ingress():
    """Q3: concurrent uploads share the aggregator's downlink max-min fairly, and it affects completion time."""
    net = FairNetwork(4, small_message_bytes=0)
    net.set_capacity(0, up=100.0, down=100.0)       # aggregator
    for node in (1, 2, 3):
        net.set_capacity(node, up=80.0, down=1000.0)

    async def scenario():
        return await asyncio.gather(*(net.transfer(node, 0, 400, delay=0.0) for node in (1, 2, 3)))

    timings = run(scenario)
    # 3 flows share 100 B/s -> 33.3 B/s each -> 400 B take 12 s; ingress saturated (utilization 1)
    for timing in timings:
        assert timing.ok and timing.t_last == pytest.approx(12.0) and timing.ingress_utilization == pytest.approx(1.0)


def test_uncontended_upload_measures_client_bandwidth():
    net = FairNetwork(2, small_message_bytes=0)
    net.set_capacity(0, up=1e9, down=1000.0)
    net.set_capacity(1, up=100.0, down=1e9)
    timing = run(lambda: net.transfer(1, 0, 500, delay=0.25))
    assert timing.t_first == pytest.approx(0.25) and timing.t_last == pytest.approx(5.25)
    assert timing.ingress_utilization == pytest.approx(0.1)  # 100 of 1000 B/s


def test_rates_are_reallocated_when_a_flow_ends():
    net = FairNetwork(3, small_message_bytes=0)
    net.set_capacity(0, up=1e9, down=100.0)
    net.set_capacity(1, up=1e9, down=1e9)
    net.set_capacity(2, up=1e9, down=1e9)

    async def scenario():
        return await asyncio.gather(net.transfer(1, 0, 100, 0.0), net.transfer(2, 0, 300, 0.0))

    short, long = run(scenario)
    assert short.t_last == pytest.approx(2.0)        # 50 B/s each until t = 2
    assert long.t_last == pytest.approx(4.0)         # then 100 B/s for the remaining 200 B


def test_estimates_converge_under_simulated_timing():
    """Test 5: f, B estimated from simulator timestamps converge to the truth (uncontended, no jitter)."""
    E, D, c, Z = 2, 400, 1.0, 200_000
    true_f, true_up = 25.0, 150_000.0
    net = FairNetwork(2, small_message_bytes=0)
    net.set_capacity(0, up=1e9, down=1e9)
    net.set_capacity(1, up=true_up, down=1e9)
    view = PopulationView({0: ClassPrior(f=5.0, B=1e4, sigma=10.0)})
    view.add_node(1, 0, D)

    async def one_round(round):
        loop = asyncio.get_event_loop()
        dispatch = loop.time()
        download_done = dispatch + 0.05                 # the model reaches the client
        t_ack = download_done + 0.05                    # its acknowledgement reaches the aggregator
        await asyncio.sleep(download_done - dispatch + E * D * c / true_f)
        timing = await net.transfer(1, 0, Z, delay=0.05)
        return est.Measurement(t_ack, timing.t_first, timing.t_last, censored=timing.ingress_utilization >= 0.9)

    async def rounds():
        for round in range(1, 6):
            m = await one_round(round)
            view.apply_record(StateRecord(1, round, 0.0, m), E, c, Z, 0.3, 0.1, 0.2)
            if round == 1:
                assert view.get(1).f == pytest.approx(true_f) and view.get(1).B == pytest.approx(true_up)
        return view.get(1)

    state = run(rounds)
    assert state.f == pytest.approx(true_f) and state.B == pytest.approx(true_up)


def test_censoring_triggers_when_ingress_saturated():
    """Test 11 (simulator part): a saturated ingress marks the sample as censored and B is not updated."""
    net = FairNetwork(3, small_message_bytes=0)
    net.set_capacity(0, up=1e9, down=100.0)
    net.set_capacity(1, up=80.0, down=1e9)
    net.set_capacity(2, up=80.0, down=1e9)

    async def scenario():
        return await asyncio.gather(net.transfer(1, 0, 500, 0.0), net.transfer(2, 0, 500, 0.0))

    a, b = run(scenario)
    assert a.ingress_utilization >= 0.9 and b.ingress_utilization >= 0.9
    view = PopulationView({0: ClassPrior(f=5.0, B=80.0, sigma=1.0)})
    view.add_node(1, 0, 100)
    m = est.Measurement(0.0, 1.0, a.t_last, censored=a.ingress_utilization >= 0.9)
    view.apply_record(StateRecord(1, 1, 0.0, m), 1, 1.0, 500, 0.3, 0.1, 0.2)
    assert view.get(1).B == 80.0 and view.get(1).f != 5.0


def test_latency_matrix_cleaning():
    raw = [[0.0, 10.0, -1.0, 5.0],
           [12.0, 0.0, 7.0, -0.5],
           [-2.0, 7.0, 0.0, 3.0],
           [5.0, -1.0, 3.0, 0.0]]
    W, valid, stats = clean_matrix(raw)
    assert W[0, 1] == 11.0                    # both valid -> mean
    assert W[1, 2] == 7.0
    assert math.isnan(W[0, 2])                # neither valid
    assert math.isnan(W[1, 3])
    assert stats["pairs_invalid"] == 2 and stats["pairs_both_valid"] == 4 and stats["entries_negative"] == 4
    assert len(valid) == 2 and stats["cities_excluded"]


def test_wondernetwork_and_fedscale_models():
    config = load_config(overrides=["network.N=30"])
    devices = DeviceModel(config.env.devices, 30, seed=1)
    assert devices.stats["fedscale_profiles_sampled"] == 30
    assert devices.stats["fedscale_profiles_total"] == 500000
    assert sorted(set(d.kappa for d in devices.devices)) == [0, 1, 2]
    latency = LatencyModel(config.env.latency, [d.kappa for d in devices.devices], seed=1)
    assert latency.stats["cities_valid"] + len(latency.stats["cities_excluded"]) == 227
    assert np.all(latency.rtt[~np.eye(30, dtype=bool)] > 0)
    again = DeviceModel(config.env.devices, 30, seed=1)
    assert [d.f for d in again.devices] == [d.f for d in devices.devices]
    assert devices.true_f(3, 7) == again.true_f(3, 7)
    priors = devices.priors(E=1, D={i: 500 for i in range(30)}, Z=1e6)
    assert priors[0].f <= priors[2].f


def test_public_prior_without_declared_classes():
    """Without declared classes there is one prior (key 0) pooled over all devices, wider than any class prior."""
    config = load_config(overrides=["network.N=60"])
    devices = DeviceModel(config.env.devices, 60, seed=1)
    D = {i: 500 for i in range(60)}
    per_class = devices.priors(E=1, D=D, Z=1e6, declared=True)
    public = devices.priors(E=1, D=D, Z=1e6, declared=False)
    assert set(per_class) == {0, 1, 2} and set(public) == {0}
    assert per_class[0].f <= public[0].f <= per_class[2].f
    assert public[0].sigma > max(p.sigma for p in per_class.values()) * 0.5


def test_dataset_size_cap_in_priors():
    """The protocol predicts with min(D, D_max): a cap below every D shortens the prior prediction."""
    config = load_config(overrides=["network.N=30", "env.devices.sigma_prior=rel"])
    devices = DeviceModel(config.env.devices, 30, seed=1)
    D = {i: 1000 for i in range(30)}
    uncapped = devices.priors(E=1, D=D, Z=1.0, declared=False)
    capped = devices.priors(E=1, D=D, Z=1.0, declared=False, D_max=100)
    assert capped[0].sigma == pytest.approx(uncapped[0].sigma / 10, rel=1e-3)


def test_cancelled_transfer_releases_the_link():
    """A sender that gives up (handover timeout) stops sending: the other flow gets the whole uplink."""
    net = FairNetwork(3, small_message_bytes=0)
    net.set_capacity(0, up=100.0, down=100.0)
    net.set_capacity(1, up=1000.0, down=1000.0)
    net.set_capacity(2, up=1000.0, down=1000.0)

    async def scenario():
        abandoned = asyncio.ensure_future(net.transfer(0, 1, 10_000, delay=0.0))
        kept = asyncio.ensure_future(net.transfer(0, 2, 1_000, delay=0.0))
        await asyncio.sleep(2.0)      # both at 50 B/s: 100 B of the kept flow sent
        abandoned.cancel()
        timing = await kept
        return timing, net.flows

    timing, flows = run(scenario)
    # After the cancel the kept flow sends its remaining 900 B at 100 B/s: done at 2 + 9 = 11 s
    assert timing.ok and timing.t_last == pytest.approx(11.0) and not flows
