"""
End-to-end runs of the protocol (small, training bypassed). Each test runs run.py in a subprocess.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = ["--config", os.path.join(ROOT, "configs", "smoke_bypass.yaml"), "run.rounds=3"]

pytestmark = pytest.mark.slow


def run(tmp_path, name, *overrides):
    run_dir = str(tmp_path / name)
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=ROOT + os.pathsep + os.path.join(ROOT, "pyipv8"))
    result = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), *BASE, *overrides, "--run-dir", run_dir],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return run_dir


def read(run_dir, stream):
    path = os.path.join(run_dir, stream + ".jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def decisions(run_dir):
    return [(r["round"], r["aggregator"], r["clients"], r["duration"], r["selection"].get("utility"),
             r["client_detail"]) for r in read(run_dir, "rounds")]


def test_determinism(tmp_path):
    """Test 13: same config and seed -> identical selection logs."""
    a = run(tmp_path, "a")
    b = run(tmp_path, "b")
    assert decisions(a) == decisions(b) and len(decisions(a)) == 3
    strip = lambda rows: [{k: v for k, v in r.items() if k != "solve_time"} for r in rows]
    assert strip(read(a, "proposals")) == strip(read(b, "proposals"))
    c = run(tmp_path, "c", "seed=7")
    assert decisions(c) != decisions(a)


def test_next_ranked_proposal_when_next_aggregator_fails(tmp_path):
    """Test 9a: the selected A(t+1) crashes on the handover -> A(t) hands over to the next-ranked proposal."""
    # A(1) must hand over to another node, otherwise there is no handover message for the failure to hit
    d = run(tmp_path, "handover_fail", "env.failures.scripted=[{round: 2, target: aggregator, when: handover}]",
            "protocol.forbid_repeat_aggregator=true")
    events = read(d, "events")
    assert any(e["kind"] == "handover_no_ack" and e["round"] == 2 for e in events)
    round2 = [r for r in read(d, "rounds") if r["round"] == 2][0]
    assert round2["selection"]["handover_rank"] >= 1
    assert any(m["kind"] == "crash" for m in read(d, "membership"))


def test_watchdog_fallback_when_aggregator_dies_mid_round(tmp_path):
    """Test 9a': A(t+1) acknowledges, dispatches and dies -> watchdog timeout -> next-ranked proposal."""
    d = run(tmp_path, "mid_round", "env.failures.scripted=[{round: 2, target: aggregator, when: dispatch, delay: 20}]",
            "fault.watchdog_timeout=400")
    events = read(d, "events")
    assert any(e["kind"] == "watchdog_timeout" and e["round"] == 2 for e in events)
    rounds = {r["round"]: r for r in read(d, "rounds")}
    assert 2 in rounds and rounds[2]["selection"]["handover_rank"] >= 1


def test_resample_then_self_serve_without_proposals(tmp_path):
    """Test 9b: no eligible candidate -> resample with the next hash index (twice) -> serve the round itself."""
    d = run(tmp_path, "no_proposals", "protocol.f_min=1e12", "run.rounds=2")
    selections = read(d, "selections")
    first = selections[0]
    assert len(first["attempts"]) == 3 and [a["k"] for a in first["attempts"]] == [0, 1, 2]
    assert first["self_serve"] and first["resamples"] == 2
    rounds = read(d, "rounds")
    assert rounds[0]["aggregator"] == first["selected_by"]


@pytest.mark.parametrize("selector", ["compass_random", "power_of_choice", "oort_like", "oracle_central"])
def test_selectors_run(tmp_path, selector):
    d = run(tmp_path, selector, "selector=%s" % selector, "run.rounds=2")
    rounds = read(d, "rounds")
    assert len(rounds) == 2 and all(len(r["clients"]) == 4 for r in rounds)


def test_plexus_port_runs(tmp_path):
    d = run(tmp_path, "plexus", "selector=plexus", "data.synthetic_train_size=6000", "run.rounds=3")
    rounds = read(d, "rounds")
    assert len(rounds) == 3 and all(r["selector"] == "plexus" for r in rounds)


@pytest.mark.parametrize("policy", ["wait_all", "fixed_deadline", "over_selection"])
def test_closing_policies_run(tmp_path, policy):
    d = run(tmp_path, policy, "policy=%s" % policy, "run.rounds=2", "closing.delta_dl=300")
    rounds = read(d, "rounds")
    assert len(rounds) == 2 and rounds[0]["policy"] == policy
    expected = 6 if policy == "over_selection" else 4  # ceil(1.3 * 4) = 6
    assert all(len(r["clients"]) == expected for r in rounds)


def test_ckks_mode_runs(tmp_path):
    d = run(tmp_path, "ckks", "he=ckks", "network.N=10", "protocol.M=8", "protocol.k_nb=6", "run.rounds=2")
    assert len(read(d, "rounds")) == 2
    summary = read(d, "summary")[0]
    assert summary["he_stats"]["num_respond"] >= 10 * 9


def test_churn_and_removal(tmp_path):
    """Joins trigger the similarity protocol; crashed nodes are removed after max_missed misses."""
    # A(t) selecting itself and then crashing is a single point of failure of the protocol (nobody else holds theta),
    # so this test forbids repeated aggregators; crashed nodes come back after 300 s.
    d = run(tmp_path, "churn", "network.initially_active=13", "env.failures.joins=[[50, 3]]",
            "env.failures.crash_prob=[0.15, 0.15, 0.15]", "env.failures.crash_downtime=300", "fault.max_missed=1",
            "protocol.forbid_repeat_aggregator=true", "run.rounds=5")
    kinds = [m["kind"] for m in read(d, "membership")]
    assert kinds.count("join") == 3
    assert "crash" in kinds and "removed_after_misses" in kinds and "rejoin" in kinds


def test_prefetch_and_snapshot_consistency(tmp_path):
    """Prefetched probes are reused; late records during selection never cause verifier rejections (test 12)."""
    d = run(tmp_path, "prefetch", "protocol.probing.prefetch=true", "run.rounds=4")
    rounds = read(d, "rounds")
    assert len(rounds) == 4
    assert any(e["kind"] == "late_record_forwarded" for e in read(d, "events")) or \
        any(r["late"] or r["missing"] for r in rounds)
    assert all(s.get("rejected", 0) == 0 for s in read(d, "selections"))
    assert sum(r["traffic"]["probe"]["messages"] for r in rounds) > 0


def test_client_backups_option(tmp_path):
    d = run(tmp_path, "backups", "fault.client_backups=true", "fault.watchdog_timeout=400",
            "env.failures.scripted=[{round: 2, target: aggregator, when: dispatch, delay: 20}]")
    rounds = {r["round"]: r for r in read(d, "rounds")}
    assert 2 in rounds and rounds[2]["selection"]["handover_rank"] >= 1


def test_plexus_port_matches_original_participants(tmp_path):
    """Sanity check: the ported Plexus logic picks the same participants per round as the original simulation."""
    from binascii import hexlify
    from collections import defaultdict
    from dlsim.compass.env.world import node_public_key

    d = run(tmp_path, "plexus_cmp", "selector=plexus", "data.synthetic_train_size=6000", "run.rounds=5")
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=ROOT + os.pathsep + os.path.join(ROOT, "pyipv8"))
    work = tmp_path / "original"
    work.mkdir()
    for name in ("data",):
        os.symlink(os.path.join(ROOT, name), str(work / name))
    # The original Plexus code calls asyncio.get_event_loop() before creating its loop, which Python >= 3.12 no longer
    # allows without a current loop; set one first (the original code itself is left untouched).
    script = os.path.join(ROOT, "simulations", "plexus", "cifar10.py")
    launcher = ("import asyncio, runpy, sys; asyncio.set_event_loop(asyncio.new_event_loop()); "
                "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
    result = subprocess.run([sys.executable, "-c", launcher, script, "--peers", "16",
                             "--sample-size", "4", "--bypass-training", "--bypass-model-transfers", "--rounds", "5",
                             "--duration", "0", "--seed", "42", "--log-level", "ERROR",
                             "--accuracy-logging-interval", "0"], cwd=str(work), env=env, capture_output=True,
                            text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    out_dir = [os.path.join(str(work), "data", x) for x in os.listdir(str(work / "data")) if x.endswith("dfl")]
    short = {hexlify(node_public_key(i)).decode()[-8:]: i for i in range(16)}
    original = defaultdict(set)
    try:
        with open(os.path.join(out_dir[0], "derived_samples.csv")) as handle:
            for line in handle.readlines()[1:]:
                _, sample_id, sample = line.strip().split(",")
                ids = sample.split("-")
                if len(ids) == 4:
                    original[int(sample_id)].add(tuple(sorted(short[x] for x in ids)))
    finally:
        import shutil
        for path in out_dir:
            shutil.rmtree(path, ignore_errors=True)
    ours = {r["round"]: tuple(r["clients"]) for r in read(d, "rounds")}
    compared = [r for r in ours if r >= 2 and r in original]
    assert len(compared) >= 3
    assert all(ours[r] in original[r] for r in compared)


def test_availability_traces_option(tmp_path):
    import pickle
    traces = {key: {"active": [0, 400.0], "inactive": [100.0 + key], "finish_time": 1000} for key in range(40)}
    path = tmp_path / "availability.pkl"
    with open(path, "wb") as handle:
        pickle.dump(traces, handle)
    d = run(tmp_path, "availability", "env.failures.availability_traces=%s" % path, "run.rounds=3",
            "protocol.forbid_repeat_aggregator=true")
    kinds = [m["kind"] for m in read(d, "membership")]
    assert kinds.count("crash") >= 1 and "rejoin" in kinds


def test_declared_classes_and_size_cap_options(tmp_path):
    """The declared-class option and an aggressive dataset-size cap both run end to end."""
    run_dir = run(tmp_path, "declared", "protocol.declared_class=true", "protocol.D_max=10")
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    assert meta["environment"]["devices"]["declared_class"] is True
    assert len(read(run_dir, "rounds")) == 3
